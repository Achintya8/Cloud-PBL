"""
ml_engine/train_predict.py
==========================
LightGBM-based time-series price forecasting engine.

Pipeline:
  1. Load historical price + weather data from RDS
  2. Run feature engineering
  3. Train LightGBM regression model per variety × horizon combination
  4. Generate 7-day and 30-day price forecasts with confidence intervals
     (using quantile regression for upper/lower bounds)
  5. Write predictions to price_predictions table
  6. Save trained model artifacts to disk
"""

import json
import os
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.logging_config import get_logger
from config.settings import ml as ml_cfg
from database.db_manager import db_manager
from ml_engine.feature_engineering import ArecaFeatureEngineer

logger = get_logger(__name__, log_file="/var/log/areca/ml_engine.log")


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class ArecaDataLoader:
    """Loads and joins price + weather data from RDS into a training DataFrame."""

    PRICE_QUERY = """
        SELECT
            mp.record_date,
            m.market_name,
            m.district,
            v.variety_name,
            mp.min_price,
            mp.max_price,
            mp.modal_price,
            mp.arrivals_tons,
            mp.source
        FROM market_prices mp
        JOIN markets   m ON m.market_id  = mp.market_id
        JOIN varieties v ON v.variety_id = mp.variety_id
        WHERE mp.record_date >= %s
          AND mp.record_date <= %s
          AND v.variety_name = %s
        ORDER BY mp.record_date ASC, m.market_name ASC
    """

    WEATHER_QUERY = """
        SELECT
            w.record_date,
            m.market_name,
            w.rainfall_mm,
            w.avg_humidity,
            w.temperature_c,
            w.wind_speed_kmh
        FROM weather_metrics w
        JOIN markets m ON m.market_id = w.region_id
        WHERE w.record_date >= %s
          AND w.record_date <= %s
        ORDER BY w.record_date ASC, m.market_name ASC
    """

    def load_variety_data(
        self,
        variety: str,
        lookback_days: int = 730,
        market: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load and merge price + weather data for a specific variety.

        Args:
            variety:       Variety name (e.g., 'Chali').
            lookback_days: How many calendar days of history to load.
            market:        If specified, filter to a single market; else aggregate.

        Returns:
            DataFrame with price and weather columns, indexed by date.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info("Loading data from RDS", variety=variety, start=str(start_date))

        # --- Price data ---
        price_rows = db_manager.execute_query(
            self.PRICE_QUERY, (start_date, end_date, variety)
        )
        if not price_rows:
            raise ValueError(f"No price data found for variety '{variety}'")

        price_df = pd.DataFrame(price_rows)
        price_df["record_date"] = pd.to_datetime(price_df["record_date"])

        if market:
            price_df = price_df[price_df["market_name"].str.lower() == market.lower()]

        # Aggregate across markets: use volume-weighted average if arrivals available
        # else use simple median
        if price_df["arrivals_tons"].notna().any():
            price_df["arrivals_tons"] = price_df["arrivals_tons"].fillna(1.0)
            agg_df = (
                price_df.groupby("record_date")
                .apply(
                    lambda g: pd.Series({
                        "modal_price": np.average(
                            g["modal_price"], weights=g["arrivals_tons"]
                        ),
                        "min_price":    g["min_price"].min(),
                        "max_price":    g["max_price"].max(),
                        "arrivals_tons": g["arrivals_tons"].sum(),
                        "variety_name":  g["variety_name"].iloc[0],
                    })
                )
                .reset_index()
            )
        else:
            agg_df = (
                price_df.groupby("record_date")
                .agg(
                    modal_price=("modal_price", "median"),
                    min_price=("min_price", "min"),
                    max_price=("max_price", "max"),
                    arrivals_tons=("arrivals_tons", "sum"),
                    variety_name=("variety_name", "first"),
                )
                .reset_index()
            )

        # --- Weather data ---
        weather_rows = db_manager.execute_query(
            self.WEATHER_QUERY, (start_date, end_date)
        )
        weather_df = pd.DataFrame(weather_rows) if weather_rows else pd.DataFrame()

        if not weather_df.empty:
            weather_df["record_date"] = pd.to_datetime(weather_df["record_date"])
            weather_agg = (
                weather_df.groupby("record_date")
                .agg(
                    rainfall_mm=("rainfall_mm", "mean"),
                    avg_humidity=("avg_humidity", "mean"),
                    temperature_c=("temperature_c", "mean"),
                )
                .reset_index()
            )
            agg_df = agg_df.merge(weather_agg, on="record_date", how="left")

        # Ensure continuous date range (fill gaps with forward-fill)
        full_range = pd.date_range(start=agg_df["record_date"].min(), end=agg_df["record_date"].max(), freq="D")
        agg_df = (
            agg_df.set_index("record_date")
            .reindex(full_range)
            .rename_axis("record_date")
            .reset_index()
        )
        agg_df["variety_name"] = variety

        # Forward-fill price gaps (market closures), then backfill
        for col in ["modal_price", "min_price", "max_price"]:
            agg_df[col] = agg_df[col].fillna(method="ffill").fillna(method="bfill")

        logger.info(
            "Data loaded",
            variety=variety,
            rows=len(agg_df),
            date_range=f"{agg_df['record_date'].min().date()} → {agg_df['record_date'].max().date()}",
        )
        return agg_df


# ---------------------------------------------------------------------------
# LightGBM Model Trainer
# ---------------------------------------------------------------------------

class LGBMForecaster:
    """
    Trains separate LightGBM models for each variety × forecast horizon.

    Confidence intervals are estimated using quantile regression with
    alpha/2 and 1-alpha/2 quantile models.
    """

    def __init__(self, variety: str, horizon: int):
        self.variety = variety
        self.horizon = horizon
        self.model: Optional[lgb.Booster] = None
        self.lower_model: Optional[lgb.Booster] = None
        self.upper_model: Optional[lgb.Booster] = None
        self.feature_engineer = ArecaFeatureEngineer()
        self.train_metrics: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # LightGBM dataset & parameter builders
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_params(objective: str = "regression", alpha: Optional[float] = None) -> Dict:
        """Build LightGBM parameters."""
        params: Dict[str, Any] = {
            "objective": objective,
            "metric": "rmse",
            "num_leaves": ml_cfg.num_leaves,
            "learning_rate": ml_cfg.learning_rate,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }
        if objective == "quantile" and alpha is not None:
            params["alpha"] = alpha
        return params

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Train three models:
          1. Main point-estimate regressor
          2. Lower quantile (alpha/2)
          3. Upper quantile (1 - alpha/2)

        Returns training metrics dict.
        """
        alpha = ml_cfg.confidence_alpha

        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_set   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_names, reference=train_set)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        logger.info(
            "Training LightGBM point estimator",
            variety=self.variety,
            horizon=self.horizon,
            train_rows=len(X_train),
        )
        self.model = lgb.train(
            params=self._build_params("regression"),
            train_set=train_set,
            num_boost_round=ml_cfg.n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # Lower quantile model
        self.lower_model = lgb.train(
            params=self._build_params("quantile", alpha=alpha / 2),
            train_set=train_set,
            num_boost_round=ml_cfg.n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # Upper quantile model
        self.upper_model = lgb.train(
            params=self._build_params("quantile", alpha=1.0 - alpha / 2),
            train_set=train_set,
            num_boost_round=ml_cfg.n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

        # Compute validation metrics
        val_preds = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        mae  = float(mean_absolute_error(y_val, val_preds))
        mape = float(np.mean(np.abs((y_val - val_preds) / np.maximum(y_val, 1e-6))) * 100)

        self.train_metrics = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape}

        # Feature importance
        importance_vals = self.model.feature_importance(importance_type="gain")
        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(feature_names, importance_vals)
        }

        logger.info(
            "Training complete",
            variety=self.variety,
            horizon=self.horizon,
            val_rmse=round(rmse, 2),
            val_mae=round(mae, 2),
            val_mape_pct=round(mape, 2),
        )
        return self.train_metrics

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.

        Returns:
            (predicted_prices, lower_bounds, upper_bounds)
        """
        if not all([self.model, self.lower_model, self.upper_model]):
            raise RuntimeError("Model not trained. Call train() first.")

        predicted = self.model.predict(X)
        lower     = self.lower_model.predict(X)
        upper     = self.upper_model.predict(X)

        # Enforce lower <= predicted <= upper
        lower = np.minimum(lower, predicted)
        upper = np.maximum(upper, predicted)

        # Floor at 0 (prices cannot be negative)
        predicted = np.maximum(predicted, 0)
        lower     = np.maximum(lower, 0)
        upper     = np.maximum(upper, 0)

        return predicted, lower, upper

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, model_dir: str) -> Dict[str, str]:
        """Save all three model artifacts and the feature engineer scaler."""
        paths: Dict[str, str] = {}
        base = Path(model_dir) / self.variety / f"h{self.horizon}"
        base.mkdir(parents=True, exist_ok=True)

        model_path = base / "model_point.pkl"
        lower_path = base / "model_lower.pkl"
        upper_path = base / "model_upper.pkl"
        fe_path    = base / "feature_engineer.pkl"

        for path, obj in [
            (model_path, self.model),
            (lower_path, self.lower_model),
            (upper_path, self.upper_model),
            (fe_path, self.feature_engineer),
        ]:
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            paths[path.stem] = str(path)

        # Save feature importance JSON
        fi_path = base / "feature_importance.json"
        with open(fi_path, "w") as f:
            json.dump(
                dict(sorted(self.feature_importance.items(), key=lambda x: -x[1])),
                f, indent=2
            )
        paths["feature_importance"] = str(fi_path)

        logger.info("Model saved", variety=self.variety, horizon=self.horizon, paths=paths)
        return paths

    def load(self, model_dir: str) -> None:
        """Load all model artifacts from disk."""
        base = Path(model_dir) / self.variety / f"h{self.horizon}"

        def _load(filename: str):
            path = base / filename
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        self.model           = _load("model_point.pkl")
        self.lower_model     = _load("model_lower.pkl")
        self.upper_model     = _load("model_upper.pkl")
        self.feature_engineer = _load("feature_engineer.pkl")

        logger.info("Model loaded", variety=self.variety, horizon=self.horizon)


# ---------------------------------------------------------------------------
# Prediction Writer
# ---------------------------------------------------------------------------

class PredictionWriter:
    """Writes model forecast results back to the price_predictions RDS table."""

    def write(
        self,
        predictions: List[Dict[str, Any]],
        run_id: str,
        model_version: str,
        metrics: Dict[str, float],
        feature_importance: Dict[str, float],
        horizon: int,
    ) -> int:
        """
        Upsert prediction rows into price_predictions.

        Args:
            predictions: List of dicts with keys:
                prediction_date, target_date, market_id, variety_id,
                predicted_price, confidence_lower, confidence_upper
            run_id:              UUID for the training run.
            model_version:       Semver string, e.g. '1.2.0'.
            metrics:             Dict with val_rmse, val_mae.
            feature_importance:  Top feature importances.
            horizon:             Forecast horizon in days.

        Returns:
            Number of rows written.
        """
        if not predictions:
            return 0

        rows = []
        fi_json = json.dumps(
            dict(list(sorted(feature_importance.items(), key=lambda x: -x[1]))[:20])
        )

        for pred in predictions:
            rows.append((
                run_id,
                pred["prediction_date"],
                pred["target_date"],
                pred["market_id"],
                pred["variety_id"],
                round(pred["predicted_price"], 2),
                round(pred["confidence_lower"], 2),
                round(pred["confidence_upper"], 2),
                horizon,
                model_version,
                metrics.get("val_rmse"),
                metrics.get("val_mae"),
                fi_json,
            ))

        written = db_manager.bulk_insert(
            table="price_predictions",
            columns=[
                "prediction_run_id", "prediction_date", "target_date",
                "market_id", "variety_id",
                "predicted_price", "confidence_lower", "confidence_upper",
                "horizon_days", "model_version", "model_rmse", "model_mae",
                "feature_importance",
            ],
            rows=rows,
            on_conflict=(
                "(prediction_date, target_date, market_id, variety_id, horizon_days) "
                "DO UPDATE SET "
                "  predicted_price  = EXCLUDED.predicted_price, "
                "  confidence_lower = EXCLUDED.confidence_lower, "
                "  confidence_upper = EXCLUDED.confidence_upper, "
                "  model_version    = EXCLUDED.model_version, "
                "  model_rmse       = EXCLUDED.model_rmse, "
                "  model_mae        = EXCLUDED.model_mae"
            ),
        )
        logger.info("Predictions written", count=written, horizon=horizon)
        return written


# ---------------------------------------------------------------------------
# Main Training & Prediction Orchestrator
# ---------------------------------------------------------------------------

class MLOrchestrator:
    """
    Coordinates the full train → predict → save → write cycle.
    """

    MODEL_VERSION = "1.0.0"

    def __init__(self):
        self.data_loader = ArecaDataLoader()
        self.pred_writer = PredictionWriter()
        model_dir = os.path.dirname(ml_cfg.model_path)
        self.model_dir = model_dir or "/opt/areca/models"

    def _get_market_variety_ids(
        self, variety: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """Look up the primary market_id and variety_id for writing predictions."""
        variety_row = db_manager.execute_query(
            "SELECT variety_id FROM varieties WHERE variety_name = %s", (variety,)
        )
        if not variety_row:
            return None, None
        variety_id = variety_row[0]["variety_id"]

        # Use first active market as representative (predictions are variety-level)
        market_row = db_manager.execute_query(
            "SELECT market_id::text FROM markets WHERE active = TRUE LIMIT 1"
        )
        market_id = market_row[0]["market_id"] if market_row else None
        return market_id, variety_id

    def run_variety(self, variety: str) -> Dict[str, Any]:
        """
        Full training + prediction pipeline for one variety.

        Returns:
            Dict with status, metrics, and prediction counts.
        """
        logger.info("Starting ML pipeline", variety=variety)
        run_id = str(__import__("uuid").uuid4())
        results: Dict[str, Any] = {"variety": variety, "horizons": {}}

        try:
            df = self.data_loader.load_variety_data(variety)
        except ValueError as exc:
            logger.error("Data loading failed", variety=variety, error=str(exc))
            return {"variety": variety, "status": "no_data", "error": str(exc)}

        if len(df) < ml_cfg.min_training_rows:
            logger.warning(
                "Insufficient data for training",
                variety=variety,
                rows=len(df),
                required=ml_cfg.min_training_rows,
            )
            return {"variety": variety, "status": "insufficient_data", "rows": len(df)}

        market_id, variety_id = self._get_market_variety_ids(variety)
        if not variety_id:
            return {"variety": variety, "status": "variety_not_found"}

        for horizon in ml_cfg.forecast_horizons:
            logger.info("Training for horizon", variety=variety, horizon=horizon)
            forecaster = LGBMForecaster(variety=variety, horizon=horizon)

            # Feature engineering
            (
                X_train, y_train, X_val, y_val, feature_names
            ) = forecaster.feature_engineer.fit_transform(
                df, horizon=horizon, val_split_ratio=ml_cfg.test_split_ratio
            )

            # Train
            metrics = forecaster.train(X_train, y_train, X_val, y_val, feature_names)

            # Save models
            forecaster.save(self.model_dir)

            # Generate future predictions
            predictions = self._generate_future_predictions(
                df=df,
                forecaster=forecaster,
                horizon=horizon,
                market_id=market_id,
                variety_id=variety_id,
            )

            # Write predictions to DB
            written = self.pred_writer.write(
                predictions=predictions,
                run_id=run_id,
                model_version=self.MODEL_VERSION,
                metrics=metrics,
                feature_importance=forecaster.feature_importance,
                horizon=horizon,
            )

            results["horizons"][horizon] = {
                "val_rmse": metrics.get("val_rmse"),
                "val_mae":  metrics.get("val_mae"),
                "predictions_written": written,
            }
            logger.info(
                "Horizon complete",
                variety=variety,
                horizon=horizon,
                **metrics,
            )

        # Log training run to DB
        self._log_training_run(
            run_id=run_id,
            variety=variety,
            rows_trained=len(df),
            results=results,
        )

        results["status"] = "success"
        return results

    def _generate_future_predictions(
        self,
        df: pd.DataFrame,
        forecaster: LGBMForecaster,
        horizon: int,
        market_id: Optional[str],
        variety_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Iteratively generate predictions for each future day in the horizon.
        Uses a recursive multi-step approach: prediction for t+1 feeds into
        the input for t+2, etc.
        """
        predictions = []
        prediction_date = date.today()

        working_df = df.copy()

        for step in range(1, horizon + 1):
            target_date = prediction_date + timedelta(days=step)

            try:
                # Build features from the latest available data
                X, _ = forecaster.feature_engineer.transform(working_df, horizon=1)
                if len(X) == 0:
                    break

                X_last = X[-1:, :]
                pred, lower, upper = forecaster.predict(X_last)

                predicted_price = float(pred[0])
                lower_price     = float(lower[0])
                upper_price     = float(upper[0])

                predictions.append({
                    "prediction_date": prediction_date,
                    "target_date":     target_date,
                    "market_id":       market_id,
                    "variety_id":      variety_id,
                    "predicted_price": predicted_price,
                    "confidence_lower": lower_price,
                    "confidence_upper": upper_price,
                })

                # Append predicted value as a new row for the next iteration
                new_row = working_df.iloc[-1:].copy()
                new_row["record_date"] = pd.Timestamp(target_date)
                new_row["modal_price"] = predicted_price
                new_row["min_price"]   = lower_price
                new_row["max_price"]   = upper_price
                working_df = pd.concat([working_df, new_row], ignore_index=True)

            except Exception as exc:
                logger.warning(
                    "Prediction failed for step",
                    step=step,
                    target=str(target_date),
                    error=str(exc),
                )
                continue

        logger.info(
            "Future predictions generated",
            variety=forecaster.variety,
            horizon=horizon,
            count=len(predictions),
        )
        return predictions

    def _log_training_run(
        self,
        run_id: str,
        variety: str,
        rows_trained: int,
        results: Dict,
    ) -> None:
        """Write a training run audit record to the database."""
        variety_row = db_manager.execute_query(
            "SELECT variety_id FROM varieties WHERE variety_name = %s", (variety,)
        )
        variety_id = variety_row[0]["variety_id"] if variety_row else None

        first_horizon_results = list(results.get("horizons", {}).values())
        val_rmse = first_horizon_results[0].get("val_rmse") if first_horizon_results else None

        try:
            db_manager.execute_query(
                """
                INSERT INTO model_training_runs
                    (run_id, variety_id, completed_at, status, rows_trained, val_rmse, model_path, hyperparameters)
                VALUES (%s, %s, NOW(), 'success', %s, %s, %s, %s)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (
                    run_id,
                    variety_id,
                    rows_trained,
                    val_rmse,
                    self.model_dir,
                    json.dumps({
                        "n_estimators": ml_cfg.n_estimators,
                        "learning_rate": ml_cfg.learning_rate,
                        "num_leaves": ml_cfg.num_leaves,
                    }),
                ),
            )
        except Exception as exc:
            logger.warning("Failed to log training run", error=str(exc))

    def run_all_varieties(self) -> Dict[str, Any]:
        """Train and predict for all configured areca nut varieties."""
        db_manager.initialize()
        all_results: Dict[str, Any] = {}

        for variety in ["Chali", "Gotu", "Kotte"]:  # Focus on primary traded varieties
            try:
                result = self.run_variety(variety)
                all_results[variety] = result
            except Exception as exc:
                logger.error("Variety pipeline failed", variety=variety, error=str(exc), exc_info=True)
                all_results[variety] = {"status": "error", "error": str(exc)}

        logger.info("All varieties processed", summary=all_results)
        return all_results


# ---------------------------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------------------------

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for the ML training + prediction job."""
    logger.info("Lambda ML handler invoked", event=event)
    orchestrator = MLOrchestrator()
    variety = event.get("variety")

    if variety:
        result = orchestrator.run_variety(variety)
    else:
        result = orchestrator.run_all_varieties()

    return {
        "statusCode": 200,
        "body": json.dumps(result, default=str),
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Areca Nut ML Training & Prediction")
    parser.add_argument("--variety", type=str, default=None, help="Variety to train (all if omitted)")
    args = parser.parse_args()

    orchestrator = MLOrchestrator()
    if args.variety:
        result = orchestrator.run_variety(args.variety)
    else:
        result = orchestrator.run_all_varieties()

    print(json.dumps(result, indent=2, default=str))
