"""
ml_engine/feature_engineering.py
=================================
Feature engineering for the areca nut price forecasting model.

Creates:
  - Lag features (t-1, t-3, t-7, t-14, t-21, t-30)
  - Rolling statistics (mean, std, min, max) for windows [7, 14, 30]
  - Date/calendar features (day of week, week of year, month, quarter)
  - Weather features (rainfall, humidity, temperature)
  - Price momentum indicators
  - Arrivals-based supply/demand features
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from config.logging_config import get_logger
from config.settings import ml as ml_cfg

logger = get_logger(__name__)


class ArecaFeatureEngineer:
    """
    Builds the full feature matrix for the LightGBM time-series model.

    Usage:
        fe = ArecaFeatureEngineer()
        X_train, y_train, X_val, y_val = fe.fit_transform(df, horizon=7)
        X_predict = fe.transform(df_latest, horizon=7)
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self._fitted = False

    # -----------------------------------------------------------------------
    # Core feature creation
    # -----------------------------------------------------------------------

    @staticmethod
    def create_lag_features(df: pd.DataFrame, price_col: str = "modal_price") -> pd.DataFrame:
        """Add lag price features for t-1, t-3, t-7, t-14, t-21, t-30 days."""
        for lag in ml_cfg.lag_days:
            df[f"lag_{lag}d"] = df[price_col].shift(lag)
        return df

    @staticmethod
    def create_rolling_features(
        df: pd.DataFrame, price_col: str = "modal_price"
    ) -> pd.DataFrame:
        """Add rolling window statistics (mean, std, min, max)."""
        for window in ml_cfg.rolling_windows:
            shifted = df[price_col].shift(1)  # avoid data leakage
            df[f"roll_mean_{window}d"] = shifted.rolling(window, min_periods=max(2, window // 2)).mean()
            df[f"roll_std_{window}d"]  = shifted.rolling(window, min_periods=max(2, window // 2)).std()
            df[f"roll_min_{window}d"]  = shifted.rolling(window, min_periods=max(2, window // 2)).min()
            df[f"roll_max_{window}d"]  = shifted.rolling(window, min_periods=max(2, window // 2)).max()
            df[f"roll_range_{window}d"] = df[f"roll_max_{window}d"] - df[f"roll_min_{window}d"]
        return df

    @staticmethod
    def create_calendar_features(df: pd.DataFrame, date_col: str = "record_date") -> pd.DataFrame:
        """Add calendar-based cyclical and ordinal features."""
        dt = pd.to_datetime(df[date_col])
        df["day_of_week"]   = dt.dt.dayofweek          # 0=Mon...6=Sun
        df["day_of_month"]  = dt.dt.day
        df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
        df["month"]         = dt.dt.month
        df["quarter"]       = dt.dt.quarter
        df["year"]          = dt.dt.year
        df["is_weekend"]    = (dt.dt.dayofweek >= 5).astype(int)
        df["is_month_start"] = dt.dt.is_month_start.astype(int)
        df["is_month_end"]   = dt.dt.is_month_end.astype(int)

        # Cyclical encoding for week_of_year (avoids week 1 / week 52 discontinuity)
        df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
        df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

        # Cyclical encoding for month
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

        # Karnataka areca nut harvest seasonality flags
        # Main harvest: Dec–Feb (Karthik season), secondary: Jun–Aug
        df["is_harvest_season"] = (
            df["month"].isin([12, 1, 2, 6, 7, 8])
        ).astype(int)
        df["is_festival_season"] = (
            df["month"].isin([10, 11])  # Diwali, Navratri — peak demand
        ).astype(int)
        return df

    @staticmethod
    def create_momentum_features(
        df: pd.DataFrame, price_col: str = "modal_price"
    ) -> pd.DataFrame:
        """Add price momentum and rate-of-change features."""
        for period in [1, 7, 14, 30]:
            shifted = df[price_col].shift(period)
            df[f"pct_change_{period}d"] = (df[price_col] - shifted) / shifted.replace(0, np.nan)
            df[f"price_diff_{period}d"] = df[price_col] - shifted

        # Exponential moving averages
        for span in [7, 14, 30]:
            df[f"ema_{span}d"] = df[price_col].ewm(span=span, min_periods=span // 2).mean()

        # RSI-like oscillator (simplified)
        delta = df[price_col].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14d"] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def create_arrivals_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-demand indicator features from market arrivals."""
        if "arrivals_tons" in df.columns:
            df["arrivals_tons"] = df["arrivals_tons"].fillna(df["arrivals_tons"].median())
            for lag in [1, 7]:
                df[f"arrivals_lag_{lag}d"] = df["arrivals_tons"].shift(lag)
            for window in [7, 30]:
                df[f"arrivals_roll_{window}d"] = (
                    df["arrivals_tons"].shift(1).rolling(window, min_periods=2).mean()
                )
            # Price-to-arrivals ratio (higher arrivals often suppress price)
            df["price_arrivals_ratio"] = (
                df["modal_price"] / df["arrivals_tons"].replace(0, np.nan)
            )
        return df

    @staticmethod
    def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
        """Merge and lag weather features into price DataFrame."""
        for col in ["rainfall_mm", "avg_humidity", "temperature_c"]:
            if col in df.columns:
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                for lag in [1, 7, 14]:
                    df[f"{col}_lag_{lag}d"] = df[col].shift(lag)
                df[f"{col}_roll_30d"] = df[col].shift(1).rolling(30, min_periods=7).mean()

        return df

    @staticmethod
    def create_variety_encoding(df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode variety for use as a categorical feature."""
        variety_order = {v: i for i, v in enumerate(["Chali", "Gotu", "Kotte", "Rashi", "Saraku"])}
        df["variety_encoded"] = df["variety_name"].map(variety_order).fillna(-1).astype(int)
        return df

    # -----------------------------------------------------------------------
    # Target creation
    # -----------------------------------------------------------------------

    @staticmethod
    def create_target(df: pd.DataFrame, horizon: int, price_col: str = "modal_price") -> pd.DataFrame:
        """
        Create the forecast target column.
        Target = price `horizon` days ahead.
        """
        df[f"target_{horizon}d"] = df[price_col].shift(-horizon)
        return df

    # -----------------------------------------------------------------------
    # Full feature matrix pipeline
    # -----------------------------------------------------------------------

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        horizon: int,
        target_col: str = "modal_price",
    ) -> pd.DataFrame:
        """Apply all feature engineering steps in order."""
        df = df.copy().sort_values("record_date").reset_index(drop=True)

        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_calendar_features(df)
        df = self.create_momentum_features(df, target_col)
        df = self.create_arrivals_features(df)
        df = self.create_weather_features(df)
        df = self.create_variety_encoding(df)
        df = self.create_target(df, horizon, target_col)

        return df

    def get_feature_columns(self, df: pd.DataFrame, horizon: int) -> List[str]:
        """Return all feature columns, excluding target and ID columns."""
        exclude = {
            "record_date", "ingested_at", "market_name", "district",
            "variety_name", "source", "raw_payload", "market_id", "variety_id",
            f"target_{horizon}d", "modal_price",  # target leakage protection
        }
        return [c for c in df.columns if c not in exclude and not c.startswith("target_")]

    # -----------------------------------------------------------------------
    # Fit-transform / transform
    # -----------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        horizon: int,
        val_split_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Build features, split train/val, fit the scaler on train only.

        Returns:
            X_train, y_train, X_val, y_val, feature_names
        """
        full = self.build_feature_matrix(df, horizon)
        target_col = f"target_{horizon}d"

        # Drop rows where target or key lags are NaN
        min_lag = max(ml_cfg.lag_days)
        full = full.iloc[min_lag:]
        full = full.dropna(subset=[target_col])

        feature_cols = self.get_feature_columns(full, horizon)
        full = full.dropna(subset=feature_cols, how="all")

        X = full[feature_cols].fillna(-999)  # LightGBM handles -999 as "missing"
        y = full[target_col].values

        # Chronological split (no random shuffle — it's time series!)
        split_idx = int(len(X) * (1 - val_split_ratio))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled   = self.scaler.transform(X_val)

        self.feature_names = feature_cols
        self._fitted = True

        logger.info(
            "Feature matrix built",
            horizon=horizon,
            train_rows=len(X_train),
            val_rows=len(X_val),
            features=len(feature_cols),
        )
        return X_train_scaled, y_train, X_val_scaled, y_val, feature_cols

    def transform(
        self, df: pd.DataFrame, horizon: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply feature engineering + scaling for inference (no fitting)."""
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")

        full = self.build_feature_matrix(df, horizon)
        min_lag = max(ml_cfg.lag_days)
        full = full.iloc[min_lag:]

        feature_cols = self.feature_names  # use same columns as training
        X = full[feature_cols].fillna(-999)
        X_scaled = self.scaler.transform(X)
        return X_scaled, feature_cols

    def save_feature_list(self, path: str) -> None:
        """Persist feature column names to JSON for reproducibility."""
        with open(path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info("Feature list saved", path=path)

    def load_feature_list(self, path: str) -> None:
        """Load saved feature column names."""
        with open(path) as f:
            self.feature_names = json.load(f)
        self._fitted = True
        logger.info("Feature list loaded", path=path, count=len(self.feature_names))
