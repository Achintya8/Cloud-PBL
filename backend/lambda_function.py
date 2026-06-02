"""
backend/lambda_function.py
==========================
FastAPI application served via Mangum (serverless ASGI adapter for AWS Lambda).

Endpoints:
  GET /api/v1/health                    → Health check
  GET /api/v1/prices/current            → Latest market prices per variety
  GET /api/v1/prices/forecast           → ML price forecasts (7 & 30 day)
  GET /api/v1/prices/history            → Historical price data
  GET /api/v1/markets                   → List of all configured markets
  GET /api/v1/varieties                 → List of all areca nut varieties

Connection pooling uses a module-level singleton so the pool persists
across Lambda invocation warm-starts.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from config.settings import app as app_cfg
from database.db_manager import db_manager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Areca Nut Price Prediction API",
    description=(
        "Production-grade API for Karnataka areca nut market prices, "
        "weather metrics, and ML-generated price forecasts. "
        "Designed for farmers, traders, and agri-market analysts."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database pool initialization on startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Initialize the DB connection pool when the application starts."""
    db_manager.initialize()
    logger.info("FastAPI application started, DB pool initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close DB connections on shutdown."""
    db_manager.close()
    logger.info("FastAPI application shutting down")


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    timestamp: datetime


class MarketRecord(BaseModel):
    market_id: str
    market_name: str
    district: str
    state: str
    latitude: Optional[float]
    longitude: Optional[float]


class VarietyRecord(BaseModel):
    variety_id: int
    variety_name: str
    local_name: Optional[str]
    description: Optional[str]


class CurrentPriceRecord(BaseModel):
    record_date: date
    market_name: str
    district: str
    variety_name: str
    min_price: float
    max_price: float
    modal_price: float
    arrivals_tons: Optional[float]
    source: str
    price_trend: Optional[str] = Field(None, description="'rising', 'falling', or 'stable'")
    trend_pct: Optional[float] = Field(None, description="Percentage change vs 7-day average")
    recommendation: Optional[str] = Field(None, description="'HOLD' or 'SELL'")


class ForecastRecord(BaseModel):
    target_date: date
    variety_name: str
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    horizon_days: int
    model_version: str


class HistoryRecord(BaseModel):
    record_date: date
    market_name: str
    variety_name: str
    modal_price: float
    min_price: float
    max_price: float
    arrivals_tons: Optional[float]


class ApiResponse(BaseModel):
    success: bool
    data: Any
    count: int
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Helper: Decimal/Date serialization
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Recursively convert Decimal and date objects to JSON-compatible types."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------

TREND_CURRENT_QUERY = """
WITH current_prices AS (
    SELECT
        mp.record_date,
        m.market_id::text,
        m.market_name,
        m.district,
        m.state,
        v.variety_name,
        mp.min_price,
        mp.max_price,
        mp.modal_price,
        mp.arrivals_tons,
        mp.source
    FROM market_prices mp
    JOIN markets   m ON m.market_id  = mp.market_id
    JOIN varieties v ON v.variety_id = mp.variety_id
    WHERE mp.record_date = (
        SELECT MAX(record_date) FROM market_prices
    )
    {variety_filter}
    {market_filter}
),
avg_7d AS (
    SELECT
        m2.market_id::text AS market_id,
        v2.variety_name,
        AVG(mp2.modal_price) AS avg_7d_price
    FROM market_prices mp2
    JOIN markets   m2 ON m2.market_id  = mp2.market_id
    JOIN varieties v2 ON v2.variety_id = mp2.variety_id
    WHERE mp2.record_date BETWEEN (SELECT MAX(record_date) FROM market_prices) - INTERVAL '7 days'
                               AND (SELECT MAX(record_date) FROM market_prices) - INTERVAL '1 day'
    GROUP BY m2.market_id, v2.variety_name
)
SELECT
    cp.*,
    ag.avg_7d_price,
    CASE
        WHEN ag.avg_7d_price IS NULL OR ag.avg_7d_price = 0 THEN NULL
        ELSE ROUND(((cp.modal_price - ag.avg_7d_price) / ag.avg_7d_price * 100)::NUMERIC, 2)
    END AS trend_pct
FROM current_prices cp
LEFT JOIN avg_7d ag ON ag.market_id = cp.market_id AND ag.variety_name = cp.variety_name
ORDER BY cp.market_name, cp.variety_name
LIMIT %s
"""


def _compute_trend_label(trend_pct: Optional[float]) -> Optional[str]:
    if trend_pct is None:
        return None
    if trend_pct > 2.0:
        return "rising"
    if trend_pct < -2.0:
        return "falling"
    return "stable"


def _compute_recommendation(trend_pct: Optional[float]) -> Optional[str]:
    """
    Simplified farmer recommendation:
    - HOLD if price is rising or stable (wait for better prices)
    - SELL if price is falling (liquidate before further decline)
    """
    if trend_pct is None:
        return None
    return "SELL" if trend_pct < -2.0 else "HOLD"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check. Verifies API and database connectivity.
    """
    db_health = db_manager.health_check()
    return {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "database": db_health,
        "timestamp": datetime.utcnow(),
    }


@app.get("/api/v1/markets", tags=["Reference Data"])
async def list_markets(
    active_only: bool = Query(True, description="Return only active markets")
) -> ApiResponse:
    """List all areca nut markets in the system."""
    query = "SELECT market_id::text, market_name, district, state, latitude, longitude FROM markets"
    if active_only:
        query += " WHERE active = TRUE"
    query += " ORDER BY state, district, market_name"

    rows = db_manager.execute_query(query)
    return ApiResponse(
        success=True,
        data=_serialise(rows),
        count=len(rows),
    )


@app.get("/api/v1/varieties", tags=["Reference Data"])
async def list_varieties() -> ApiResponse:
    """List all tracked areca nut variety types."""
    rows = db_manager.execute_query(
        "SELECT variety_id, variety_name, local_name, description FROM varieties WHERE active = TRUE ORDER BY variety_id"
    )
    return ApiResponse(success=True, data=_serialise(rows), count=len(rows))


@app.get("/api/v1/prices/current", tags=["Prices"])
async def get_current_prices(
    variety: Optional[str] = Query(None, description="Filter by variety name (e.g. 'Chali')"),
    market:  Optional[str] = Query(None, description="Filter by market name (e.g. 'Shimoga')"),
    limit:   int           = Query(100, ge=1, le=1000, description="Max records to return"),
) -> ApiResponse:
    """
    Fetch the latest recorded market prices for all areca nut varieties.

    Returns price trend indicators:
    - **rising** (≥ +2% vs 7-day avg) → Green — **HOLD**
    - **stable** (±2% vs 7-day avg)   → Amber
    - **falling** (≤ -2% vs 7-day avg) → Red — **SELL**
    """
    variety_filter = "AND v.variety_name = %(variety)s" if variety else ""
    market_filter  = "AND LOWER(m.market_name) = LOWER(%(market)s)" if market else ""

    query = TREND_CURRENT_QUERY.format(
        variety_filter=variety_filter,
        market_filter=market_filter,
    )

    params: list = []
    if variety and market:
        params = [variety, market, limit]
    elif variety:
        params = [variety, limit]
    elif market:
        params = [market, limit]
    else:
        params = [limit]

    # Rebuild the query with positional parameters (psycopg2 style)
    variety_clause = "AND v.variety_name = %s" if variety else ""
    market_clause  = "AND LOWER(m.market_name) = LOWER(%s)" if market else ""

    base_query = TREND_CURRENT_QUERY.replace(
        "{variety_filter}", variety_clause
    ).replace(
        "{market_filter}", market_clause
    )

    try:
        rows = db_manager.execute_query(base_query, tuple(params))
    except Exception as exc:
        logger.error("current prices query failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    # Augment with trend label and recommendation
    enriched = []
    for row in rows:
        trend_pct = float(row["trend_pct"]) if row.get("trend_pct") is not None else None
        enriched.append({
            **_serialise(row),
            "price_trend":    _compute_trend_label(trend_pct),
            "recommendation": _compute_recommendation(trend_pct),
        })

    return ApiResponse(
        success=True,
        data=enriched,
        count=len(enriched),
        message=f"Prices as of {enriched[0]['record_date']}" if enriched else "No data available",
    )


@app.get("/api/v1/prices/forecast", tags=["Forecast"])
async def get_price_forecast(
    variety:  Optional[str] = Query(None, description="Filter by variety (e.g. 'Chali')"),
    horizon:  Optional[int] = Query(None, description="Forecast horizon: 7 or 30 days"),
    limit:    int           = Query(200, ge=1, le=500),
) -> ApiResponse:
    """
    Fetch ML model-generated price forecasts for areca nut varieties.

    Returns point predictions with confidence intervals (90% by default).
    """
    conditions = ["pp.target_date >= CURRENT_DATE"]
    params: List[Any] = []

    if variety:
        conditions.append("v.variety_name = %s")
        params.append(variety)

    if horizon:
        if horizon not in (7, 30):
            raise HTTPException(
                status_code=400,
                detail="horizon must be 7 or 30"
            )
        conditions.append("pp.horizon_days = %s")
        params.append(horizon)

    where_clause = " AND ".join(conditions)
    params.append(limit)

    query = f"""
        SELECT
            pp.target_date,
            v.variety_name,
            m.market_name,
            pp.predicted_price,
            pp.confidence_lower,
            pp.confidence_upper,
            pp.horizon_days,
            pp.model_version,
            pp.model_rmse,
            pp.prediction_date AS model_run_date
        FROM price_predictions pp
        JOIN varieties v ON v.variety_id = pp.variety_id
        JOIN markets   m ON m.market_id  = pp.market_id
        WHERE pp.prediction_date = (
            SELECT MAX(prediction_date) FROM price_predictions
        )
          AND {where_clause}
        ORDER BY pp.horizon_days, v.variety_name, pp.target_date
        LIMIT %s
    """

    try:
        rows = db_manager.execute_query(query, tuple(params))
    except Exception as exc:
        logger.error("forecast query failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    return ApiResponse(
        success=True,
        data=_serialise(rows),
        count=len(rows),
        message=(
            "No forecast data available. Ensure the ML training job has run successfully."
            if not rows else None
        ),
    )


@app.get("/api/v1/prices/history", tags=["Prices"])
async def get_price_history(
    variety: Optional[str] = Query(None, description="Filter by variety"),
    market:  Optional[str] = Query(None, description="Filter by market name"),
    days:    int           = Query(90, ge=7, le=730, description="Days of history"),
    limit:   int           = Query(500, ge=1, le=5000),
) -> ApiResponse:
    """
    Fetch historical price data for charting and analysis.
    """
    conditions = ["mp.record_date >= CURRENT_DATE - (%s * INTERVAL '1 day')"]
    params: List[Any] = [days]

    if variety:
        conditions.append("v.variety_name = %s")
        params.append(variety)
    if market:
        conditions.append("LOWER(m.market_name) LIKE LOWER(%s)")
        params.append(f"%{market}%")

    where_clause = " AND ".join(conditions)
    params.append(limit)

    query = f"""
        SELECT
            mp.record_date,
            m.market_name,
            m.district,
            v.variety_name,
            mp.modal_price,
            mp.min_price,
            mp.max_price,
            mp.arrivals_tons,
            mp.source
        FROM market_prices mp
        JOIN markets   m ON m.market_id  = mp.market_id
        JOIN varieties v ON v.variety_id = mp.variety_id
        WHERE {where_clause}
        ORDER BY mp.record_date DESC, m.market_name, v.variety_name
        LIMIT %s
    """

    try:
        rows = db_manager.execute_query(query, tuple(params))
    except Exception as exc:
        logger.error("history query failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Database query failed") from exc

    return ApiResponse(success=True, data=_serialise(rows), count=len(rows))


@app.get("/api/v1/prices/summary", tags=["Prices"])
async def get_price_summary() -> ApiResponse:
    """
    Summary statistics: latest prices, 30-day averages, and active market count.
    Used by the frontend dashboard summary cards.
    """
    query = """
        SELECT
            v.variety_name,
            COUNT(DISTINCT mp.market_id)         AS active_markets,
            MAX(mp.record_date)                  AS latest_date,
            AVG(mp.modal_price)::NUMERIC(10,2)   AS avg_modal_price_today,
            MIN(mp.modal_price)                  AS min_price,
            MAX(mp.modal_price)                  AS max_price,
            SUM(mp.arrivals_tons)                AS total_arrivals_tons
        FROM market_prices mp
        JOIN varieties v ON v.variety_id = mp.variety_id
        WHERE mp.record_date = (SELECT MAX(record_date) FROM market_prices)
        GROUP BY v.variety_name
        ORDER BY v.variety_name
    """
    rows = db_manager.execute_query(query)
    return ApiResponse(success=True, data=_serialise(rows), count=len(rows))


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=str(request.url), error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Mangum handler — AWS Lambda entry point
# ---------------------------------------------------------------------------

handler = Mangum(app, lifespan="off")


def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    AWS Lambda entry point.
    Routes API Gateway v2 (HTTP API) proxy events through Mangum → FastAPI.
    """
    logger.info("Lambda API invocation", path=event.get("rawPath", "unknown"))
    return handler(event, context)


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from config.settings import app as app_cfg

    uvicorn.run(
        "backend.lambda_function:app",
        host="0.0.0.0",
        port=app_cfg.api_port,
        reload=True,
        log_level=app_cfg.log_level.lower(),
    )
