"""
scripts/seed_data.py
====================
Seeds the database with realistic synthetic areca nut price data
for the last 90 days, bypassing the external scrapers.

Usage:
    python scripts/seed_data.py
    python scripts/seed_data.py --days 180
"""

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.logging_config import get_logger
from database.db_manager import db_manager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Realistic base prices per variety (INR/quintal, Karnataka 2025-26 typical)
# ---------------------------------------------------------------------------
VARIETY_BASE_PRICES = {
    "Chali":  {"min": 38000, "max": 52000, "modal": 44000},  # Most common processed form
    "Gotu":   {"min": 28000, "max": 40000, "modal": 33000},  # Whole/raw
    "Kotte":  {"min": 20000, "max": 32000, "modal": 25000},  # Tender
    "Rashi":  {"min": 30000, "max": 42000, "modal": 35000},  # Mixed grade
    "Saraku": {"min": 35000, "max": 48000, "modal": 40000},  # Processed premium
}

# Market-specific price adjustments (some markets command a premium)
MARKET_ADJUSTMENTS = {
    "Shimoga":       1.05,
    "Sagara":        1.02,
    "Thirthahalli":  0.98,
    "Sagar":         1.00,
    "Mudigere":      1.03,
    "Chikkamagaluru": 1.01,
    "Mangaluru":     1.07,  # Port city, premium buyer
    "Hassan":        0.99,
    "Puttur":        1.04,
    "Bantwal":       1.02,
}

ARRIVALS_BASE = {  # Tonnes/day typical
    "Shimoga": 45.0,
    "Sagara":  30.0,
    "Thirthahalli": 18.0,
    "Sagar":   22.0,
    "Mudigere": 15.0,
    "Chikkamagaluru": 28.0,
    "Mangaluru": 55.0,
    "Hassan":  20.0,
    "Puttur":  25.0,
    "Bantwal": 20.0,
}


def _seasonal_factor(d: date) -> float:
    """
    Areca nut prices peak Nov-Feb (post-harvest demand) and dip May-Aug.
    Returns a multiplier around 1.0.
    """
    day_of_year = d.timetuple().tm_yday
    # Cosine wave: peaks around day 15 (Jan 15), troughs around day 195 (Jul 14)
    return 1.0 + 0.12 * math.cos(2 * math.pi * (day_of_year - 15) / 365)


def _trend_factor(d: date, start_date: date) -> float:
    """Mild upward trend over the period (~8% per year)."""
    days_elapsed = (d - start_date).days
    return 1.0 + (0.08 / 365) * days_elapsed


def _weekly_pattern(d: date) -> float:
    """Prices dip slightly on Sundays (markets closed) and Mondays (thin trading)."""
    wd = d.weekday()  # 0=Mon, 6=Sun
    if wd == 6:   # Sunday
        return 0.0   # No trading
    if wd == 0:   # Monday — thin arrivals
        return 0.97
    if wd == 5:   # Saturday — slightly lower
        return 0.99
    return 1.0


def generate_price(
    variety: str,
    market: str,
    d: date,
    start_date: date,
    rng: random.Random,
) -> tuple[float, float, float, float | None]:
    """Return (min_price, max_price, modal_price, arrivals_tons) for a given day."""
    base = VARIETY_BASE_PRICES[variety]
    adj = MARKET_ADJUSTMENTS.get(market, 1.0)
    seasonal = _seasonal_factor(d)
    trend = _trend_factor(d, start_date)
    weekly = _weekly_pattern(d)

    if weekly == 0.0:
        return None  # No market on Sundays

    # Random daily noise ±3%
    noise = rng.gauss(1.0, 0.03)
    factor = adj * seasonal * trend * weekly * noise

    modal = round(base["modal"] * factor, 2)
    spread = base["max"] - base["min"]
    half_spread = spread * rng.uniform(0.35, 0.65)
    min_p = round(modal - half_spread * 0.6, 2)
    max_p = round(modal + half_spread * 0.4, 2)

    # Clamp modal to [min, max]
    modal = max(min_p, min(modal, max_p))

    # Arrivals vary by season (more post-harvest)
    base_arr = ARRIVALS_BASE.get(market, 20.0)
    arrivals = round(base_arr * seasonal * rng.uniform(0.7, 1.4), 1)

    return min_p, max_p, modal, arrivals


def seed(days: int = 90) -> None:
    db_manager.initialize()

    # Load market and variety IDs from DB
    markets_rows = db_manager.execute_query(
        "SELECT market_id::text, market_name FROM markets WHERE active = TRUE"
    )
    varieties_rows = db_manager.execute_query(
        "SELECT variety_id, variety_name FROM varieties WHERE active = TRUE"
    )

    market_cache = {r["market_name"]: r["market_id"] for r in markets_rows}
    variety_cache = {r["variety_name"]: r["variety_id"] for r in varieties_rows}

    print(f"Markets: {list(market_cache.keys())}")
    print(f"Varieties: {list(variety_cache.keys())}")

    rng = random.Random(42)  # Deterministic seed for reproducibility
    today = date.today()
    start_date = today - timedelta(days=days)

    rows = []
    skipped_sundays = 0

    current = start_date
    while current <= today:
        for market_name, market_id in market_cache.items():
            for variety_name, variety_id in variety_cache.items():
                if market_name not in MARKET_ADJUSTMENTS:
                    # Market exists in DB but no price config — use neutral adjustment
                    MARKET_ADJUSTMENTS[market_name] = 1.0
                if variety_name not in VARIETY_BASE_PRICES:
                    continue

                result = generate_price(variety_name, market_name, current, start_date, rng)
                if result is None:
                    skipped_sundays += 1
                    continue

                min_p, max_p, modal, arrivals = result
                rows.append((
                    current,
                    market_id,
                    variety_id,
                    min_p,
                    max_p,
                    modal,
                    arrivals,
                    "synthetic-seed",
                    json.dumps({"generated": True, "seed_date": str(today)}),
                ))
        current += timedelta(days=1)

    print(f"Generated {len(rows)} price records ({skipped_sundays} Sundays skipped).")

    if not rows:
        print("No rows to insert.")
        return

    inserted = db_manager.bulk_insert(
        table="market_prices",
        columns=[
            "record_date", "market_id", "variety_id",
            "min_price", "max_price", "modal_price",
            "arrivals_tons", "source", "raw_payload",
        ],
        rows=rows,
        on_conflict=(
            "(record_date, market_id, variety_id) "
            "DO UPDATE SET "
            "  min_price     = EXCLUDED.min_price, "
            "  max_price     = EXCLUDED.max_price, "
            "  modal_price   = EXCLUDED.modal_price, "
            "  arrivals_tons = EXCLUDED.arrivals_tons, "
            "  ingested_at   = NOW()"
        ),
    )
    print(f"✅ Inserted/updated {len(rows)} rows into market_prices.")

    # Refresh materialized views
    try:
        db_manager.execute_query("SELECT refresh_materialized_views()")
        print("✅ Materialized views refreshed.")
    except Exception as exc:
        print(f"⚠️  Could not refresh views: {exc}")

    # Verify
    count_rows = db_manager.execute_query("SELECT COUNT(*) as c FROM market_prices")
    print(f"✅ market_prices now has {count_rows[0]['c']} rows.")

    db_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed areca nut price data")
    parser.add_argument("--days", type=int, default=90, help="Number of days to seed (default: 90)")
    args = parser.parse_args()
    seed(args.days)
