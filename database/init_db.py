"""
database/init_db.py
===================
Database initialization script: applies schema DDL and seeds reference data.
Run this once after the RDS instance is provisioned.

Usage:
    python -m database.init_db
    python -m database.init_db --drop-all  (DANGER: drops all tables first)
"""

import argparse
import sys
from pathlib import Path

import psycopg2

from config.logging_config import get_logger
from config.settings import db as db_cfg

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

MARKET_SEEDS = [
    ("Shimoga",         "Karnataka", "Shivamogga",   "KAR001", 13.9299, 75.5681),
    ("Sagara",          "Karnataka", "Shivamogga",   "KAR002", 14.1673, 75.0267),
    ("Thirthahalli",    "Karnataka", "Shivamogga",   "KAR003", 13.6866, 75.2302),
    ("Mudigere",        "Karnataka", "Chikkamagaluru","KAR004",13.1333, 75.6333),
    ("Chikkamagaluru",  "Karnataka", "Chikkamagaluru","KAR005",13.3153, 75.7754),
    ("Mangaluru",       "Karnataka", "Dakshina Kannada","KAR006",12.9141,74.8560),
    ("Hassan",          "Karnataka", "Hassan",        "KAR007", 13.0069, 76.1003),
    ("Puttur",          "Karnataka", "Dakshina Kannada","KAR008",12.7598,75.1986),
    ("Bantwal",         "Karnataka", "Dakshina Kannada","KAR009",12.8957,75.0359),
    ("Sagar",           "Karnataka", "Shivamogga",   "KAR010", 14.1673, 75.0267),
    ("Ankola",          "Karnataka", "Uttara Kannada","KAR011", 14.6593, 74.2974),
    ("Sirsi",           "Karnataka", "Uttara Kannada","KAR012", 14.6219, 74.8378),
    ("Sullia",          "Karnataka", "Dakshina Kannada","KAR013",12.5571,75.3870),
    ("Belthangady",     "Karnataka", "Dakshina Kannada","KAR014",12.9850,75.2981),
    ("Udupi",           "Karnataka", "Udupi",         "KAR015", 13.3409, 74.7421),
]


def get_connection():
    """Create a raw psycopg2 connection."""
    conn = psycopg2.connect(db_cfg.psycopg2_dsn)
    conn.autocommit = False
    return conn


def drop_all(conn):
    """Drop all tables (use only in development)."""
    logger.warning("Dropping all tables — this is destructive!")
    with conn.cursor() as cur:
        cur.execute("""
            DROP SCHEMA public CASCADE;
            CREATE SCHEMA public;
            GRANT ALL ON SCHEMA public TO postgres;
            GRANT ALL ON SCHEMA public TO public;
        """)
    conn.commit()
    logger.info("All tables dropped")


def apply_schema(conn):
    """Apply the DDL schema from schema.sql."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    logger.info("Applying schema DDL", path=str(SCHEMA_PATH))
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    logger.info("Schema applied successfully")


def seed_markets(conn):
    """Insert default Karnataka areca nut markets."""
    logger.info("Seeding markets table")
    with conn.cursor() as cur:
        for market_name, state, district, code, lat, lon in MARKET_SEEDS:
            cur.execute(
                """
                INSERT INTO markets (market_name, state, district, apmc_code, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (market_name, district) DO UPDATE SET
                    apmc_code = EXCLUDED.apmc_code,
                    latitude  = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    active    = TRUE
                """,
                (market_name, state, district, code, lat, lon),
            )
    conn.commit()
    logger.info("Markets seeded", count=len(MARKET_SEEDS))


def verify_schema(conn):
    """Verify that core tables exist."""
    required_tables = [
        "markets", "varieties", "market_prices",
        "weather_metrics", "price_predictions",
        "model_training_runs", "etl_runs",
    ]
    with conn.cursor() as cur:
        for table in required_tables:
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s",
                (table,)
            )
            count = cur.fetchone()[0]
            if count == 0:
                logger.error("Table missing after schema apply", table=table)
                return False
            logger.info("Table verified", table=table)
    return True


def main(drop_all_first: bool = False):
    logger.info("Starting database initialization", host=db_cfg.host, db=db_cfg.name)

    conn = None
    try:
        conn = get_connection()

        if drop_all_first:
            drop_all(conn)

        apply_schema(conn)
        seed_markets(conn)

        ok = verify_schema(conn)
        if not ok:
            logger.error("Schema verification failed")
            sys.exit(1)

        logger.info("Database initialization complete")

    except psycopg2.OperationalError as exc:
        logger.error(
            "Cannot connect to database",
            error=str(exc),
            host=db_cfg.host,
            port=db_cfg.port,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Initialization failed", error=str(exc), exc_info=True)
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Areca Price System database")
    parser.add_argument(
        "--drop-all",
        action="store_true",
        help="Drop and recreate all tables (DESTRUCTIVE — dev only)",
    )
    args = parser.parse_args()
    main(drop_all_first=args.drop_all)
