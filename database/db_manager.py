"""
database/db_manager.py
======================
Database connection pool manager with SQLAlchemy and raw psycopg2 support.
Provides a thread-safe connection pool for Lambda/EC2 environments.
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import psycopg2.pool
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from config.logging_config import get_logger
from config.settings import db as db_cfg

logger = get_logger(__name__)


class DatabaseManager:
    """
    Singleton database manager providing:
      - SQLAlchemy engine with connection pooling
      - Raw psycopg2 connection pool for bulk writes
      - Context managers for safe session/connection handling
    """

    _instance: Optional["DatabaseManager"] = None
    _engine = None
    _session_factory = None
    _psycopg2_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("Initializing database connection pools", host=db_cfg.host, db=db_cfg.name)

        # --- SQLAlchemy Engine ---
        self._engine = create_engine(
            db_cfg.url,
            poolclass=QueuePool,
            pool_size=db_cfg.pool_size,
            max_overflow=db_cfg.max_overflow,
            pool_timeout=db_cfg.pool_timeout,
            pool_pre_ping=True,   # reconnect on stale connections
            pool_recycle=3600,    # recycle connections every hour
            echo=False,
        )

        # Emit a connection warning on checkout failure
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_conn, _):
            dbapi_conn.set_session(autocommit=False)
            logger.debug("New SQLAlchemy DB connection established")

        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=True,
            expire_on_commit=False,
        )

        # --- psycopg2 ThreadedConnectionPool ---
        self._psycopg2_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=db_cfg.pool_size + db_cfg.max_overflow,
            dsn=db_cfg.psycopg2_dsn,
        )

        self._initialized = True
        logger.info("Database pools initialized successfully")

    # -----------------------------------------------------------------------
    # SQLAlchemy session context manager
    # -----------------------------------------------------------------------
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Yield a transactional SQLAlchemy session with automatic rollback."""
        if not self._initialized:
            self.initialize()

        sess: Session = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception as exc:
            sess.rollback()
            logger.error("Session rolled back due to exception", exc_info=True, error=str(exc))
            raise
        finally:
            sess.close()

    # -----------------------------------------------------------------------
    # psycopg2 connection context manager (for bulk COPY / raw queries)
    # -----------------------------------------------------------------------
    @contextmanager
    def raw_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Yield a raw psycopg2 connection from the thread pool."""
        if not self._initialized:
            self.initialize()

        conn = None
        try:
            conn = self._psycopg2_pool.getconn()
            conn.autocommit = False
            yield conn
            conn.commit()
        except Exception as exc:
            if conn:
                conn.rollback()
            logger.error("Raw connection error, rolled back", error=str(exc), exc_info=True)
            raise
        finally:
            if conn:
                self._psycopg2_pool.putconn(conn)

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------
    def execute_scalar(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute a query and return the first column of the first row."""
        with self.raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return result[0] if result else None

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        as_dict: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute a query and return all rows as a list of dicts if applicable."""
        with self.raw_connection() as conn:
            cursor_factory = psycopg2.extras.RealDictCursor if as_dict else None
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                cur.execute(query, params)
                if cur.description is None:
                    return []
                return [dict(row) for row in cur.fetchall()]

    def bulk_insert(
        self,
        table: str,
        columns: List[str],
        rows: List[Tuple],
        on_conflict: str = "DO NOTHING",
    ) -> int:
        """
        Efficient bulk insert using psycopg2 execute_values.

        Args:
            table:       Target table name.
            columns:     List of column names.
            rows:        List of value tuples matching column order.
            on_conflict: SQL conflict resolution clause.

        Returns:
            Number of rows inserted.
        """
        if not rows:
            return 0

        col_str = ", ".join(columns)
        sql = (
            f"INSERT INTO {table} ({col_str}) VALUES %s "
            f"ON CONFLICT {on_conflict}"
        )

        with self.raw_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, rows, template=None, page_size=db_cfg.pool_size * 20
                )
                return cur.rowcount

    def refresh_views(self) -> None:
        """Refresh all materialized views after an ETL run."""
        with self.raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT refresh_materialized_views();")
        logger.info("Materialized views refreshed")

    def health_check(self) -> Dict[str, Any]:
        """Return database health status."""
        try:
            start = time.monotonic()
            result = self.execute_scalar("SELECT version();")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "version": str(result),
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    def close(self) -> None:
        """Gracefully close all connection pools."""
        if self._engine:
            self._engine.dispose()
        if self._psycopg2_pool:
            self._psycopg2_pool.closeall()
        self._initialized = False
        logger.info("Database connection pools closed")


# ---------------------------------------------------------------------------
# Module-level singleton (import and use directly)
# ---------------------------------------------------------------------------
db_manager = DatabaseManager()
