"""
scripts/run_local.py
====================
Local development runner — starts all services without AWS.
Runs:
  1. FastAPI backend (uvicorn)
  2. Frontend server (uvicorn)
Expects PostgreSQL running locally (Docker or native).

Usage:
    python scripts/run_local.py
    python scripts/run_local.py --etl          # Also run ETL once
    python scripts/run_local.py --train        # Also train ML model
    python scripts/run_local.py --init-db      # Initialize database
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def run_command(cmd: list, bg: bool = False, env_extra: dict = None):
    import os
    env = {**os.environ, **(env_extra or {})}
    if bg:
        return subprocess.Popen(cmd, env=env)
    else:
        return subprocess.run(cmd, env=env, check=True)


def init_db():
    print("🗄️  Initializing database…")
    run_command([sys.executable, "-m", "database.init_db"])
    print("✅ Database initialized")


def run_etl():
    print("📥 Running ETL pipeline (yesterday's data)…")
    run_command([sys.executable, "-m", "etl.etl_pipeline", "--backfill", "7"])
    print("✅ ETL complete")


def run_training():
    print("🤖 Training ML models…")
    run_command([sys.executable, "-m", "ml_engine.train_predict"])
    print("✅ ML training complete")


def start_backend():
    print("🚀 Starting FastAPI backend on port 8000…")
    return run_command(
        [
            sys.executable, "-m", "uvicorn",
            "backend.lambda_function:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info",
        ],
        bg=True,
    )


def start_frontend():
    print("🌐 Starting frontend on port 8080…")
    return run_command(
        [
            sys.executable, "-m", "uvicorn",
            "frontend.server:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload",
            "--log-level", "info",
        ],
        bg=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Local development runner")
    parser.add_argument("--init-db",  action="store_true")
    parser.add_argument("--etl",      action="store_true")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--no-frontend", action="store_true")
    args = parser.parse_args()

    if args.init_db:
        init_db()

    if args.etl:
        run_etl()

    if args.train:
        run_training()

    procs = []

    backend_proc  = start_backend()
    procs.append(backend_proc)
    time.sleep(2)

    if not args.no_frontend:
        frontend_proc = start_frontend()
        procs.append(frontend_proc)
        time.sleep(1)

    print("\n" + "="*60)
    print("✅ All services running:")
    print("   🔗 Backend API:  http://localhost:8000/api/docs")
    print("   🌐 Frontend:     http://localhost:8080")
    print("   📈 Grafana:      http://localhost:3000")
    print("   Press Ctrl+C to stop all services.")
    print("="*60 + "\n")

    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services…")
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
