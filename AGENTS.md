# AGENTS.md — Areca Nut Price Prediction System
# Agent / AI Assistant Orientation Guide

> This file exists for AI coding agents. Read it fully before modifying any code.
> It describes project conventions, module boundaries, data models, and the rules
> that must be respected to avoid breaking the system.

---

## 1. Project Overview

A production-grade cloud platform that:
1. **Scrapes** Karnataka areca nut market prices from Agmarknet (ETL)
2. **Stores** them in a PostgreSQL database (RDS in production)
3. **Trains** a LightGBM model to forecast prices 7 and 30 days ahead (ML Engine)
4. **Serves** data and predictions via a FastAPI REST API (Backend)
5. **Visualises** everything via a Jinja2 frontend and Grafana dashboards
6. **Deploys** to AWS using a pure-boto3 idempotent provisioner (no Terraform)

Target users: Karnataka areca nut farmers, traders, and agri-market analysts.

---

## 2. Repository Layout

```
areca-price-system/
├── backend/
│   └── lambda_function.py      # FastAPI app (also works as AWS Lambda via Mangum)
├── config/
│   ├── settings.py             # All config — dataclasses reading from env vars
│   └── logging_config.py       # JSON-structured logger factory
├── database/
│   ├── schema.sql              # Full DDL — source of truth for the DB schema
│   ├── init_db.py              # Runs schema.sql against the configured DB
│   └── db_manager.py           # Connection pool + typed query helpers
├── etl/
│   └── etl_pipeline.py         # Scraper + data-cleaning + DB ingestion
├── frontend/
│   ├── server.py               # Uvicorn/Jinja2 frontend server (port 8080)
│   ├── templates/              # HTML templates (Jinja2)
│   └── static/                 # CSS, JS, images
├── grafana/
│   ├── dashboards/             # Dashboard JSON definitions
│   └── provisioning/           # Grafana datasource / dashboard YAML
├── infra/
│   └── deploy_infra.py         # Boto3 AWS provisioner — idempotent
├── ml_engine/
│   ├── feature_engineering.py  # Time-series + weather feature builder
│   └── train_predict.py        # LightGBM training, evaluation, prediction
├── scripts/
│   └── run_local.py            # Local dev orchestrator
├── .env.example                # All env vars documented
└── requirements.txt            # Pinned Python dependencies
```

---

## 3. Configuration System

**File:** `config/settings.py`

All configuration is loaded from environment variables into plain Python `@dataclass` objects. There are **six config objects**, each a singleton instantiated at module load time:

| Singleton | Dataclass    | Covers                                      |
|-----------|--------------|---------------------------------------------|
| `db`      | `DBConfig`   | PostgreSQL connection, pool settings         |
| `aws`     | `AWSConfig`  | Region, EC2, RDS, Lambda, VPC, API GW        |
| `etl`     | `ETLConfig`  | Scrape URLs, target markets/varieties, retries |
| `ml`      | `MLConfig`   | Model paths, hyperparameters, lag/rolling windows |
| `grafana` | `GrafanaConfig` | Host, port, credentials, dashboard UIDs  |
| `app`     | `AppConfig`  | Environment, log level, ports, CORS, secret key |

**Usage pattern throughout the codebase:**
```python
from config.settings import app as app_cfg, db as db_cfg, ml as ml_cfg
```

**Rule:** Never read `os.getenv()` directly outside `config/settings.py`. All new config must go through a dataclass field there.

---

## 4. Logging

**File:** `config/logging_config.py`

All modules use a single factory function:
```python
from config.logging_config import get_logger
logger = get_logger(__name__)                          # stdout JSON only
logger = get_logger(__name__, log_file="/var/log/areca/module.log")  # + file
```

Logs are emitted as **single-line JSON** to stdout (CloudWatch-compatible) and optionally to a file. The `JSONFormatter` attaches: `timestamp`, `level`, `logger`, `module`, `function`, `line`, `message`, `environment`, and any exception details.

**Rule:** Never use `print()` for diagnostic output. Always use the logger.

---

## 5. Database

### 5.1 Schema (`database/schema.sql`)

The schema is the **single source of truth**. Always update `schema.sql` before writing any DB code.

Core tables:

| Table                  | Description                                       |
|------------------------|---------------------------------------------------|
| `markets`              | Reference: Karnataka APMC market locations        |
| `varieties`            | Reference: areca nut varieties (5 pre-seeded)     |
| `market_prices`        | Fact: daily min/max/modal price + arrivals        |
| `weather_metrics`      | Fact: daily rainfall, humidity, temperature       |
| `price_predictions`    | Fact: ML forecast output per market/variety/horizon |
| `model_training_runs`  | Audit: training job metadata + metrics            |
| `etl_runs`             | Audit: ETL job metadata + record counts           |

Materialised views (refresh via `SELECT refresh_materialized_views()`):
- `mv_latest_prices` — latest price per market/variety
- `mv_variety_stats_30d` — 30-day aggregate stats per variety

### 5.2 DB Manager (`database/db_manager.py`)

Use `db_manager` (singleton) for all database interactions. Do **not** open raw `psycopg2` connections elsewhere.

### 5.3 Constraints to respect

- `market_prices` has a unique constraint on `(record_date, market_id, variety_id)` — upsert, don't re-insert.
- `price_predictions` has a unique constraint on `(prediction_date, target_date, market_id, variety_id, horizon_days)`.
- `horizon_days` only accepts `7` or `30`.
- `modal_price` must be `BETWEEN min_price AND max_price`.
- All monetary values are `NUMERIC(10, 2)` — use `Decimal`, not `float`.

---

## 6. ETL Pipeline

**File:** `etl/etl_pipeline.py`

- Scrapes from **Agmarknet** (`https://agmarknet.gov.in/SearchCmmMkt.aspx`)
- Fetches weather from **Open-Meteo** (no API key required by default)
- Writes to `market_prices` and `weather_metrics`
- Logs each run to `etl_runs`
- Is idempotent — duplicate records are skipped via `ON CONFLICT DO NOTHING`

Target markets (from `ETLConfig.target_markets`):
`Shimoga`, `Sagara`, `Thirthahalli`, `Sagar`, `Mudigere`, `Chikkamagaluru`, `Mangaluru`, `Hassan`, `Puttur`, `Bantwal`

Target varieties (from `ETLConfig.target_varieties`):
`Chali`, `Gotu`, `Kotte`, `Rashi`, `Saraku`

**Run manually:**
```bash
python -m etl.etl_pipeline                 # yesterday only
python -m etl.etl_pipeline --backfill 30   # last 30 days
```

---

## 7. ML Engine

**Files:** `ml_engine/feature_engineering.py`, `ml_engine/train_predict.py`

### Feature engineering
- **Lag features:** `[1, 3, 7, 14, 21, 30]` days (from `MLConfig.lag_days`)
- **Rolling windows:** `[7, 14, 30]` days — mean, std, min, max (from `MLConfig.rolling_windows`)
- **Calendar features:** day-of-week, month, week-of-year, is-weekend
- **Weather features:** rainfall, humidity, temperature joined from `weather_metrics`

### Model
- **Algorithm:** LightGBM (regression, RMSE objective)
- **Forecast horizons:** 7-day and 30-day (separate models per variety per horizon)
- **Confidence intervals:** quantile regression ensemble (`n_quantile_estimators=100`)
- **Minimum training data:** 90 rows (`MLConfig.min_training_rows`) — skips training if insufficient

### Artifacts saved to disk
| File                   | Env var        | Default path                    |
|------------------------|----------------|---------------------------------|
| `lgbm_areca.pkl`       | `MODEL_PATH`   | `/opt/areca/models/lgbm_areca.pkl` |
| `scaler.pkl`           | `SCALER_PATH`  | `/opt/areca/models/scaler.pkl`  |
| `features.json`        | `FEATURES_PATH`| `/opt/areca/models/features.json` |

> **Note:** `.pkl` and `.joblib` files are in `.gitignore`. Never commit model binaries.

**Run manually:**
```bash
python -m ml_engine.train_predict
```

---

## 8. Backend API

**File:** `backend/lambda_function.py`

- Built with **FastAPI 0.111** + **Uvicorn**
- Wrapped with **Mangum** for AWS Lambda compatibility
- `app` object is the ASGI application; `handler = Mangum(app)` is the Lambda handler

### Endpoints

| Method | Path                         | Description                         |
|--------|------------------------------|-------------------------------------|
| GET    | `/api/v1/health`             | Health check + DB ping              |
| GET    | `/api/v1/prices/current`     | Latest prices from `mv_latest_prices` |
| GET    | `/api/v1/prices/forecast`    | ML predictions (7 & 30-day)         |
| GET    | `/api/v1/prices/history`     | Historical `market_prices` time series |
| GET    | `/api/v1/markets`            | All active markets                  |
| GET    | `/api/v1/varieties`          | All active varieties                |

Interactive docs: `http://localhost:8000/api/docs` (Swagger), `/api/redoc` (ReDoc)

### CORS
Configured via `AppConfig.cors_origins` (env: `CORS_ORIGINS`). Default allows `localhost:8080`.

### Lambda cold-start optimisation
The DB connection pool is a module-level singleton to survive warm starts. Do not move it inside endpoint functions.

---

## 9. Frontend

**File:** `frontend/server.py`

- Serves Jinja2 HTML templates from `frontend/templates/`
- Static assets from `frontend/static/`
- Proxies API calls to the backend at `BACKEND_API_URL` (default: `http://localhost:8000/api/v1`)
- Runs on port `8080` (env: `FRONTEND_PORT`)

---

## 10. Infrastructure

**File:** `infra/deploy_infra.py`

Pure-boto3, **no Terraform**. Uses a "Flocci" pattern — declarative resource specs executed by boto3.

Resources provisioned (all idempotent — safe to re-run):
1. VPC (`10.42.0.0/16`) + subnets + Internet Gateway + route tables
2. Security groups (EC2, RDS, Lambda)
3. RDS PostgreSQL 15 (`db.t3.micro` by default)
4. EC2 (`t3.medium`) for frontend + Grafana
5. IAM role + inline policy for Lambda
6. Lambda function (`python3.11` runtime, FastAPI via Mangum)
7. API Gateway HTTP API → Lambda integration
8. EventBridge rules:
   - ETL daily at `06:00 UTC`
   - ML training weekly on Sunday
9. CloudWatch Log Groups

**Run:**
```bash
python -m infra.deploy_infra
```

**AWS region:** `ap-south-1` (Mumbai). Change via `AWS_REGION`.

---

## 11. Coding Conventions

### Python style
- **Python 3.10+** minimum. Use `match/case` only where it improves clarity.
- **Type hints** on all function signatures and class attributes.
- **Docstrings** on every public function, class, and module (Google style).
- **PEP 8** — 4-space indentation, max 100 chars per line.
- No bare `except:` — always catch specific exception types.
- Use `Decimal` for all monetary values. Never use `float` for prices.

### Imports
Standard library → third-party → local, separated by blank lines. No wildcard imports.

### Adding new config
1. Add a field to the relevant dataclass in `config/settings.py`.
2. Add the env var with a description to `.env.example`.
3. Document it in `README.md` under Environment Variables.

### Adding new API endpoints
1. Add the route to `backend/lambda_function.py`.
2. Update the endpoints table in both `README.md` and `AGENTS.md`.
3. Add a Pydantic response model — never return raw dicts from endpoints.

### Adding new DB tables/columns
1. Update `database/schema.sql` first.
2. Re-run `python -m database.init_db` (it uses `IF NOT EXISTS`).
3. Update `db_manager.py` with any new query helpers needed.

---

## 12. Environment & Secrets

- **Never** commit `.env` — it is in `.gitignore`.
- **Never** hardcode credentials, API keys, or AWS account IDs anywhere in code.
- All secrets are read exclusively via `config/settings.py` from environment variables.
- For local dev, copy `.env.example` → `.env` and fill in values.
- `.env.example` must always be kept up to date — it is the contract for all required env vars.

---

## 13. What NOT to Do

| ❌ Don't                                       | ✅ Do instead                                        |
|-----------------------------------------------|-----------------------------------------------------|
| Use `print()` for logging                      | Use `get_logger(__name__)`                          |
| Read `os.getenv()` outside `config/settings.py`| Add a field to the appropriate config dataclass     |
| Use `float` for price values                   | Use `Decimal` or `NUMERIC(10,2)` in SQL             |
| Open raw `psycopg2` connections                | Use `db_manager` singleton                          |
| Insert into `market_prices` without upsert    | Use `ON CONFLICT DO NOTHING` or upsert logic        |
| Commit `.env`, `.pkl`, or `.log` files        | They are in `.gitignore` — keep it that way         |
| Add Terraform or CDK files                    | Use `infra/deploy_infra.py` (pure boto3)            |
| Return raw `dict` from API endpoints          | Define and return a Pydantic response model         |
| Use `float` for `horizon_days`                | It must be `int` and only `7` or `30`              |
| Modify schema by running SQL directly on prod | Update `schema.sql` then run `init_db.py`           |

---

## 14. Running Locally — Quick Reference

```bash
# 1. Clone + virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Configure env
cp .env.example .env   # then edit .env

# 4. Start Postgres (Docker)
docker run -d --name areca-postgres \
  -e POSTGRES_DB=areca_db -e POSTGRES_USER=areca_admin \
  -e POSTGRES_PASSWORD=your_password -p 5432:5432 postgres:15

# 5. Full first-time setup
python scripts/run_local.py --init-db --etl --train

# 6. Day-to-day development
python scripts/run_local.py

# URLs
# Frontend  → http://localhost:8080
# API Docs  → http://localhost:8000/api/docs
# Grafana   → http://localhost:3000
```

---

## 15. Key External Dependencies

| Service / Library   | Purpose                          | Docs / Notes                              |
|---------------------|----------------------------------|-------------------------------------------|
| `fastapi`           | REST API framework               | https://fastapi.tiangolo.com              |
| `mangum`            | Lambda ASGI adapter              | https://mangum.io                         |
| `lightgbm`          | Gradient boosting ML model       | https://lightgbm.readthedocs.io           |
| `psycopg2-binary`   | PostgreSQL driver                | Thread-safe; binary for portability       |
| `sqlalchemy`        | ORM + connection pooling         | v2.0 async-compatible                     |
| `boto3`             | AWS SDK for infrastructure       | Region: `ap-south-1`                      |
| `beautifulsoup4`    | HTML scraping (Agmarknet)        | Parser: `lxml`                            |
| `open-meteo`        | Free weather API (no key needed) | https://open-meteo.com                    |
| `python-dotenv`     | Load `.env` into `os.environ`    | Loaded implicitly via `config/settings.py`|
| `rich`              | CLI output formatting            | Used in scripts for human-friendly output |
| `prometheus-client` | Metrics exposure                 | Used with Grafana monitoring              |

---

## 16. Grafana

Dashboards and provisioning configs live in `grafana/`. Grafana reads prices directly from the PostgreSQL database via the `areca-rds-postgres` datasource (UID defined in `GrafanaConfig.datasource_uid`).

To refresh materialized views after new data is ingested, call:
```sql
SELECT refresh_materialized_views();
```

Or it runs automatically at the end of a successful ETL run.
