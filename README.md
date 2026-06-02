# 🌿 Areca Nut Price Prediction System

A production-grade cloud-based platform for monitoring Karnataka areca nut market prices and generating ML-powered price forecasts. Designed for farmers, traders, and agri-market analysts.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Cloud Deployment (AWS)](#cloud-deployment-aws)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [ML Engine](#ml-engine)
- [Monitoring (Grafana)](#monitoring-grafana)
- [ETL Pipeline](#etl-pipeline)

---

## Overview

This system scrapes areca nut market price data, stores it in a PostgreSQL database, trains a LightGBM forecasting model, and serves predictions via a FastAPI backend. A Jinja2-based frontend and Grafana dashboards provide visualization.

**Key capabilities:**
- Real-time market price ingestion via an ETL pipeline
- 7-day and 30-day ML price forecasts
- REST API with OpenAPI documentation
- Grafana dashboards for price analytics
- AWS-native deployment (Lambda + RDS + EC2 + API Gateway)

---

## Architecture

```
Internet ──► API Gateway ──► Lambda (FastAPI/Mangum)
                                    │
                              ┌─────┴─────┐
                              │  RDS      │  ← ETL writes prices
                              │ PostgreSQL│  ← ML reads history
                              └───────────┘

EC2 Instance:
  ├── Frontend server  (port 8080)
  └── Grafana          (port 3000)

EventBridge Rules:
  ├── ETL  job  → daily at 06:00 UTC
  └── ML train → weekly on Sunday
```

---

## Tech Stack

| Layer        | Technology                               |
|--------------|------------------------------------------|
| API          | FastAPI 0.111, Uvicorn, Mangum           |
| Database     | PostgreSQL 15 (AWS RDS / local Docker)   |
| ORM          | SQLAlchemy 2.0, Alembic                  |
| ML           | LightGBM 4.4, scikit-learn, pandas, numpy|
| ETL          | requests, BeautifulSoup4, lxml           |
| Frontend     | Jinja2 templates, static files           |
| Infra        | boto3 (VPC, EC2, RDS, Lambda, API GW)   |
| Monitoring   | Grafana, Prometheus client               |
| Config       | python-dotenv, pydantic-settings         |

---

## Project Structure

```
areca-price-system/
├── backend/
│   └── lambda_function.py   # FastAPI app (Lambda-compatible via Mangum)
├── config/
│   ├── settings.py          # Pydantic settings from env vars
│   └── logging_config.py    # Structured logging setup
├── database/
│   ├── schema.sql            # DB schema (tables, indexes, views)
│   ├── init_db.py            # Schema initialisation script
│   └── db_manager.py         # Connection pool + query helpers
├── etl/
│   └── etl_pipeline.py      # Data scraping & ingestion pipeline
├── frontend/
│   ├── server.py            # Uvicorn/Jinja2 frontend server
│   ├── templates/           # HTML templates
│   └── static/              # CSS, JS, assets
├── grafana/
│   ├── dashboards/          # Dashboard JSON definitions
│   └── provisioning/        # Grafana datasource/dashboard provisioning
├── infra/
│   └── deploy_infra.py      # Boto3 AWS infrastructure provisioner
├── ml_engine/
│   ├── feature_engineering.py  # Feature creation for model training
│   └── train_predict.py        # LightGBM training + prediction
├── scripts/
│   └── run_local.py         # Local dev runner (starts all services)
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Prerequisites

| Requirement        | Version / Notes                           |
|--------------------|-------------------------------------------|
| Python             | 3.10 or 3.11 recommended                  |
| PostgreSQL         | 15.x (local install **or** Docker)        |
| pip                | Latest                                    |
| AWS CLI            | Only needed for cloud deployment          |
| Docker (optional)  | Easiest way to run PostgreSQL locally     |

---

## Local Development Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd areca-price-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values (see [Environment Variables](#environment-variables) for details). At minimum, set the database connection fields:

```env
RDS_HOST=localhost
RDS_PORT=5432
RDS_DB_NAME=areca_db
RDS_USER=areca_admin
RDS_PASSWORD=your_password
```

### 5. Start a local PostgreSQL instance

**Option A — Docker (recommended):**

```bash
docker run -d \
  --name areca-postgres \
  -e POSTGRES_DB=areca_db \
  -e POSTGRES_USER=areca_admin \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:15
```

**Option B — Native PostgreSQL:**

```bash
sudo -u postgres psql -c "CREATE USER areca_admin WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "CREATE DATABASE areca_db OWNER areca_admin;"
```

### 6. Initialise the database schema

```bash
python scripts/run_local.py --init-db
```

Or directly:

```bash
python -m database.init_db
```

### 7. Run the ETL pipeline (populate initial data)

```bash
python scripts/run_local.py --etl
```

Or to backfill the last N days:

```bash
python -m etl.etl_pipeline --backfill 30
```

### 8. Train the ML model

```bash
python scripts/run_local.py --train
```

Or directly:

```bash
python -m ml_engine.train_predict
```

### 9. Start all services

```bash
python scripts/run_local.py
```

This starts:
- **Backend API** → `http://localhost:8000`
- **Frontend** → `http://localhost:8080`

**Additional flags:**

```bash
# Init DB + run ETL + train model + start all services
python scripts/run_local.py --init-db --etl --train

# Start only the backend (skip frontend)
python scripts/run_local.py --no-frontend
```

### 10. Verify services

| Service        | URL                                  |
|----------------|--------------------------------------|
| Frontend       | http://localhost:8080                |
| API Docs       | http://localhost:8000/api/docs       |
| API ReDoc      | http://localhost:8000/api/redoc      |
| Grafana        | http://localhost:3000                |

---

## Cloud Deployment (AWS)

> **Warning:** This will provision real AWS resources and may incur costs.

### 1. Configure AWS credentials

```bash
aws configure
# Or set in .env:
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=ap-south-1
```

### 2. Fill in all AWS-specific `.env` values

Set your `AWS_ACCOUNT_ID`, `EC2_KEY_PAIR`, `LAMBDA_ROLE_ARN`, and any other required fields in `.env`.

### 3. Run the infrastructure provisioner

```bash
python -m infra.deploy_infra
```

This idempotently provisions:

1. VPC, subnets, internet gateway, route tables
2. Security groups (EC2, RDS, Lambda)
3. RDS PostgreSQL instance
4. EC2 instance (frontend + Grafana)
5. IAM role & policy for Lambda execution
6. Lambda function (FastAPI via Mangum)
7. API Gateway HTTP API → Lambda integration
8. EventBridge rules for scheduled ETL & ML training
9. CloudWatch log groups

### 4. Post-deployment

- SSH into the EC2 instance and run the initial ETL + training steps.
- Access the API via the API Gateway URL printed at the end of the provisioner output.

---

## Environment Variables

All configuration is driven by `.env`. Copy `.env.example` to `.env` and populate the fields.

| Variable               | Description                                        | Required |
|------------------------|----------------------------------------------------|----------|
| `APP_ENV`              | `production` or `development`                      | ✅       |
| `SECRET_KEY`           | Random secret for signing                          | ✅       |
| `RDS_HOST`             | PostgreSQL host                                    | ✅       |
| `RDS_PORT`             | PostgreSQL port (default: `5432`)                  | ✅       |
| `RDS_DB_NAME`          | Database name                                      | ✅       |
| `RDS_USER`             | Database user                                      | ✅       |
| `RDS_PASSWORD`         | Database password                                  | ✅       |
| `AWS_REGION`           | AWS region (e.g. `ap-south-1`)                     | AWS only |
| `AWS_ACCOUNT_ID`       | 12-digit AWS account ID                            | AWS only |
| `AWS_ACCESS_KEY_ID`    | AWS access key                                     | AWS only |
| `AWS_SECRET_ACCESS_KEY`| AWS secret key                                     | AWS only |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password                           | ✅       |
| `MODEL_PATH`           | Path to saved LightGBM model `.pkl`                | ✅       |
| `FRONTEND_PORT`        | Frontend server port (default: `8080`)             | ✅       |
| `API_PORT`             | Backend API port (default: `8000`)                 | ✅       |

See `.env.example` for the full list with descriptions.

---

## API Reference

The backend exposes a versioned REST API at `/api/v1/`. Interactive docs are available at `/api/docs` (Swagger UI) and `/api/redoc`.

| Method | Endpoint                     | Description                              |
|--------|------------------------------|------------------------------------------|
| GET    | `/api/v1/health`             | Health check                             |
| GET    | `/api/v1/prices/current`     | Latest market prices per variety         |
| GET    | `/api/v1/prices/forecast`    | ML price forecasts (7 & 30-day)          |
| GET    | `/api/v1/prices/history`     | Historical price time series             |
| GET    | `/api/v1/markets`            | List of configured markets               |
| GET    | `/api/v1/varieties`          | List of areca nut varieties              |

---

## ML Engine

The ML engine (`ml_engine/`) uses a **LightGBM** gradient boosting model.

| File                    | Purpose                                          |
|-------------------------|--------------------------------------------------|
| `feature_engineering.py`| Builds time-series & weather features            |
| `train_predict.py`      | Model training, evaluation, and inference        |

**Train manually:**

```bash
python -m ml_engine.train_predict
```

Key hyperparameters (configurable via `.env`):

| Parameter        | Default | Env Var            |
|------------------|---------|--------------------|
| `n_estimators`   | 500     | `N_ESTIMATORS`     |
| `learning_rate`  | 0.05    | `LEARNING_RATE`    |
| `num_leaves`     | 63      | `NUM_LEAVES`       |
| Min training rows| 90      | `MIN_TRAINING_ROWS`|

---

## Monitoring (Grafana)

Grafana dashboards are provisioned automatically via files in `grafana/provisioning/`.

**Start Grafana (local):**

```bash
docker run -d \
  --name areca-grafana \
  -p 3000:3000 \
  -v $(pwd)/grafana/provisioning:/etc/grafana/provisioning \
  -v $(pwd)/grafana/dashboards:/var/lib/grafana/dashboards \
  -e GF_SECURITY_ADMIN_PASSWORD=your_grafana_password \
  grafana/grafana:latest
```

Access at: **http://localhost:3000** (default credentials: `admin` / value of `GRAFANA_ADMIN_PASSWORD`)

---

## ETL Pipeline

The ETL pipeline (`etl/etl_pipeline.py`) scrapes areca nut market price data and stores it in PostgreSQL.

**Run manually:**

```bash
# Run for yesterday
python -m etl.etl_pipeline

# Backfill last 30 days
python -m etl.etl_pipeline --backfill 30
```

**Configuration (via `.env`):**

| Variable        | Default | Description                   |
|-----------------|---------|-------------------------------|
| `SCRAPE_TIMEOUT`| 30      | HTTP request timeout (seconds) |
| `MAX_RETRIES`   | 3       | Retry attempts on failure      |
| `RETRY_DELAY`   | 5       | Delay between retries (seconds)|
| `ETL_BATCH_SIZE`| 100     | Records per DB insert batch    |

In production, ETL runs daily at 06:00 UTC via an EventBridge rule automatically configured by the infrastructure provisioner.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow PEP 8 and use type hints throughout.
3. Add or update docstrings for any changed modules.
4. Test locally with `python scripts/run_local.py` before opening a PR.

---

## License

This project is for academic and research purposes.
