"""
config/settings.py
==================
Central configuration for the Areca Nut Price Prediction System.
All sensitive values are loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
@dataclass
class DBConfig:
    host: str = os.getenv("RDS_HOST", "localhost")
    port: int = int(os.getenv("RDS_PORT", "5432"))
    name: str = os.getenv("RDS_DB_NAME", "areca_db")
    user: str = os.getenv("RDS_USER", "areca_admin")
    password: str = os.getenv("RDS_PASSWORD", "change_me_in_production")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def psycopg2_dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.name} "
            f"user={self.user} password={self.password} "
            f"connect_timeout=10 sslmode=prefer"
        )


# ---------------------------------------------------------------------------
# AWS / Flocci Infrastructure
# ---------------------------------------------------------------------------
@dataclass
class AWSConfig:
    region: str = os.getenv("AWS_REGION", "ap-south-1")
    account_id: str = os.getenv("AWS_ACCOUNT_ID", "123456789012")

    # EC2
    ec2_instance_type: str = os.getenv("EC2_INSTANCE_TYPE", "t3.medium")
    ec2_ami_id: str = os.getenv("EC2_AMI_ID", "ami-0f58b397bc5c1f2e8")  # Ubuntu 22.04 ap-south-1
    ec2_key_pair: str = os.getenv("EC2_KEY_PAIR", "areca-keypair")
    ec2_instance_id: Optional[str] = os.getenv("EC2_INSTANCE_ID", None)

    # RDS
    rds_instance_class: str = os.getenv("RDS_INSTANCE_CLASS", "db.t3.micro")
    rds_allocated_storage: int = int(os.getenv("RDS_ALLOCATED_STORAGE", "20"))
    rds_engine_version: str = os.getenv("RDS_ENGINE_VERSION", "15.4")
    rds_identifier: str = os.getenv("RDS_IDENTIFIER", "areca-price-db")
    rds_multi_az: bool = os.getenv("RDS_MULTI_AZ", "false").lower() == "true"

    # Lambda
    lambda_function_name: str = os.getenv("LAMBDA_FUNCTION_NAME", "areca-price-api")
    lambda_runtime: str = "python3.11"
    lambda_timeout: int = int(os.getenv("LAMBDA_TIMEOUT", "30"))
    lambda_memory: int = int(os.getenv("LAMBDA_MEMORY", "512"))
    lambda_role_arn: str = os.getenv(
        "LAMBDA_ROLE_ARN",
        "arn:aws:iam::123456789012:role/areca-lambda-execution-role",
    )

    # VPC / Networking
    vpc_id: str = os.getenv("VPC_ID", "")
    subnet_ids: List[str] = field(
        default_factory=lambda: os.getenv("SUBNET_IDS", "").split(",")
    )
    public_subnet_ids: List[str] = field(
        default_factory=lambda: os.getenv("PUBLIC_SUBNET_IDS", "").split(",")
    )

    # API Gateway
    api_gateway_name: str = os.getenv("API_GATEWAY_NAME", "areca-price-api-gateway")
    api_stage: str = os.getenv("API_STAGE", "prod")


# ---------------------------------------------------------------------------
# ETL / Scraping
# ---------------------------------------------------------------------------
@dataclass
class ETLConfig:
    agmarknet_base_url: str = "https://agmarknet.gov.in"
    agmarknet_price_url: str = (
        "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    )
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "")
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"

    # Karnataka areca nut markets
    target_markets: List[str] = field(
        default_factory=lambda: [
            "Shimoga",
            "Sagara",
            "Thirthahalli",
            "Sagar",
            "Mudigere",
            "Chikkamagaluru",
            "Mangaluru",
            "Hassan",
            "Puttur",
            "Bantwal",
        ]
    )

    target_varieties: List[str] = field(
        default_factory=lambda: ["Chali", "Gotu", "Kotte", "Rashi", "Saraku"]
    )

    # Coordinates for weather (Karnataka areca belt)
    weather_locations: dict = field(
        default_factory=lambda: {
            "Shimoga": {"lat": 13.9299, "lon": 75.5681},
            "Sagara": {"lat": 14.1673, "lon": 75.0267},
            "Mudigere": {"lat": 13.1333, "lon": 75.6333},
            "Chikkamagaluru": {"lat": 13.3153, "lon": 75.7754},
            "Mangaluru": {"lat": 12.9141, "lon": 74.8560},
        }
    )

    scrape_timeout_sec: int = int(os.getenv("SCRAPE_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay_sec: int = int(os.getenv("RETRY_DELAY", "5"))
    batch_size: int = int(os.getenv("ETL_BATCH_SIZE", "100"))


# ---------------------------------------------------------------------------
# ML Engine
# ---------------------------------------------------------------------------
@dataclass
class MLConfig:
    model_path: str = os.getenv("MODEL_PATH", "/opt/areca/models/lgbm_areca.pkl")
    scaler_path: str = os.getenv("SCALER_PATH", "/opt/areca/models/scaler.pkl")
    features_path: str = os.getenv("FEATURES_PATH", "/opt/areca/models/features.json")
    forecast_horizons: List[int] = field(default_factory=lambda: [7, 30])
    min_training_rows: int = int(os.getenv("MIN_TRAINING_ROWS", "90"))
    test_split_ratio: float = float(os.getenv("TEST_SPLIT_RATIO", "0.15"))
    n_estimators: int = int(os.getenv("N_ESTIMATORS", "500"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.05"))
    num_leaves: int = int(os.getenv("NUM_LEAVES", "63"))
    confidence_alpha: float = float(os.getenv("CONFIDENCE_ALPHA", "0.10"))
    lag_days: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 21, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    lgbm_objective: str = "regression"
    lgbm_metric: str = "rmse"
    n_quantile_estimators: int = 100  # for confidence intervals


# ---------------------------------------------------------------------------
# Grafana
# ---------------------------------------------------------------------------
@dataclass
class GrafanaConfig:
    host: str = os.getenv("GRAFANA_HOST", "localhost")
    port: int = int(os.getenv("GRAFANA_PORT", "3000"))
    admin_user: str = os.getenv("GRAFANA_ADMIN_USER", "admin")
    admin_password: str = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin_secure_pass")
    datasource_uid: str = "areca-rds-postgres"
    dashboard_uid: str = "areca-price-dashboard"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def embed_url(self) -> str:
        return (
            f"{self.base_url}/d/{self.dashboard_uid}"
            f"?orgId=1&kiosk=tv&theme=dark"
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    environment: str = os.getenv("APP_ENV", "production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "/var/log/areca")
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "8080"))
    api_port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS", "http://localhost:8080,http://0.0.0.0:8080"
        ).split(",")
    )
    secret_key: str = os.getenv("SECRET_KEY", "change-this-secret-key-immediately")


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------
db = DBConfig()
aws = AWSConfig()
etl = ETLConfig()
ml = MLConfig()
grafana = GrafanaConfig()
app = AppConfig()
