-- ===========================================================================
-- database/schema.sql
-- Full DDL for the Areca Nut Price Prediction System
-- PostgreSQL 15+
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- fuzzy text search for market names
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- composite GIN indexes

-- ---------------------------------------------------------------------------
-- Lookup / Reference Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS markets (
    market_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_name     VARCHAR(128) NOT NULL,
    state           VARCHAR(64)  NOT NULL DEFAULT 'Karnataka',
    district        VARCHAR(64)  NOT NULL,
    apmc_code       VARCHAR(32)  UNIQUE,
    latitude        NUMERIC(9, 6),
    longitude       NUMERIC(9, 6),
    active          BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_market_name_district UNIQUE (market_name, district)
);

CREATE TABLE IF NOT EXISTS varieties (
    variety_id      SMALLSERIAL  PRIMARY KEY,
    variety_name    VARCHAR(64)  NOT NULL UNIQUE,
    local_name      VARCHAR(64),
    description     TEXT,
    active          BOOLEAN      NOT NULL DEFAULT TRUE
);

-- Seed reference data
INSERT INTO varieties (variety_name, local_name, description) VALUES
    ('Chali',  'ಚಾಲಿ',  'Dried and processed areca nut — most traded variety'),
    ('Gotu',   'ಗೋಟು',  'Raw whole areca nut with husk'),
    ('Kotte',  'ಕೊಟ್ಟೆ', 'Tender areca nut, semi-processed'),
    ('Rashi',  'ರಾಶಿ',  'Bulk heap grade — mixed quality'),
    ('Saraku', 'ಸರಕು',  'Commercial grade processed supari')
ON CONFLICT (variety_name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Core Fact Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS market_prices (
    id              BIGSERIAL    PRIMARY KEY,
    record_date     DATE         NOT NULL,
    market_id       UUID         NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    variety_id      SMALLINT     NOT NULL REFERENCES varieties(variety_id),
    min_price       NUMERIC(10, 2) NOT NULL CHECK (min_price >= 0),
    max_price       NUMERIC(10, 2) NOT NULL CHECK (max_price >= min_price),
    modal_price     NUMERIC(10, 2) NOT NULL CHECK (modal_price BETWEEN min_price AND max_price),
    arrivals_tons   NUMERIC(12, 3) CHECK (arrivals_tons >= 0),
    source          VARCHAR(64)  DEFAULT 'agmarknet',
    raw_payload     JSONB,
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_price_date_market_variety UNIQUE (record_date, market_id, variety_id)
);

CREATE TABLE IF NOT EXISTS weather_metrics (
    id              BIGSERIAL    PRIMARY KEY,
    record_date     DATE         NOT NULL,
    region_id       UUID         NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    rainfall_mm     NUMERIC(8, 2) CHECK (rainfall_mm >= 0),
    avg_humidity    NUMERIC(5, 2) CHECK (avg_humidity BETWEEN 0 AND 100),
    temperature_c   NUMERIC(5, 2) CHECK (temperature_c BETWEEN -10 AND 55),
    wind_speed_kmh  NUMERIC(6, 2) CHECK (wind_speed_kmh >= 0),
    source          VARCHAR(64)  DEFAULT 'open-meteo',
    ingested_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_weather_date_region UNIQUE (record_date, region_id)
);

CREATE TABLE IF NOT EXISTS price_predictions (
    id                  BIGSERIAL    PRIMARY KEY,
    prediction_run_id   UUID         NOT NULL DEFAULT uuid_generate_v4(),
    prediction_date     DATE         NOT NULL,   -- date the model was run
    target_date         DATE         NOT NULL,   -- date being predicted
    market_id           UUID         NOT NULL REFERENCES markets(market_id) ON DELETE CASCADE,
    variety_id          SMALLINT     NOT NULL REFERENCES varieties(variety_id),
    predicted_price     NUMERIC(10, 2) NOT NULL CHECK (predicted_price >= 0),
    confidence_lower    NUMERIC(10, 2) NOT NULL CHECK (confidence_lower >= 0),
    confidence_upper    NUMERIC(10, 2) NOT NULL CHECK (confidence_upper >= confidence_lower),
    horizon_days        SMALLINT     NOT NULL CHECK (horizon_days IN (7, 30)),
    model_version       VARCHAR(32)  NOT NULL DEFAULT '1.0.0',
    model_rmse          NUMERIC(10, 4),
    model_mae           NUMERIC(10, 4),
    feature_importance  JSONB,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_pred_run_target_market_variety
        UNIQUE (prediction_date, target_date, market_id, variety_id, horizon_days)
);

-- Model training audit
CREATE TABLE IF NOT EXISTS model_training_runs (
    id              BIGSERIAL    PRIMARY KEY,
    run_id          UUID         NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    variety_id      SMALLINT     REFERENCES varieties(variety_id),
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          VARCHAR(16)  NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'success', 'failed')),
    rows_trained    INTEGER,
    train_rmse      NUMERIC(10, 4),
    val_rmse        NUMERIC(10, 4),
    train_mae       NUMERIC(10, 4),
    val_mae         NUMERIC(10, 4),
    model_path      TEXT,
    error_message   TEXT,
    hyperparameters JSONB,
    feature_list    JSONB
);

-- ETL run audit
CREATE TABLE IF NOT EXISTS etl_runs (
    id              BIGSERIAL    PRIMARY KEY,
    run_id          UUID         NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    source          VARCHAR(64)  NOT NULL,
    started_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          VARCHAR(16)  NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'success', 'partial', 'failed')),
    records_fetched INTEGER      DEFAULT 0,
    records_written INTEGER      DEFAULT 0,
    records_skipped INTEGER      DEFAULT 0,
    error_message   TEXT,
    metadata        JSONB
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

-- market_prices
CREATE INDEX IF NOT EXISTS idx_mp_record_date      ON market_prices (record_date DESC);
CREATE INDEX IF NOT EXISTS idx_mp_market_variety   ON market_prices (market_id, variety_id);
CREATE INDEX IF NOT EXISTS idx_mp_date_market_var  ON market_prices (record_date DESC, market_id, variety_id);
CREATE INDEX IF NOT EXISTS idx_mp_source           ON market_prices (source);

-- weather_metrics
CREATE INDEX IF NOT EXISTS idx_wm_record_date      ON weather_metrics (record_date DESC);
CREATE INDEX IF NOT EXISTS idx_wm_region_date      ON weather_metrics (region_id, record_date DESC);

-- price_predictions
CREATE INDEX IF NOT EXISTS idx_pp_prediction_date  ON price_predictions (prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_pp_target_date      ON price_predictions (target_date DESC);
CREATE INDEX IF NOT EXISTS idx_pp_market_variety   ON price_predictions (market_id, variety_id, horizon_days);
CREATE INDEX IF NOT EXISTS idx_pp_run_id           ON price_predictions (prediction_run_id);

-- ---------------------------------------------------------------------------
-- Materialized View: Latest prices per market/variety
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_latest_prices AS
SELECT DISTINCT ON (mp.market_id, mp.variety_id)
    mp.id,
    mp.record_date,
    m.market_name,
    m.district,
    v.variety_name,
    mp.min_price,
    mp.max_price,
    mp.modal_price,
    mp.arrivals_tons,
    mp.source,
    mp.ingested_at
FROM market_prices mp
JOIN markets       m ON m.market_id  = mp.market_id
JOIN varieties     v ON v.variety_id = mp.variety_id
ORDER BY mp.market_id, mp.variety_id, mp.record_date DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_latest_prices
    ON mv_latest_prices (market_name, variety_name);

-- ---------------------------------------------------------------------------
-- Materialized View: 30-day price stats per variety
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_variety_stats_30d AS
SELECT
    v.variety_name,
    m.market_name,
    COUNT(*)                         AS data_points,
    AVG(mp.modal_price)::NUMERIC(10,2) AS avg_modal_price,
    MIN(mp.modal_price)              AS min_modal_price,
    MAX(mp.modal_price)              AS max_modal_price,
    STDDEV(mp.modal_price)::NUMERIC(10,4) AS price_stddev,
    SUM(mp.arrivals_tons)            AS total_arrivals_tons,
    MAX(mp.record_date)              AS latest_date
FROM market_prices mp
JOIN varieties     v ON v.variety_id = mp.variety_id
JOIN markets       m ON m.market_id  = mp.market_id
WHERE mp.record_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY v.variety_name, m.market_name;

-- ---------------------------------------------------------------------------
-- Utility Functions
-- ---------------------------------------------------------------------------

-- Auto-update updated_at timestamp on markets table
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE TRIGGER set_timestamp_markets
    BEFORE UPDATE ON markets
    FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();

-- Convenience function: refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_latest_prices;
    REFRESH MATERIALIZED VIEW mv_variety_stats_30d;
END;
$$;
