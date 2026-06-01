-- Quantile + trend prediction tables (idempotent upsert key: symbol, timeframe, target_ts)

CREATE TABLE IF NOT EXISTS quantile_predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    candle_ts TIMESTAMPTZ NOT NULL,
    target_ts TIMESTAMPTZ NOT NULL,
    model_run_id VARCHAR(64),
    last_close DOUBLE PRECISION,
    q10 DOUBLE PRECISION NOT NULL,
    q50 DOUBLE PRECISION NOT NULL,
    q90 DOUBLE PRECISION NOT NULL,
    log_return_q10 DOUBLE PRECISION,
    log_return_q50 DOUBLE PRECISION,
    log_return_q90 DOUBLE PRECISION,
    UNIQUE (symbol, timeframe, target_ts)
);

CREATE TABLE IF NOT EXISTS trend_predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(16) NOT NULL,
    candle_ts TIMESTAMPTZ NOT NULL,
    target_ts TIMESTAMPTZ NOT NULL,
    model_run_id VARCHAR(64),
    trend_last_ts TIMESTAMPTZ,
    predicted_trend VARCHAR(16) NOT NULL,
    prob_down DOUBLE PRECISION,
    prob_up DOUBLE PRECISION,
    logits TEXT,
    UNIQUE (symbol, timeframe, target_ts)
);

CREATE INDEX IF NOT EXISTS idx_quantile_predictions_target_ts
    ON quantile_predictions (target_ts DESC);

CREATE INDEX IF NOT EXISTS idx_trend_predictions_target_ts
    ON trend_predictions (target_ts DESC);
