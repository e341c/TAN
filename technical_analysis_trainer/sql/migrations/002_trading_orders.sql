-- Trading bot order history (idempotent key: mode + exchange_order_id)

CREATE TABLE IF NOT EXISTS trading_orders (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    exchange_order_id   VARCHAR(64) NOT NULL,
    mode                VARCHAR(16) NOT NULL,
    market_type         VARCHAR(16) NOT NULL,

    symbol              VARCHAR(32) NOT NULL,
    side                VARCHAR(8) NOT NULL,
    order_kind          VARCHAR(24) NOT NULL,
    order_type          VARCHAR(24) NOT NULL,
    status              VARCHAR(16) NOT NULL DEFAULT 'filled',

    quantity            DOUBLE PRECISION,
    price               DOUBLE PRECISION,
    quote_usdt          DOUBLE PRECISION,

    stop_loss           DOUBLE PRECISION,
    take_profit         DOUBLE PRECISION,

    strategy            VARCHAR(64),
    signal              VARCHAR(16),
    action              VARCHAR(24),
    reason              TEXT,
    target_ts           TIMESTAMPTZ,

    error_message       TEXT,
    trade_id            UUID,

    UNIQUE (mode, exchange_order_id)
);

CREATE INDEX IF NOT EXISTS idx_trading_orders_created_at
    ON trading_orders (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trading_orders_symbol
    ON trading_orders (symbol, created_at DESC);
