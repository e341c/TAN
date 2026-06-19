package orders

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
)

const (
	StatusFilled = "filled"
	StatusFailed = "failed"
)

type Record struct {
	ExchangeOrderID string
	Mode            string
	MarketType      string
	Symbol          string
	Side            string
	OrderKind       string
	OrderType       string
	Status          string
	Quantity        *float64
	Price           *float64
	QuoteUSDT       *float64
	StopLoss        *float64
	TakeProfit      *float64
	Strategy        string
	Signal          string
	Action          string
	Reason          string
	TargetTS        *time.Time
	ErrorMessage    string
	TradeID         string
}

type Store struct {
	conn *pgx.Conn
}

func NewStore(ctx context.Context, databaseURL string) (*Store, error) {
	conn, err := pgx.Connect(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("orders db connect: %w", err)
	}
	return &Store{conn: conn}, nil
}

func (s *Store) Close(ctx context.Context) error {
	if s.conn == nil {
		return nil
	}
	return s.conn.Close(ctx)
}

func (s *Store) Insert(ctx context.Context, r Record) error {
	const q = `
INSERT INTO trading_orders (
    exchange_order_id, mode, market_type,
    symbol, side, order_kind, order_type, status,
    quantity, price, quote_usdt,
    stop_loss, take_profit,
    strategy, signal, action, reason, target_ts,
    error_message, trade_id
) VALUES (
    $1, $2, $3,
    $4, $5, $6, $7, $8,
    $9, $10, $11,
    $12, $13,
    $14, $15, $16, $17, $18,
    $19, $20::uuid
)
ON CONFLICT (mode, exchange_order_id) DO NOTHING`

	var tradeID any
	if r.TradeID != "" {
		tradeID = r.TradeID
	}

	_, err := s.conn.Exec(ctx, q,
		r.ExchangeOrderID, r.Mode, r.MarketType,
		r.Symbol, r.Side, r.OrderKind, r.OrderType, r.Status,
		r.Quantity, r.Price, r.QuoteUSDT,
		r.StopLoss, r.TakeProfit,
		nullIfEmpty(r.Strategy), nullIfEmpty(r.Signal), nullIfEmpty(r.Action), nullIfEmpty(r.Reason), r.TargetTS,
		nullIfEmpty(r.ErrorMessage), tradeID,
	)
	if err != nil {
		return fmt.Errorf("insert trading order: %w", err)
	}
	return nil
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func FloatPtr(v float64) *float64 {
	return &v
}

var failedOrderSeq atomic.Uint64

func FailedOrderID() string {
	return fmt.Sprintf("failed-%d-%d", time.Now().UnixNano(), failedOrderSeq.Add(1))
}
