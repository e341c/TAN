package bot

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/exchange"
	"github.com/eriktabys/tan-model/trading_bot/internal/orders"
	"github.com/eriktabys/tan-model/trading_bot/internal/strategy"
)

type mockOrderWriter struct {
	records []orders.Record
}

func (m *mockOrderWriter) Insert(_ context.Context, r orders.Record) error {
	m.records = append(m.records, r)
	return nil
}

func testBot(orderStore orders.Writer) *Bot {
	return &Bot{
		cfg: &config.Config{
			Bot:    config.BotConfig{Mode: "paper"},
			Market: config.MarketConfig{Type: "spot", Symbol: "BTCUSDT"},
			Strategy: config.StrategyConfig{
				Name: "test_strategy",
			},
		},
		executor:   exchange.NewPaperExecutor(),
		orderStore: orderStore,
	}
}

func TestPersistFilledOrderOpen(t *testing.T) {
	mock := &mockOrderWriter{}
	b := testBot(mock)
	targetTS := time.Date(2026, 6, 12, 13, 0, 0, 0, time.UTC)

	b.persistFilledOrder(
		context.Background(),
		strategy.SignalOpenLong,
		"open",
		&exchange.OrderResult{
			Symbol:   "BTCUSDT",
			Side:     exchange.SideBuy,
			Quantity: 0.001,
			Price:    68000,
			OrderID:  "paper-123",
			Mode:     "paper",
		},
		&targetTS,
		"open",
		"market",
		"trade-uuid-1",
	)

	if len(mock.records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(mock.records))
	}
	r := mock.records[0]
	if r.Status != orders.StatusFilled || r.ExchangeOrderID != "paper-123" || r.TradeID != "trade-uuid-1" {
		t.Fatalf("unexpected record: %+v", r)
	}
}

func TestPersistFailedOrderOpen(t *testing.T) {
	mock := &mockOrderWriter{}
	b := testBot(mock)
	targetTS := time.Date(2026, 6, 12, 13, 0, 0, 0, time.UTC)

	b.persistFailedOrder(
		context.Background(),
		strategy.SignalOpenLong,
		"open",
		&exchange.OrderRequest{
			Symbol:    "BTCUSDT",
			Side:      exchange.SideBuy,
			QuoteUSDT: 100,
			Price:     68000,
		},
		nil,
		68000,
		&targetTS,
		errors.New("spot open order: insufficient balance"),
	)

	if len(mock.records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(mock.records))
	}
	r := mock.records[0]
	if r.Status != orders.StatusFailed {
		t.Fatalf("expected failed status, got %q", r.Status)
	}
	if r.ErrorMessage == "" {
		t.Fatal("expected error message")
	}
}
