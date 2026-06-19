package orders

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func TestFailedOrderIDUnique(t *testing.T) {
	a := FailedOrderID()
	b := FailedOrderID()
	if a == b {
		t.Fatalf("expected unique failed order IDs, got %q", a)
	}
	if !strings.HasPrefix(a, "failed-") || !strings.Contains(a, "-") {
		t.Fatalf("unexpected failed order ID format: %q", a)
	}
}

func TestInsertRoundtrip(t *testing.T) {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		t.Skip("DATABASE_URL not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	store, err := NewStore(ctx, databaseURL)
	if err != nil {
		t.Skipf("database not reachable: %v", err)
	}
	defer store.Close(context.Background())

	orderID := FailedOrderID()
	qty := 0.001
	px := 68000.0
	targetTS := time.Date(2026, 6, 12, 13, 0, 0, 0, time.UTC)

	err = store.Insert(ctx, Record{
		ExchangeOrderID: orderID,
		Mode:            "paper",
		MarketType:      "spot",
		Symbol:          "BTCUSDT",
		Side:            "BUY",
		OrderKind:       "open",
		OrderType:       "market",
		Status:          StatusFilled,
		Quantity:        &qty,
		Price:           &px,
		Strategy:        "test",
		Signal:          "open_long",
		Action:          "open",
		TargetTS:        &targetTS,
	})
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Idempotent re-insert
	if err := store.Insert(ctx, Record{
		ExchangeOrderID: orderID,
		Mode:            "paper",
		MarketType:      "spot",
		Symbol:          "BTCUSDT",
		Side:            "BUY",
		OrderKind:       "open",
		OrderType:       "market",
		Status:          StatusFilled,
	}); err != nil {
		t.Fatalf("re-insert: %v", err)
	}
}
