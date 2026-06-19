package state

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/eriktabys/tan-model/trading_bot/internal/exchange"
)

func TestStoreSaveLoadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	store := NewStore(filepath.Join(dir, "state.json"))

	openedAt := time.Date(2026, 6, 10, 12, 0, 0, 0, time.UTC)
	original := &BotState{
		LastTargetTS: "2026-06-10T13:00:00Z",
		LastEntryAt:  openedAt,
		TradeID:      "550e8400-e29b-41d4-a716-446655440000",
		Position: &exchange.Position{
			Side:       exchange.SideBuy,
			Symbol:     "BTCUSDT",
			Quantity:   0.001,
			EntryPrice: 68000,
			OpenedAt:   openedAt,
			StopLoss:   67000,
			TakeProfit: 70000,
		},
	}

	if err := store.Save(original); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := store.Load()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if loaded.LastTargetTS != original.LastTargetTS {
		t.Fatalf("last_target_ts: got %q want %q", loaded.LastTargetTS, original.LastTargetTS)
	}
	if loaded.TradeID != original.TradeID {
		t.Fatalf("trade_id: got %q want %q", loaded.TradeID, original.TradeID)
	}
	if loaded.Position == nil {
		t.Fatal("expected position")
	}
	if loaded.Position.Symbol != "BTCUSDT" || loaded.Position.StopLoss != 67000 {
		t.Fatalf("unexpected position: %+v", loaded.Position)
	}
}

func TestStoreLoadMissingFile(t *testing.T) {
	store := NewStore(filepath.Join(t.TempDir(), "missing.json"))
	st, err := store.Load()
	if err != nil {
		t.Fatalf("load missing: %v", err)
	}
	if st.LastTargetTS != "" || st.Position != nil {
		t.Fatalf("expected empty state, got %+v", st)
	}
}
