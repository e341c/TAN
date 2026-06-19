package exchange

import "testing"

func TestPaperCheckProtectiveExitLong(t *testing.T) {
	p := NewPaperExecutor()
	p.RestorePosition(&Position{
		Side:       SideBuy,
		Symbol:     "BTCUSDT",
		Quantity:   0.001,
		EntryPrice: 100,
		StopLoss:   90,
		TakeProfit: 110,
	})

	triggered, reason := p.CheckProtectiveExit(89)
	if !triggered || reason == "" {
		t.Fatalf("expected stop-loss trigger, got triggered=%v reason=%q", triggered, reason)
	}

	triggered, _ = p.CheckProtectiveExit(111)
	if !triggered {
		t.Fatal("expected take-profit trigger")
	}

	triggered, _ = p.CheckProtectiveExit(100)
	if triggered {
		t.Fatal("expected no trigger at entry price")
	}
}

func TestPaperCheckProtectiveExitShort(t *testing.T) {
	p := NewPaperExecutor()
	p.RestorePosition(&Position{
		Side:       SideSell,
		Symbol:     "BTCUSDT",
		Quantity:   0.001,
		EntryPrice: 100,
		StopLoss:   110,
		TakeProfit: 90,
	})

	triggered, _ := p.CheckProtectiveExit(111)
	if !triggered {
		t.Fatal("expected short stop-loss trigger")
	}

	triggered, _ = p.CheckProtectiveExit(89)
	if !triggered {
		t.Fatal("expected short take-profit trigger")
	}
}
