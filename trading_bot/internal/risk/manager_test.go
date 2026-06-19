package risk

import (
	"testing"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/predictions"
	"github.com/eriktabys/tan-model/trading_bot/internal/strategy"
)

func TestApproveOpenLong(t *testing.T) {
	cfg := &config.Config{
		Market: config.MarketConfig{Symbol: "BTCUSDT"},
		Risk: config.RiskConfig{
			QuoteSizeUSDT:    100,
			MaxOpenPositions: 1,
			StopLoss:         config.PriceLevelConfig{Type: "quantile_band", LongBand: "q10"},
			TakeProfit:       config.PriceLevelConfig{Type: "quantile_band", LongBand: "q90"},
		},
	}
	snap := &predictions.Snapshot{
		Quantile: predictions.Quantile{Q10: 90, Q50: 100, Q90: 110, NextTime: "2026-06-10T15:00:00Z"},
		Candles:  []predictions.Candle{{Close: 100}},
	}
	m := New(cfg)
	req, err := m.Approve(strategy.SignalOpenLong, snap, nil)
	if err != nil {
		t.Fatalf("approve: %v", err)
	}
	if req == nil || req.QuoteUSDT != 100 {
		t.Fatalf("unexpected request: %+v", req)
	}
	if req.StopLoss != 90 || req.TakeProfit != 110 {
		t.Fatalf("unexpected SL/TP: %f %f", req.StopLoss, req.TakeProfit)
	}
}
