package strategy

import (
	"testing"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/predictions"
)

func TestEvaluateOpenLong(t *testing.T) {
	cfg := &config.Config{
		Market: config.MarketConfig{Type: "spot", PositionSide: "long"},
		Strategy: config.StrategyConfig{
			Entry: config.EntryRules{
				Long: config.RuleGroup{
					All: []config.Condition{
						{Field: "trend.predicted_trend", Op: "eq", Value: "UP"},
						{Field: "trend.probabilities.UP", Op: "gte", Value: 0.6},
					},
				},
			},
		},
	}
	snap := &predictions.Snapshot{
		Quantile: predictions.Quantile{Q10: 90, Q50: 100, Q90: 110},
		Trend: predictions.Trend{
			PredictedTrend: "UP",
			Probabilities:  map[string]float64{"UP": 0.7, "DOWN": 0.3},
		},
		Candles: []predictions.Candle{{Close: 100}},
	}

	sig, _ := New(cfg).Evaluate(snap, PositionNone)
	if sig != SignalOpenLong {
		t.Fatalf("expected OPEN_LONG, got %s", sig)
	}
}

func TestEvaluateExitLong(t *testing.T) {
	cfg := &config.Config{
		Market: config.MarketConfig{Type: "spot"},
		Strategy: config.StrategyConfig{
			Exit: config.ExitRules{
				Long: config.RuleGroup{
					Any: []config.Condition{
						{Field: "trend.predicted_trend", Op: "eq", Value: "DOWN"},
					},
				},
			},
		},
	}
	snap := &predictions.Snapshot{
		Trend: predictions.Trend{PredictedTrend: "DOWN"},
	}

	sig, _ := New(cfg).Evaluate(snap, PositionLong)
	if sig != SignalClose {
		t.Fatalf("expected CLOSE, got %s", sig)
	}
}
