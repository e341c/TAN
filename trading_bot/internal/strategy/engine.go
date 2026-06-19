package strategy

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/predictions"
)

type Signal string

const (
	SignalNone      Signal = "NONE"
	SignalOpenLong  Signal = "OPEN_LONG"
	SignalOpenShort Signal = "OPEN_SHORT"
	SignalClose     Signal = "CLOSE"
)

type PositionSide string

const (
	PositionNone  PositionSide = ""
	PositionLong  PositionSide = "LONG"
	PositionShort PositionSide = "SHORT"
)

type Engine struct {
	cfg *config.Config
}

func New(cfg *config.Config) *Engine {
	return &Engine{cfg: cfg}
}

func (e *Engine) Evaluate(snap *predictions.Snapshot, pos PositionSide) (Signal, string) {
	fields := buildFields(snap)

	if pos == PositionLong {
		if e.cfg.Strategy.Exit.Long.Any != nil || e.cfg.Strategy.Exit.Long.All != nil {
			if evalGroup(e.cfg.Strategy.Exit.Long, fields) {
				return SignalClose, "exit.long rules matched"
			}
		}
		return SignalNone, "holding long, no exit"
	}
	if pos == PositionShort {
		if e.cfg.Strategy.Exit.Short.Any != nil || e.cfg.Strategy.Exit.Short.All != nil {
			if evalGroup(e.cfg.Strategy.Exit.Short, fields) {
				return SignalClose, "exit.short rules matched"
			}
		}
		return SignalNone, "holding short, no exit"
	}

	canLong := e.cfg.Market.Type == "spot" ||
		e.cfg.Market.PositionSide == "long" ||
		e.cfg.Market.PositionSide == "both"
	if canLong && evalGroup(e.cfg.Strategy.Entry.Long, fields) {
		return SignalOpenLong, "entry.long rules matched"
	}

	canShort := e.cfg.Market.Type == "futures" &&
		(e.cfg.Market.PositionSide == "short" || e.cfg.Market.PositionSide == "both")
	if canShort && evalGroup(e.cfg.Strategy.Entry.Short, fields) {
		return SignalOpenShort, "entry.short rules matched"
	}

	return SignalNone, "no entry rules matched"
}

func buildFields(snap *predictions.Snapshot) map[string]interface{} {
	fields := map[string]interface{}{
		"trend.predicted_trend":      snap.Trend.PredictedTrend,
		"quantile.q10":               snap.Quantile.Q10,
		"quantile.q50":               snap.Quantile.Q50,
		"quantile.q90":               snap.Quantile.Q90,
		"quantile.spread_pct":        snap.SpreadPct(),
		"meta.last_close":            snap.LastClose(),
		"meta.symbol":                snap.Meta.Symbol,
		"meta.interval":              snap.Meta.Interval,
	}
	for k, v := range snap.Trend.Probabilities {
		fields["trend.probabilities."+k] = v
	}
	if len(snap.Trend.Logits) >= 2 {
		fields["trend.logits.0"] = snap.Trend.Logits[0]
		fields["trend.logits.1"] = snap.Trend.Logits[1]
	}
	return fields
}

func evalGroup(group config.RuleGroup, fields map[string]interface{}) bool {
	if len(group.All) > 0 {
		for _, c := range group.All {
			if !evalCondition(c, fields) {
				return false
			}
		}
		return true
	}
	if len(group.Any) > 0 {
		for _, c := range group.Any {
			if evalCondition(c, fields) {
				return true
			}
		}
		return false
	}
	return false
}

func evalCondition(c config.Condition, fields map[string]interface{}) bool {
	actual, ok := fields[c.Field]
	if !ok {
		return false
	}
	op := strings.ToLower(strings.TrimSpace(c.Op))
	switch op {
	case "eq":
		return compareEqual(actual, c.Value)
	case "ne":
		return !compareEqual(actual, c.Value)
	case "gt":
		return compareNumeric(actual, c.Value) > 0
	case "gte":
		return compareNumeric(actual, c.Value) >= 0
	case "lt":
		return compareNumeric(actual, c.Value) < 0
	case "lte":
		return compareNumeric(actual, c.Value) <= 0
	case "in":
		return valueIn(actual, c.Value)
	default:
		return false
	}
}

func compareEqual(a, b interface{}) bool {
	af, aOk := toFloat64(a)
	bf, bOk := toFloat64(b)
	if aOk && bOk {
		return af == bf
	}
	return fmt.Sprint(a) == fmt.Sprint(b)
}

func compareNumeric(a, b interface{}) int {
	af, aOk := toFloat64(a)
	bf, bOk := toFloat64(b)
	if !aOk || !bOk {
		as := strings.ToLower(fmt.Sprint(a))
		bs := strings.ToLower(fmt.Sprint(b))
		if as == bs {
			return 0
		}
		if as > bs {
			return 1
		}
		return -1
	}
	if af > bf {
		return 1
	}
	if af < bf {
		return -1
	}
	return 0
}

func valueIn(actual, allowed interface{}) bool {
	switch list := allowed.(type) {
	case []interface{}:
		for _, item := range list {
			if compareEqual(actual, item) {
				return true
			}
		}
	case []string:
		for _, item := range list {
			if compareEqual(actual, item) {
				return true
			}
		}
	}
	return false
}

func toFloat64(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		f, err := strconv.ParseFloat(n, 64)
		return f, err == nil
	default:
		return 0, false
	}
}
