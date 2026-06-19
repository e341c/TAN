package risk

import (
	"fmt"
	"time"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/exchange"
	"github.com/eriktabys/tan-model/trading_bot/internal/predictions"
	"github.com/eriktabys/tan-model/trading_bot/internal/strategy"
)

type Manager struct {
	cfg          *config.Config
	lastEntryAt  time.Time
	lastTargetTS string
}

func New(cfg *config.Config) *Manager {
	return &Manager{cfg: cfg}
}

func (m *Manager) AlreadyProcessed(targetTS string) bool {
	return m.lastTargetTS == targetTS
}

func (m *Manager) MarkProcessed(targetTS string) {
	m.lastTargetTS = targetTS
}

func (m *Manager) Approve(
	sig strategy.Signal,
	snap *predictions.Snapshot,
	pos *exchange.Position,
) (*exchange.OrderRequest, error) {
	if sig == strategy.SignalNone {
		return nil, nil
	}

	price := snap.LastClose()
	if price <= 0 {
		return nil, fmt.Errorf("missing last close price")
	}

	if sig == strategy.SignalClose {
		if pos == nil {
			return nil, nil
		}
		return &exchange.OrderRequest{
			Symbol: m.cfg.Market.Symbol,
			Side:   closeSide(pos.Side),
			Price:  price,
		}, nil
	}

	if pos != nil {
		return nil, fmt.Errorf("position already open")
	}
	if !m.lastEntryAt.IsZero() && time.Since(m.lastEntryAt) < time.Duration(m.cfg.Risk.CooldownMinutes)*time.Minute {
		return nil, fmt.Errorf("cooldown active")
	}

	sl, tp := m.priceLevels(sig, snap, price)
	side := exchange.SideBuy
	if sig == strategy.SignalOpenShort {
		side = exchange.SideSell
	}

	return &exchange.OrderRequest{
		Symbol:     m.cfg.Market.Symbol,
		Side:       side,
		QuoteUSDT:  m.cfg.Risk.QuoteSizeUSDT,
		Price:      price,
		StopLoss:   sl,
		TakeProfit: tp,
	}, nil
}

func (m *Manager) RecordEntry() {
	m.lastEntryAt = time.Now().UTC()
}

func (m *Manager) Restore(lastTargetTS string, lastEntryAt time.Time) {
	m.lastTargetTS = lastTargetTS
	m.lastEntryAt = lastEntryAt
}

func (m *Manager) Snapshot() (lastTargetTS string, lastEntryAt time.Time) {
	return m.lastTargetTS, m.lastEntryAt
}

func (m *Manager) priceLevels(sig strategy.Signal, snap *predictions.Snapshot, entry float64) (float64, float64) {
	slCfg := m.cfg.Risk.StopLoss
	tpCfg := m.cfg.Risk.TakeProfit

	var sl, tp float64
	if slCfg.Type == "quantile_band" {
		if sig == strategy.SignalOpenLong {
			sl = bandPrice(slCfg.LongBand, snap)
		} else {
			sl = bandPrice(slCfg.ShortBand, snap)
		}
	} else if slCfg.Type == "fixed_pct" && slCfg.Pct > 0 {
		if sig == strategy.SignalOpenLong {
			sl = entry * (1 - slCfg.Pct/100)
		} else {
			sl = entry * (1 + slCfg.Pct/100)
		}
	}

	if tpCfg.Type == "quantile_band" {
		if sig == strategy.SignalOpenLong {
			tp = bandPrice(tpCfg.LongBand, snap)
		} else {
			tp = bandPrice(tpCfg.ShortBand, snap)
		}
	} else if tpCfg.Type == "fixed_pct" && tpCfg.Pct > 0 {
		if sig == strategy.SignalOpenLong {
			tp = entry * (1 + tpCfg.Pct/100)
		} else {
			tp = entry * (1 - tpCfg.Pct/100)
		}
	}
	return sl, tp
}

func bandPrice(band string, snap *predictions.Snapshot) float64 {
	switch band {
	case "q10":
		return snap.Quantile.Q10
	case "q50":
		return snap.Quantile.Q50
	case "q90":
		return snap.Quantile.Q90
	default:
		return 0
	}
}

func closeSide(open exchange.Side) exchange.Side {
	if open == exchange.SideBuy {
		return exchange.SideSell
	}
	return exchange.SideBuy
}
