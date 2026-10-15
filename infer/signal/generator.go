package signal

import (
	"fmt"
	"infer/config"
	"math"
)

// Input -> Signal (гистерезис + sizing)
type Generator struct {
	cfg  config.Generator
	prev map[string]Side
}

func NewGenerator(cfg config.Generator) *Generator {
	prev := make(map[string]Side)

	return &Generator{
		cfg:  cfg,
		prev: prev,
	}
}

func (g *Generator) Next(in Input) Signal {
	side := g.decide(in.PUp, g.prev[in.Symbol])
	g.prev[in.Symbol] = side

	conf := math.Abs(in.PUp-0.5) * 2
	return Signal{
		ID:           fmt.Sprintf("%s-%d-%s", in.Symbol, in.BarTime.UnixMilli(), in.ModelVersion),
		Symbol:       in.Symbol,
		Side:         side,
		Confidence:   conf,
		TargetWeight: g.weight(side, conf, in.Sigma),
		Price:        in.Price,
		BarTime:      in.BarTime,
		ModelVersion: in.ModelVersion,
	}
}

func (g *Generator) decide(p float64, prev Side) Side {
	switch {
	case p >= g.cfg.EnterLong:
		return Long
	case p <= g.cfg.EnterShort:
		return Short
	case prev == Long && p > g.cfg.ExitLong:
		return Long // держим, пока не упали ниже порога выхода
	case prev == Short && p < g.cfg.ExitShort:
		return Short
	}
	return Flat
}

// Vol targeting: чем выше волатильность, тем меньше позиция.
func (g *Generator) weight(side Side, conf, sigma float64) float64 {
	if side == Flat || sigma <= 0 {
		return 0
	}
	w := math.Min(g.cfg.TargetVol/sigma*conf, g.cfg.MaxWeight)
	return float64(side) * w
}
