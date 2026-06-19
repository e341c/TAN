package exchange

import (
	"fmt"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
)

func NewExecutor(cfg *config.Config, apiKey, apiSecret string) (Executor, error) {
	switch cfg.Bot.Mode {
	case "paper":
		return NewPaperExecutor(), nil
	case "live":
		return NewLiveExecutor(cfg, apiKey, apiSecret)
	default:
		return nil, fmt.Errorf("unknown bot mode %q", cfg.Bot.Mode)
	}
}
