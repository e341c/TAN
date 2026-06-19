package config

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Bot         BotConfig         `yaml:"bot"`
	Market      MarketConfig      `yaml:"market"`
	Predictions PredictionsConfig `yaml:"predictions"`
	Strategy    StrategyConfig    `yaml:"strategy"`
	Risk        RiskConfig        `yaml:"risk"`
	Exchange    ExchangeConfig    `yaml:"exchange"`
}

type BotConfig struct {
	Mode     string `yaml:"mode"`
	LogLevel string `yaml:"log_level"`
}

type MarketConfig struct {
	Type         string `yaml:"type"`
	Symbol       string `yaml:"symbol"`
	Interval     string `yaml:"interval"`
	Margin       bool   `yaml:"margin"`
	Leverage     int    `yaml:"leverage"`
	PositionSide string `yaml:"position_side"`
}

type PredictionsConfig struct {
	Source      string `yaml:"source"`
	APIBaseURL  string `yaml:"api_base_url"`
	DatabaseURL string `yaml:"database_url"`
	Schedule    string `yaml:"schedule"`
}

type StrategyConfig struct {
	Name  string      `yaml:"name"`
	Entry EntryRules  `yaml:"entry"`
	Exit  ExitRules   `yaml:"exit"`
}

type EntryRules struct {
	Long  RuleGroup `yaml:"long"`
	Short RuleGroup `yaml:"short"`
}

type ExitRules struct {
	Long  RuleGroup `yaml:"long"`
	Short RuleGroup `yaml:"short"`
}

type RuleGroup struct {
	All []Condition `yaml:"all"`
	Any []Condition `yaml:"any"`
}

type Condition struct {
	Field string      `yaml:"field"`
	Op    string      `yaml:"op"`
	Value interface{}   `yaml:"value"`
}

type RiskConfig struct {
	QuoteSizeUSDT    float64          `yaml:"quote_size_usdt"`
	MaxOpenPositions int              `yaml:"max_open_positions"`
	CooldownMinutes  int              `yaml:"cooldown_minutes"`
	StopLoss         PriceLevelConfig `yaml:"stop_loss"`
	TakeProfit       PriceLevelConfig `yaml:"take_profit"`
}

type PriceLevelConfig struct {
	Type      string  `yaml:"type"`
	LongBand  string  `yaml:"long_band"`
	ShortBand string  `yaml:"short_band"`
	Pct       float64 `yaml:"pct"`
}

type ExchangeConfig struct {
	Testnet      bool `yaml:"testnet"`
	RecvWindowMS int  `yaml:"recv_window_ms"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *Config) Validate() error {
	mode := strings.ToLower(strings.TrimSpace(c.Bot.Mode))
	if mode != "paper" && mode != "live" {
		return fmt.Errorf("bot.mode must be paper or live, got %q", c.Bot.Mode)
	}
	c.Bot.Mode = mode

	marketType := strings.ToLower(strings.TrimSpace(c.Market.Type))
	if marketType != "spot" && marketType != "futures" {
		return fmt.Errorf("market.type must be spot or futures, got %q", c.Market.Type)
	}
	c.Market.Type = marketType

	if c.Market.Symbol == "" {
		return fmt.Errorf("market.symbol is required")
	}
	if c.Market.Interval == "" {
		c.Market.Interval = "1h"
	}
	if c.Market.Leverage <= 0 {
		c.Market.Leverage = 1
	}

	posSide := strings.ToLower(strings.TrimSpace(c.Market.PositionSide))
	if posSide == "" {
		posSide = "both"
	}
	if posSide != "long" && posSide != "short" && posSide != "both" {
		return fmt.Errorf("market.position_side must be long, short, or both")
	}
	c.Market.PositionSide = posSide

	source := strings.ToLower(strings.TrimSpace(c.Predictions.Source))
	if source != "api" && source != "database" {
		return fmt.Errorf("predictions.source must be api or database")
	}
	c.Predictions.Source = source

	if c.Predictions.Schedule == "" {
		c.Predictions.Schedule = "5 * * * *"
	}
	if c.Predictions.Source == "api" && strings.TrimSpace(c.Predictions.APIBaseURL) == "" {
		return fmt.Errorf("predictions.api_base_url is required when source=api")
	}
	if c.Predictions.Source == "database" && strings.TrimSpace(c.Predictions.DatabaseURL) == "" {
		return fmt.Errorf("predictions.database_url is required when source=database")
	}

	if c.Risk.QuoteSizeUSDT <= 0 {
		return fmt.Errorf("risk.quote_size_usdt must be positive")
	}
	if c.Risk.MaxOpenPositions <= 0 {
		c.Risk.MaxOpenPositions = 1
	}
	if c.Risk.CooldownMinutes < 0 {
		c.Risk.CooldownMinutes = 0
	}

	if c.Exchange.RecvWindowMS <= 0 {
		c.Exchange.RecvWindowMS = 5000
	}

	return nil
}

func ValidateLiveCredentials(mode, apiKey, apiSecret string) error {
	if mode != "live" {
		return nil
	}
	if strings.TrimSpace(apiKey) == "" || strings.TrimSpace(apiSecret) == "" {
		return fmt.Errorf("live mode requires BINANCE_API_KEY and BINANCE_API_SECRET")
	}
	return nil
}
