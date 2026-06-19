package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/eriktabys/tan-model/trading_bot/internal/bot"
	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/exchange"
	"github.com/eriktabys/tan-model/trading_bot/internal/orders"
	"github.com/eriktabys/tan-model/trading_bot/internal/predictions"
	"github.com/eriktabys/tan-model/trading_bot/internal/state"
)

func main() {
	cfgPath := os.Getenv("STRATEGY_CONFIG")
	if cfgPath == "" {
		cfgPath = "config/strategy.yaml"
	}

	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	apiKey := os.Getenv("BINANCE_API_KEY")
	apiSecret := os.Getenv("BINANCE_API_SECRET")
	if err := config.ValidateLiveCredentials(cfg.Bot.Mode, apiKey, apiSecret); err != nil {
		log.Fatalf("credentials: %v", err)
	}

	statePath := os.Getenv("STATE_PATH")
	if statePath == "" {
		statePath = "/var/lib/trading_bot/state.json"
	}
	stateStore := state.NewStore(statePath)

	var orderStore orders.Writer
	var orderStoreClose func(context.Context) error
	if databaseURL := os.Getenv("DATABASE_URL"); databaseURL != "" {
		store, err := orders.NewStore(context.Background(), databaseURL)
		if err != nil {
			log.Fatalf("orders store: %v", err)
		}
		orderStore = store
		orderStoreClose = store.Close
		log.Printf("order persistence enabled (DATABASE_URL)")
	} else {
		log.Printf("order persistence disabled: DATABASE_URL not set")
	}

	predictor, closeFn, err := predictions.NewClient(
		cfg.Predictions.Source,
		cfg.Predictions.APIBaseURL,
		cfg.Predictions.DatabaseURL,
		cfg.Market.Symbol,
		cfg.Market.Interval,
	)
	if err != nil {
		log.Fatalf("predictions client: %v", err)
	}

	executor, err := exchange.NewExecutor(cfg, apiKey, apiSecret)
	if err != nil {
		log.Fatalf("executor: %v", err)
	}

	st, err := stateStore.Load()
	if err != nil {
		log.Fatalf("state load: %v", err)
	}

	tradingBot := bot.New(cfg, predictor, executor, stateStore, orderStore)
	tradingBot.Restore(st)
	if st.Position != nil {
		log.Printf("restored position: %s %s qty=%.8f entry=%.2f",
			st.Position.Side, st.Position.Symbol, st.Position.Quantity, st.Position.EntryPrice)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := tradingBot.Start(ctx); err != nil {
		log.Fatalf("start: %v", err)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := tradingBot.Stop(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
	if closeFn != nil {
		_ = closeFn(shutdownCtx)
	}
	if orderStoreClose != nil {
		_ = orderStoreClose(shutdownCtx)
	}
}
