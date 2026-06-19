package bot

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/robfig/cron/v3"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
	"github.com/eriktabys/tan-model/trading_bot/internal/exchange"
	"github.com/eriktabys/tan-model/trading_bot/internal/orders"
	"github.com/eriktabys/tan-model/trading_bot/internal/predictions"
	"github.com/eriktabys/tan-model/trading_bot/internal/risk"
	"github.com/eriktabys/tan-model/trading_bot/internal/state"
	"github.com/eriktabys/tan-model/trading_bot/internal/strategy"
)

type Bot struct {
	cfg        *config.Config
	predictor  predictions.Client
	executor   exchange.Executor
	strategy   *strategy.Engine
	risk       *risk.Manager
	stateStore *state.Store
	orderStore orders.Writer
	cron       *cron.Cron
	healthSrv  *http.Server
	mu         sync.Mutex
	lastRun    time.Time
	lastError  string
	tradeID    string
}

type RunLog struct {
	Time     string  `json:"time"`
	Mode     string  `json:"mode"`
	Strategy string  `json:"strategy"`
	Signal   string  `json:"signal"`
	Action   string  `json:"action"`
	Price    float64 `json:"price,omitempty"`
	Reason   string  `json:"reason"`
	TargetTS string  `json:"target_ts,omitempty"`
	OrderID  string  `json:"order_id,omitempty"`
	Error    string  `json:"error,omitempty"`
}

func New(
	cfg *config.Config,
	predictor predictions.Client,
	executor exchange.Executor,
	stateStore *state.Store,
	orderStore orders.Writer,
) *Bot {
	return &Bot{
		cfg:        cfg,
		predictor:  predictor,
		executor:   executor,
		strategy:   strategy.New(cfg),
		risk:       risk.New(cfg),
		stateStore: stateStore,
		orderStore: orderStore,
	}
}

func (b *Bot) Restore(st *state.BotState) {
	if st == nil {
		return
	}
	b.risk.Restore(st.LastTargetTS, st.LastEntryAt)
	b.tradeID = st.TradeID
	if st.Position != nil {
		b.executor.RestorePosition(st.Position)
	}
}

func (b *Bot) Start(ctx context.Context) error {
	b.cron = cron.New(cron.WithLocation(time.UTC))
	_, err := b.cron.AddFunc(b.cfg.Predictions.Schedule, func() {
		if err := b.RunOnce(ctx); err != nil {
			b.mu.Lock()
			b.lastError = err.Error()
			b.mu.Unlock()
			log.Printf("run error: %v", err)
		}
	})
	if err != nil {
		return fmt.Errorf("schedule cron: %w", err)
	}
	b.cron.Start()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", b.healthHandler)
	b.healthSrv = &http.Server{
		Addr:    ":8081",
		Handler: mux,
	}
	go func() {
		if err := b.healthSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("health server: %v", err)
		}
	}()

	log.Printf("%s bot started strategy=%s market=%s/%s schedule=%q",
		strings.ToUpper(b.cfg.Bot.Mode),
		b.cfg.Strategy.Name,
		b.cfg.Market.Type,
		b.cfg.Market.Symbol,
		b.cfg.Predictions.Schedule,
	)
	return nil
}

func (b *Bot) Stop(ctx context.Context) error {
	if b.cron != nil {
		b.cron.Stop()
	}
	if b.healthSrv != nil {
		return b.healthSrv.Shutdown(ctx)
	}
	return nil
}

func (b *Bot) healthHandler(w http.ResponseWriter, r *http.Request) {
	b.mu.Lock()
	defer b.mu.Unlock()
	status := "ok"
	if b.lastError != "" {
		status = "degraded"
	}
	resp := map[string]interface{}{
		"status":     status,
		"mode":       b.cfg.Bot.Mode,
		"strategy":   b.cfg.Strategy.Name,
		"last_run":   b.lastRun.UTC().Format(time.RFC3339),
		"last_error": b.lastError,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (b *Bot) RunOnce(ctx context.Context) error {
	defer b.saveState()

	snap, err := b.predictor.Fetch(ctx)
	if err != nil {
		return err
	}

	targetTS := snap.Quantile.NextTime
	price := snap.LastClose()
	targetTSPtr := parseTargetTS(targetTS)

	posBeforeSync := b.executor.GetPosition(b.cfg.Market.Symbol)
	hadPosition := posBeforeSync != nil
	syncedPos, err := b.executor.SyncPosition(b.cfg.Market.Symbol)
	if err != nil {
		log.Printf("sync position warning: %v", err)
		syncedPos = b.executor.GetPosition(b.cfg.Market.Symbol)
	}
	if hadPosition && syncedPos == nil {
		b.persistExchangeProtectiveFill(ctx, posBeforeSync, price, targetTS, targetTSPtr)
		b.emitLog(RunLog{
			Time:     time.Now().UTC().Format(time.RFC3339),
			Mode:     b.executor.Mode(),
			Strategy: b.cfg.Strategy.Name,
			Signal:   string(strategy.SignalClose),
			Action:   "sl_tp_filled",
			Price:    price,
			Reason:   "position closed by exchange protective order",
			TargetTS: targetTS,
		})
		b.tradeID = ""
		b.mu.Lock()
		b.lastRun = time.Now().UTC()
		b.lastError = ""
		b.mu.Unlock()
		return nil
	}

	if b.risk.AlreadyProcessed(targetTS) {
		b.emitLog(RunLog{
			Time:     time.Now().UTC().Format(time.RFC3339),
			Mode:     b.executor.Mode(),
			Strategy: b.cfg.Strategy.Name,
			Signal:   string(strategy.SignalNone),
			Action:   "skip",
			Reason:   "target_ts already processed",
			TargetTS: targetTS,
		})
		return nil
	}

	pos := syncedPos
	var currentPos strategy.PositionSide
	if pos != nil {
		if pos.Side == exchange.SideBuy {
			currentPos = strategy.PositionLong
		} else {
			currentPos = strategy.PositionShort
		}
	}

	sig, reason := b.strategy.Evaluate(snap, currentPos)

	if sig == strategy.SignalNone && pos != nil {
		if checker, ok := b.executor.(exchange.ProtectiveChecker); ok {
			if triggered, tpReason := checker.CheckProtectiveExit(price); triggered {
				sig = strategy.SignalClose
				reason = tpReason
			}
		}
	}

	req, err := b.risk.Approve(sig, snap, pos)
	if err != nil {
		b.emitLog(RunLog{
			Time:     time.Now().UTC().Format(time.RFC3339),
			Mode:     b.executor.Mode(),
			Strategy: b.cfg.Strategy.Name,
			Signal:   string(sig),
			Action:   "reject",
			Reason:   err.Error(),
			TargetTS: targetTS,
		})
		b.risk.MarkProcessed(targetTS)
		b.mu.Lock()
		b.lastRun = time.Now().UTC()
		b.lastError = ""
		b.mu.Unlock()
		return nil
	}

	action := "hold"
	var result *exchange.OrderResult

	if req != nil {
		switch sig {
		case strategy.SignalClose:
			action = "close"
			result, err = b.executor.Close(pos, price)
			if err != nil {
				b.persistFailedOrder(ctx, sig, action, req, pos, price, targetTSPtr, err)
			} else if result != nil {
				b.persistFilledOrder(ctx, sig, action, result, targetTSPtr, "close", "market", b.tradeID)
				b.tradeID = ""
			}
		case strategy.SignalOpenLong, strategy.SignalOpenShort:
			action = "open"
			result, err = b.executor.Open(*req)
			if err != nil {
				b.persistFailedOrder(ctx, sig, action, req, nil, price, targetTSPtr, err)
			} else if result != nil {
				b.tradeID = uuid.New().String()
				b.persistFilledOrder(ctx, sig, action, result, targetTSPtr, "open", "market", b.tradeID)
				b.risk.RecordEntry()
				if b.cfg.Bot.Mode == "live" {
					if live, ok := b.executor.(*exchange.LiveExecutor); ok {
						openPos := b.executor.GetPosition(b.cfg.Market.Symbol)
						if openPos != nil && (openPos.StopLoss > 0 || openPos.TakeProfit > 0) {
							if protectErr := live.PlaceProtectiveOrders(openPos); protectErr != nil {
								log.Printf("protective orders warning: %v", protectErr)
								b.persistProtectiveFailure(ctx, sig, openPos, targetTSPtr, protectErr)
								b.emitLog(RunLog{
									Time:     time.Now().UTC().Format(time.RFC3339),
									Mode:     b.executor.Mode(),
									Strategy: b.cfg.Strategy.Name,
									Signal:   string(sig),
									Action:   "protective_error",
									Price:    price,
									Reason:   protectErr.Error(),
									TargetTS: targetTS,
									OrderID:  result.OrderID,
								})
							} else {
								b.persistProtectiveOrders(ctx, sig, openPos, targetTSPtr)
							}
						}
					}
				}
			}
		}
	}

	runLog := RunLog{
		Time:     time.Now().UTC().Format(time.RFC3339),
		Mode:     b.executor.Mode(),
		Strategy: b.cfg.Strategy.Name,
		Signal:   string(sig),
		Action:   action,
		Price:    price,
		Reason:   reason,
		TargetTS: targetTS,
	}
	if err != nil {
		runLog.Error = err.Error()
		runLog.Action = "error"
	}
	if result != nil {
		runLog.OrderID = result.OrderID
		runLog.Price = result.Price
	}
	b.emitLog(runLog)

	b.risk.MarkProcessed(targetTS)
	b.mu.Lock()
	b.lastRun = time.Now().UTC()
	if err != nil {
		b.lastError = err.Error()
	} else {
		b.lastError = ""
	}
	b.mu.Unlock()

	return err
}

func (b *Bot) persistExchangeProtectiveFill(ctx context.Context, pos *exchange.Position, price float64, targetTS string, targetTSPtr *time.Time) {
	if b.orderStore == nil || pos == nil {
		return
	}
	orderID := fmt.Sprintf("exchange-close-%s", targetTS)
	orderKind := "close"
	orderType := "market"
	side := exchange.SideSell
	if pos.Side == exchange.SideSell {
		side = exchange.SideBuy
	}
	if pos.StopLossOrderID != "" {
		orderID = pos.StopLossOrderID
		orderKind = "stop_loss"
		orderType = "stop_loss"
	} else if pos.TakeProfitOrderID != "" {
		orderID = pos.TakeProfitOrderID
		orderKind = "take_profit"
		orderType = "take_profit"
	}
	qty := orders.FloatPtr(pos.Quantity)
	px := orders.FloatPtr(price)
	b.persistOrder(ctx, orders.Record{
		ExchangeOrderID: orderID,
		Mode:            b.executor.Mode(),
		MarketType:      b.cfg.Market.Type,
		Symbol:          pos.Symbol,
		Side:            string(side),
		OrderKind:       orderKind,
		OrderType:       orderType,
		Status:          orders.StatusFilled,
		Quantity:        qty,
		Price:           px,
		Strategy:        b.cfg.Strategy.Name,
		Signal:          string(strategy.SignalClose),
		Action:          "sl_tp_filled",
		Reason:          "position closed by exchange protective order",
		TargetTS:        targetTSPtr,
		TradeID:         b.tradeID,
	})
}

func (b *Bot) persistFilledOrder(
	ctx context.Context,
	sig strategy.Signal,
	action string,
	result *exchange.OrderResult,
	targetTS *time.Time,
	orderKind, orderType, tradeID string,
) {
	if b.orderStore == nil || result == nil {
		return
	}
	qty := orders.FloatPtr(result.Quantity)
	px := orders.FloatPtr(result.Price)
	var sl, tp *float64
	if result.StopLoss > 0 {
		sl = orders.FloatPtr(result.StopLoss)
	}
	if result.TakeProfit > 0 {
		tp = orders.FloatPtr(result.TakeProfit)
	}
	b.persistOrder(ctx, orders.Record{
		ExchangeOrderID: result.OrderID,
		Mode:            result.Mode,
		MarketType:      b.cfg.Market.Type,
		Symbol:          result.Symbol,
		Side:            string(result.Side),
		OrderKind:       orderKind,
		OrderType:       orderType,
		Status:          orders.StatusFilled,
		Quantity:        qty,
		Price:           px,
		StopLoss:        sl,
		TakeProfit:      tp,
		Strategy:        b.cfg.Strategy.Name,
		Signal:          string(sig),
		Action:          action,
		TargetTS:        targetTS,
		TradeID:         tradeID,
	})
}

func (b *Bot) persistFailedOrder(
	ctx context.Context,
	sig strategy.Signal,
	action string,
	req *exchange.OrderRequest,
	pos *exchange.Position,
	price float64,
	targetTS *time.Time,
	orderErr error,
) {
	if b.orderStore == nil || req == nil {
		return
	}
	side := string(req.Side)
	symbol := req.Symbol
	var sl, tp *float64
	if req.StopLoss > 0 {
		sl = orders.FloatPtr(req.StopLoss)
	}
	if req.TakeProfit > 0 {
		tp = orders.FloatPtr(req.TakeProfit)
	}
	quote := orders.FloatPtr(req.QuoteUSDT)
	px := orders.FloatPtr(price)
	if action == "close" && pos != nil {
		side = string(pos.Side)
		if pos.Side == exchange.SideBuy {
			side = string(exchange.SideSell)
		} else {
			side = string(exchange.SideBuy)
		}
		qty := orders.FloatPtr(pos.Quantity)
		b.persistOrder(ctx, orders.Record{
			ExchangeOrderID: orders.FailedOrderID(),
			Mode:            b.executor.Mode(),
			MarketType:      b.cfg.Market.Type,
			Symbol:          pos.Symbol,
			Side:            side,
			OrderKind:       "close",
			OrderType:       "market",
			Status:          orders.StatusFailed,
			Quantity:        qty,
			Price:           px,
			Strategy:        b.cfg.Strategy.Name,
			Signal:          string(sig),
			Action:          action,
			TargetTS:        targetTS,
			ErrorMessage:    orderErr.Error(),
			TradeID:         b.tradeID,
		})
		return
	}
	b.persistOrder(ctx, orders.Record{
		ExchangeOrderID: orders.FailedOrderID(),
		Mode:            b.executor.Mode(),
		MarketType:      b.cfg.Market.Type,
		Symbol:          symbol,
		Side:            side,
		OrderKind:       "open",
		OrderType:       "market",
		Status:          orders.StatusFailed,
		Price:           px,
		QuoteUSDT:       quote,
		StopLoss:        sl,
		TakeProfit:      tp,
		Strategy:        b.cfg.Strategy.Name,
		Signal:          string(sig),
		Action:          action,
		TargetTS:        targetTS,
		ErrorMessage:    orderErr.Error(),
		TradeID:         b.tradeID,
	})
}

func (b *Bot) persistProtectiveOrders(ctx context.Context, sig strategy.Signal, pos *exchange.Position, targetTS *time.Time) {
	if b.orderStore == nil || pos == nil {
		return
	}
	closeSide := exchange.SideSell
	if pos.Side == exchange.SideSell {
		closeSide = exchange.SideBuy
	}
	qty := orders.FloatPtr(pos.Quantity)
	if pos.StopLossOrderID != "" {
		sl := orders.FloatPtr(pos.StopLoss)
		b.persistOrder(ctx, orders.Record{
			ExchangeOrderID: pos.StopLossOrderID,
			Mode:            b.executor.Mode(),
			MarketType:      b.cfg.Market.Type,
			Symbol:          pos.Symbol,
			Side:            string(closeSide),
			OrderKind:       "stop_loss",
			OrderType:       "stop_loss",
			Status:          orders.StatusFilled,
			Quantity:        qty,
			Price:           sl,
			StopLoss:        sl,
			Strategy:        b.cfg.Strategy.Name,
			Signal:          string(sig),
			Action:          "protective",
			TargetTS:        targetTS,
			TradeID:         b.tradeID,
		})
	}
	if pos.TakeProfitOrderID != "" {
		tp := orders.FloatPtr(pos.TakeProfit)
		b.persistOrder(ctx, orders.Record{
			ExchangeOrderID: pos.TakeProfitOrderID,
			Mode:            b.executor.Mode(),
			MarketType:      b.cfg.Market.Type,
			Symbol:          pos.Symbol,
			Side:            string(closeSide),
			OrderKind:       "take_profit",
			OrderType:       "take_profit",
			Status:          orders.StatusFilled,
			Quantity:        qty,
			Price:           tp,
			TakeProfit:      tp,
			Strategy:        b.cfg.Strategy.Name,
			Signal:          string(sig),
			Action:          "protective",
			TargetTS:        targetTS,
			TradeID:         b.tradeID,
		})
	}
}

func (b *Bot) persistProtectiveFailure(ctx context.Context, sig strategy.Signal, pos *exchange.Position, targetTS *time.Time, protectErr error) {
	if b.orderStore == nil || pos == nil {
		return
	}
	closeSide := exchange.SideSell
	if pos.Side == exchange.SideSell {
		closeSide = exchange.SideBuy
	}
	b.persistOrder(ctx, orders.Record{
		ExchangeOrderID: orders.FailedOrderID(),
		Mode:            b.executor.Mode(),
		MarketType:      b.cfg.Market.Type,
		Symbol:          pos.Symbol,
		Side:            string(closeSide),
		OrderKind:       "stop_loss",
		OrderType:       "stop_loss",
		Status:          orders.StatusFailed,
		Strategy:        b.cfg.Strategy.Name,
		Signal:          string(sig),
		Action:          "protective_error",
		TargetTS:        targetTS,
		ErrorMessage:    protectErr.Error(),
		TradeID:         b.tradeID,
	})
}

func (b *Bot) persistOrder(ctx context.Context, r orders.Record) {
	if err := b.orderStore.Insert(ctx, r); err != nil {
		log.Printf("order persist warning: %v", err)
	}
}

func parseTargetTS(targetTS string) *time.Time {
	if targetTS == "" {
		return nil
	}
	t, err := time.Parse(time.RFC3339, targetTS)
	if err != nil {
		return nil
	}
	utc := t.UTC()
	return &utc
}

func (b *Bot) saveState() {
	if b.stateStore == nil {
		return
	}
	lastTS, lastEntry := b.risk.Snapshot()
	st := &state.BotState{
		LastTargetTS: lastTS,
		LastEntryAt:  lastEntry,
		TradeID:      b.tradeID,
		Position:     b.executor.GetPosition(b.cfg.Market.Symbol),
	}
	if err := b.stateStore.Save(st); err != nil {
		log.Printf("state save warning: %v", err)
	}
}

func (b *Bot) emitLog(entry RunLog) {
	data, _ := json.Marshal(entry)
	prefix := strings.ToUpper(b.cfg.Bot.Mode)
	log.Printf("%s %s", prefix, string(data))
	if path := os.Getenv("TRADE_LOG_PATH"); path != "" {
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err == nil {
			defer f.Close()
			_, _ = f.Write(append(data, '\n'))
		}
	}
}
