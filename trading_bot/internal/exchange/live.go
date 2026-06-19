package exchange

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	binance "github.com/adshao/go-binance/v2"
	futures "github.com/adshao/go-binance/v2/futures"

	"github.com/eriktabys/tan-model/trading_bot/internal/config"
)

type LiveExecutor struct {
	cfg        *config.Config
	apiKey     string
	apiSecret  string
	spotClient *binance.Client
	futClient  *futures.Client
	mode       string
	position   *Position
}

func NewLiveExecutor(cfg *config.Config, apiKey, apiSecret string) (*LiveExecutor, error) {
	e := &LiveExecutor{
		cfg:       cfg,
		apiKey:    apiKey,
		apiSecret: apiSecret,
		mode:      "live",
	}
	if cfg.Exchange.Testnet {
		binance.UseTestnet = true
		futures.UseTestnet = true
	}
	if cfg.Market.Type == "futures" {
		e.futClient = futures.NewClient(apiKey, apiSecret)
	} else {
		e.spotClient = binance.NewClient(apiKey, apiSecret)
	}
	return e, nil
}

func (l *LiveExecutor) Mode() string {
	return l.mode
}

func (l *LiveExecutor) GetPosition(symbol string) *Position {
	if l.position != nil && l.position.Symbol == symbol {
		cp := *l.position
		return &cp
	}
	return nil
}

func (l *LiveExecutor) RestorePosition(pos *Position) {
	if pos == nil {
		l.position = nil
		return
	}
	cp := *pos
	l.position = &cp
}

func (l *LiveExecutor) SyncPosition(symbol string) (*Position, error) {
	if l.position == nil || l.position.Symbol != symbol {
		return nil, nil
	}

	if l.cfg.Market.Type == "futures" {
		return l.syncFuturesPosition(symbol)
	}
	return l.syncSpotPosition(symbol)
}

func (l *LiveExecutor) syncSpotPosition(symbol string) (*Position, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	baseAsset := spotBaseAsset(symbol)
	acct, err := l.spotClient.NewGetAccountService().Do(ctx)
	if err != nil {
		return l.GetPosition(symbol), fmt.Errorf("spot account: %w", err)
	}

	var free float64
	for _, bal := range acct.Balances {
		if bal.Asset == baseAsset {
			free, _ = strconv.ParseFloat(bal.Free, 64)
			break
		}
	}

	threshold := l.position.Quantity * 0.05
	if free < threshold {
		_ = l.cancelProtectiveOrders(ctx, symbol)
		l.position = nil
		return nil, nil
	}
	return l.GetPosition(symbol), nil
}

func (l *LiveExecutor) syncFuturesPosition(symbol string) (*Position, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	positions, err := l.futClient.NewGetPositionRiskService().Symbol(symbol).Do(ctx)
	if err != nil {
		return l.GetPosition(symbol), fmt.Errorf("futures position risk: %w", err)
	}

	var amt float64
	for _, p := range positions {
		if p.Symbol != symbol {
			continue
		}
		amt, _ = strconv.ParseFloat(p.PositionAmt, 64)
		break
	}

	if l.position.Side == SideBuy && amt <= 0 {
		_ = l.cancelProtectiveOrdersFutures(ctx, symbol)
		l.position = nil
		return nil, nil
	}
	if l.position.Side == SideSell && amt >= 0 {
		_ = l.cancelProtectiveOrdersFutures(ctx, symbol)
		l.position = nil
		return nil, nil
	}
	return l.GetPosition(symbol), nil
}

func (l *LiveExecutor) Open(req OrderRequest) (*OrderResult, error) {
	if l.cfg.Market.Type == "futures" {
		return l.openFutures(req)
	}
	return l.openSpot(req)
}

func (l *LiveExecutor) Close(pos *Position, price float64) (*OrderResult, error) {
	if l.cfg.Market.Type == "futures" {
		return l.closeFutures(pos)
	}
	return l.closeSpot(pos)
}

func (l *LiveExecutor) PlaceProtectiveOrders(pos *Position) error {
	if pos == nil {
		return nil
	}
	if err := validProtectiveLevels(pos.Side, pos.EntryPrice, pos.StopLoss, pos.TakeProfit); err != nil {
		return err
	}
	if l.cfg.Market.Type == "futures" {
		return l.placeProtectiveFutures(pos)
	}
	return l.placeProtectiveSpot(pos)
}

func (l *LiveExecutor) placeProtectiveSpot(pos *Position) error {
	if l.spotClient == nil {
		return fmt.Errorf("spot client not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	closeSide := binance.SideTypeSell
	if pos.Side == SideSell {
		closeSide = binance.SideTypeBuy
	}
	qty := formatQty(pos.Quantity)

	if pos.StopLoss > 0 {
		order, err := l.spotClient.NewCreateOrderService().
			Symbol(pos.Symbol).
			Side(closeSide).
			Type(binance.OrderTypeStopLoss).
			Quantity(qty).
			StopPrice(formatPrice(pos.StopLoss)).
			Do(ctx)
		if err != nil {
			return fmt.Errorf("spot stop-loss order: %w", err)
		}
		pos.StopLossOrderID = strconv.FormatInt(order.OrderID, 10)
	}

	if pos.TakeProfit > 0 {
		order, err := l.spotClient.NewCreateOrderService().
			Symbol(pos.Symbol).
			Side(closeSide).
			Type(binance.OrderTypeTakeProfit).
			Quantity(qty).
			StopPrice(formatPrice(pos.TakeProfit)).
			Do(ctx)
		if err != nil {
			if pos.StopLossOrderID != "" {
				_, _ = l.spotClient.NewCancelOrderService().
					Symbol(pos.Symbol).
					OrderID(parseOrderID(pos.StopLossOrderID)).
					Do(ctx)
			}
			return fmt.Errorf("spot take-profit order: %w", err)
		}
		pos.TakeProfitOrderID = strconv.FormatInt(order.OrderID, 10)
	}

	if l.position != nil && l.position.Symbol == pos.Symbol {
		l.position.StopLossOrderID = pos.StopLossOrderID
		l.position.TakeProfitOrderID = pos.TakeProfitOrderID
	}
	return nil
}

func (l *LiveExecutor) placeProtectiveFutures(pos *Position) error {
	if l.futClient == nil {
		return fmt.Errorf("futures client not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	closeSide := futures.SideTypeSell
	posSide := futures.PositionSideTypeLong
	if pos.Side == SideSell {
		closeSide = futures.SideTypeBuy
		posSide = futures.PositionSideTypeShort
	}
	qty := formatQty(pos.Quantity)

	if pos.StopLoss > 0 {
		order, err := l.futClient.NewCreateOrderService().
			Symbol(pos.Symbol).
			Side(closeSide).
			PositionSide(posSide).
			Type(futures.OrderType("STOP_MARKET")).
			StopPrice(formatPrice(pos.StopLoss)).
			Quantity(qty).
			ReduceOnly(true).
			Do(ctx)
		if err != nil {
			return fmt.Errorf("futures stop-loss order: %w", err)
		}
		pos.StopLossOrderID = strconv.FormatInt(order.OrderID, 10)
	}

	if pos.TakeProfit > 0 {
		order, err := l.futClient.NewCreateOrderService().
			Symbol(pos.Symbol).
			Side(closeSide).
			PositionSide(posSide).
			Type(futures.OrderType("TAKE_PROFIT_MARKET")).
			StopPrice(formatPrice(pos.TakeProfit)).
			Quantity(qty).
			ReduceOnly(true).
			Do(ctx)
		if err != nil {
			if pos.StopLossOrderID != "" {
				_, _ = l.futClient.NewCancelOrderService().
					Symbol(pos.Symbol).
					OrderID(parseOrderID(pos.StopLossOrderID)).
					Do(ctx)
			}
			return fmt.Errorf("futures take-profit order: %w", err)
		}
		pos.TakeProfitOrderID = strconv.FormatInt(order.OrderID, 10)
	}

	if l.position != nil && l.position.Symbol == pos.Symbol {
		l.position.StopLossOrderID = pos.StopLossOrderID
		l.position.TakeProfitOrderID = pos.TakeProfitOrderID
	}
	return nil
}

func (l *LiveExecutor) openSpot(req OrderRequest) (*OrderResult, error) {
	if l.spotClient == nil {
		return nil, fmt.Errorf("spot client not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	side := binance.SideTypeBuy
	if req.Side == SideSell {
		side = binance.SideTypeSell
	}

	order, err := l.spotClient.NewCreateOrderService().
		Symbol(req.Symbol).
		Side(side).
		Type(binance.OrderTypeMarket).
		QuoteOrderQty(fmt.Sprintf("%.2f", req.QuoteUSDT)).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("spot open order: %w", err)
	}

	qty, price := parseFill(order.ExecutedQuantity, order.CummulativeQuoteQuantity)
	l.position = &Position{
		Side:       req.Side,
		Symbol:     req.Symbol,
		Quantity:   qty,
		EntryPrice: price,
		OpenedAt:   time.Now().UTC(),
		StopLoss:   req.StopLoss,
		TakeProfit: req.TakeProfit,
	}
	return &OrderResult{
		Symbol:     req.Symbol,
		Side:       req.Side,
		Quantity:   qty,
		Price:      price,
		OrderID:    strconv.FormatInt(order.OrderID, 10),
		StopLoss:   req.StopLoss,
		TakeProfit: req.TakeProfit,
		Mode:       l.mode,
	}, nil
}

func (l *LiveExecutor) closeSpot(pos *Position) (*OrderResult, error) {
	if l.spotClient == nil {
		return nil, fmt.Errorf("spot client not initialized")
	}
	if pos == nil {
		return nil, fmt.Errorf("no position to close")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := l.cancelProtectiveOrders(ctx, pos.Symbol); err != nil {
		return nil, err
	}

	side := binance.SideTypeSell
	if pos.Side == SideSell {
		side = binance.SideTypeBuy
	}

	order, err := l.spotClient.NewCreateOrderService().
		Symbol(pos.Symbol).
		Side(side).
		Type(binance.OrderTypeMarket).
		Quantity(formatQty(pos.Quantity)).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("spot close order: %w", err)
	}

	qty, price := parseFill(order.ExecutedQuantity, order.CummulativeQuoteQuantity)
	l.position = nil
	closeSide := SideSell
	if pos.Side == SideSell {
		closeSide = SideBuy
	}
	return &OrderResult{
		Symbol:   pos.Symbol,
		Side:     closeSide,
		Quantity: qty,
		Price:    price,
		OrderID:  strconv.FormatInt(order.OrderID, 10),
		Mode:     l.mode,
	}, nil
}

func (l *LiveExecutor) openFutures(req OrderRequest) (*OrderResult, error) {
	if l.futClient == nil {
		return nil, fmt.Errorf("futures client not initialized")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if _, err := l.futClient.NewChangeLeverageService().
		Symbol(req.Symbol).
		Leverage(l.cfg.Market.Leverage).
		Do(ctx); err != nil {
		return nil, fmt.Errorf("set leverage: %w", err)
	}

	side := futures.SideTypeBuy
	posSide := futures.PositionSideTypeLong
	if req.Side == SideSell {
		side = futures.SideTypeSell
		posSide = futures.PositionSideTypeShort
	}

	qty := req.QuoteUSDT / req.Price
	order, err := l.futClient.NewCreateOrderService().
		Symbol(req.Symbol).
		Side(side).
		PositionSide(posSide).
		Type(futures.OrderTypeMarket).
		Quantity(formatQty(qty)).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("futures open order: %w", err)
	}

	fillQty, fillPrice := parseFill(order.ExecutedQuantity, order.CumQuote)
	l.position = &Position{
		Side:       req.Side,
		Symbol:     req.Symbol,
		Quantity:   fillQty,
		EntryPrice: fillPrice,
		OpenedAt:   time.Now().UTC(),
		StopLoss:   req.StopLoss,
		TakeProfit: req.TakeProfit,
	}
	return &OrderResult{
		Symbol:     req.Symbol,
		Side:       req.Side,
		Quantity:   fillQty,
		Price:      fillPrice,
		OrderID:    strconv.FormatInt(order.OrderID, 10),
		StopLoss:   req.StopLoss,
		TakeProfit: req.TakeProfit,
		Mode:       l.mode,
	}, nil
}

func (l *LiveExecutor) closeFutures(pos *Position) (*OrderResult, error) {
	if l.futClient == nil {
		return nil, fmt.Errorf("futures client not initialized")
	}
	if pos == nil {
		return nil, fmt.Errorf("no position to close")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := l.cancelProtectiveOrdersFutures(ctx, pos.Symbol); err != nil {
		return nil, err
	}

	side := futures.SideTypeSell
	posSide := futures.PositionSideTypeLong
	if pos.Side == SideSell {
		side = futures.SideTypeBuy
		posSide = futures.PositionSideTypeShort
	}

	order, err := l.futClient.NewCreateOrderService().
		Symbol(pos.Symbol).
		Side(side).
		PositionSide(posSide).
		Type(futures.OrderTypeMarket).
		Quantity(formatQty(pos.Quantity)).
		Do(ctx)
	if err != nil {
		return nil, fmt.Errorf("futures close order: %w", err)
	}

	fillQty, fillPrice := parseFill(order.ExecutedQuantity, order.CumQuote)
	l.position = nil
	closeSide := SideSell
	if pos.Side == SideSell {
		closeSide = SideBuy
	}
	return &OrderResult{
		Symbol:   pos.Symbol,
		Side:     closeSide,
		Quantity: fillQty,
		Price:    fillPrice,
		OrderID:  strconv.FormatInt(order.OrderID, 10),
		Mode:     l.mode,
	}, nil
}

func (l *LiveExecutor) cancelProtectiveOrders(ctx context.Context, symbol string) error {
	if l.position == nil || l.spotClient == nil {
		return nil
	}
	if l.position.StopLossOrderID != "" {
		_, err := l.spotClient.NewCancelOrderService().
			Symbol(symbol).
			OrderID(parseOrderID(l.position.StopLossOrderID)).
			Do(ctx)
		if err != nil && !isUnknownOrder(err) {
			return fmt.Errorf("cancel stop-loss: %w", err)
		}
	}
	if l.position.TakeProfitOrderID != "" {
		_, err := l.spotClient.NewCancelOrderService().
			Symbol(symbol).
			OrderID(parseOrderID(l.position.TakeProfitOrderID)).
			Do(ctx)
		if err != nil && !isUnknownOrder(err) {
			return fmt.Errorf("cancel take-profit: %w", err)
		}
	}
	return nil
}

func (l *LiveExecutor) cancelProtectiveOrdersFutures(ctx context.Context, symbol string) error {
	if l.position == nil || l.futClient == nil {
		return nil
	}
	if l.position.StopLossOrderID != "" {
		_, err := l.futClient.NewCancelOrderService().
			Symbol(symbol).
			OrderID(parseOrderID(l.position.StopLossOrderID)).
			Do(ctx)
		if err != nil && !isUnknownOrder(err) {
			return fmt.Errorf("cancel stop-loss: %w", err)
		}
	}
	if l.position.TakeProfitOrderID != "" {
		_, err := l.futClient.NewCancelOrderService().
			Symbol(symbol).
			OrderID(parseOrderID(l.position.TakeProfitOrderID)).
			Do(ctx)
		if err != nil && !isUnknownOrder(err) {
			return fmt.Errorf("cancel take-profit: %w", err)
		}
	}
	return nil
}

func parseFill(qtyStr, quoteStr string) (float64, float64) {
	qty, _ := strconv.ParseFloat(qtyStr, 64)
	quote, _ := strconv.ParseFloat(quoteStr, 64)
	price := 0.0
	if qty > 0 {
		price = quote / qty
	}
	return qty, price
}

func formatQty(q float64) string {
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.8f", q), "0"), ".")
}

func parseOrderID(id string) int64 {
	n, _ := strconv.ParseInt(id, 10, 64)
	return n
}

func spotBaseAsset(symbol string) string {
	if strings.HasSuffix(symbol, "USDT") {
		return strings.TrimSuffix(symbol, "USDT")
	}
	return symbol
}

func isUnknownOrder(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unknown order") || strings.Contains(msg, "-2011")
}
