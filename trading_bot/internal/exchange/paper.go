package exchange

import (
	"fmt"
	"sync"
	"time"
)

type PaperExecutor struct {
	mu       sync.Mutex
	position *Position
	mode     string
}

func NewPaperExecutor() *PaperExecutor {
	return &PaperExecutor{mode: "paper"}
}

func (p *PaperExecutor) Mode() string {
	return p.mode
}

func (p *PaperExecutor) GetPosition(symbol string) *Position {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.position != nil && p.position.Symbol == symbol {
		cp := *p.position
		return &cp
	}
	return nil
}

func (p *PaperExecutor) SyncPosition(symbol string) (*Position, error) {
	return p.GetPosition(symbol), nil
}

func (p *PaperExecutor) RestorePosition(pos *Position) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if pos == nil {
		p.position = nil
		return
	}
	cp := *pos
	p.position = &cp
}

func (p *PaperExecutor) CheckProtectiveExit(price float64) (bool, string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.position == nil || price <= 0 {
		return false, ""
	}
	pos := p.position
	if pos.Side == SideBuy {
		if pos.StopLoss > 0 && price <= pos.StopLoss {
			return true, fmt.Sprintf("paper stop-loss hit: price %.2f <= %.2f", price, pos.StopLoss)
		}
		if pos.TakeProfit > 0 && price >= pos.TakeProfit {
			return true, fmt.Sprintf("paper take-profit hit: price %.2f >= %.2f", price, pos.TakeProfit)
		}
	} else {
		if pos.StopLoss > 0 && price >= pos.StopLoss {
			return true, fmt.Sprintf("paper stop-loss hit: price %.2f >= %.2f", price, pos.StopLoss)
		}
		if pos.TakeProfit > 0 && price <= pos.TakeProfit {
			return true, fmt.Sprintf("paper take-profit hit: price %.2f <= %.2f", price, pos.TakeProfit)
		}
	}
	return false, ""
}

func (p *PaperExecutor) Open(req OrderRequest) (*OrderResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.position != nil {
		return nil, fmt.Errorf("position already open for %s", req.Symbol)
	}
	if req.Price <= 0 {
		return nil, fmt.Errorf("invalid price")
	}
	qty := req.QuoteUSDT / req.Price
	p.position = &Position{
		Side:       req.Side,
		Symbol:     req.Symbol,
		Quantity:   qty,
		EntryPrice: req.Price,
		OpenedAt:   time.Now().UTC(),
		StopLoss:   req.StopLoss,
		TakeProfit: req.TakeProfit,
	}
	return &OrderResult{
		Symbol:     req.Symbol,
		Side:       req.Side,
		Quantity:   qty,
		Price:      req.Price,
		OrderID:    fmt.Sprintf("paper-%d", time.Now().UnixNano()),
		StopLoss:   req.StopLoss,
		TakeProfit: req.TakeProfit,
		Mode:       p.mode,
	}, nil
}

func (p *PaperExecutor) Close(pos *Position, price float64) (*OrderResult, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.position == nil {
		return nil, fmt.Errorf("no open position")
	}
	if price <= 0 {
		price = p.position.EntryPrice
	}
	closeSide := SideSell
	if p.position.Side == SideSell {
		closeSide = SideBuy
	}
	result := &OrderResult{
		Symbol:   p.position.Symbol,
		Side:     closeSide,
		Quantity: p.position.Quantity,
		Price:    price,
		OrderID:  fmt.Sprintf("paper-close-%d", time.Now().UnixNano()),
		Mode:     p.mode,
	}
	p.position = nil
	return result, nil
}
