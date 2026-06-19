package exchange

import "time"

type Side string

const (
	SideBuy  Side = "BUY"
	SideSell Side = "SELL"
)

type Position struct {
	Side              Side      `json:"side"`
	Symbol            string    `json:"symbol"`
	Quantity          float64   `json:"quantity"`
	EntryPrice        float64   `json:"entry_price"`
	OpenedAt          time.Time `json:"opened_at"`
	StopLoss          float64   `json:"stop_loss"`
	TakeProfit        float64   `json:"take_profit"`
	StopLossOrderID   string    `json:"stop_loss_order_id,omitempty"`
	TakeProfitOrderID string    `json:"take_profit_order_id,omitempty"`
}

type OrderRequest struct {
	Symbol     string
	Side       Side
	QuoteUSDT  float64
	Price      float64
	StopLoss   float64
	TakeProfit float64
}

type OrderResult struct {
	Symbol     string
	Side       Side
	Quantity   float64
	Price      float64
	OrderID    string
	StopLoss   float64
	TakeProfit float64
	Mode       string
}

type Executor interface {
	Open(req OrderRequest) (*OrderResult, error)
	Close(pos *Position, price float64) (*OrderResult, error)
	GetPosition(symbol string) *Position
	SyncPosition(symbol string) (*Position, error)
	RestorePosition(pos *Position)
	Mode() string
}
