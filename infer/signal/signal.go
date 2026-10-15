package signal

import (
	"time"

	"github.com/shopspring/decimal"
)

type Side int8

const (
	Short Side = -1
	Flat  Side = 0
	Long  Side = 1
)

type Signal struct {
	ID           string
	Symbol       string
	Side         Side
	Confidence   float64
	TargetWeight float64
	Price        decimal.Decimal
	BarTime      time.Time
	ModelVersion string
}

type Input struct {
	Symbol       string
	BarTime      time.Time
	PUp          float64 // вероятность роста от модели, 0-1
	Sigma        float64
	Price        decimal.Decimal
	ModelVersion string
}
