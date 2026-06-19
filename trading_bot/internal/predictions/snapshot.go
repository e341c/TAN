package predictions

import "time"

type Candle struct {
	Time  int64   `json:"time"`
	Open  float64 `json:"open"`
	High  float64 `json:"high"`
	Low   float64 `json:"low"`
	Close float64 `json:"close"`
}

type Quantile struct {
	NextTime   string    `json:"next_time"`
	Q10        float64   `json:"q10"`
	Q50        float64   `json:"q50"`
	Q90        float64   `json:"q90"`
	LogReturns []float64 `json:"log_returns"`
}

type Trend struct {
	PredictedTrend string             `json:"predicted_trend"`
	Probabilities  map[string]float64 `json:"probabilities"`
	Logits         []float64          `json:"logits"`
}

type Meta struct {
	ServerTime   string `json:"server_time"`
	Symbol       string `json:"symbol"`
	Interval     string `json:"interval"`
	Bars         int    `json:"bars"`
	TrendRunID   string `json:"trend_run_id"`
	TrendLastTS  string `json:"trend_last_ts"`
}

type Snapshot struct {
	Candles  []Candle `json:"candles"`
	Quantile Quantile `json:"quantile"`
	Trend    Trend    `json:"trend"`
	Meta     Meta     `json:"meta"`
}

func (s *Snapshot) TargetTS() (time.Time, error) {
	return time.Parse(time.RFC3339, s.Quantile.NextTime)
}

func (s *Snapshot) LastClose() float64 {
	if len(s.Candles) == 0 {
		return 0
	}
	return s.Candles[len(s.Candles)-1].Close
}

func (s *Snapshot) SpreadPct() float64 {
	if s.Quantile.Q50 == 0 {
		return 0
	}
	return (s.Quantile.Q90 - s.Quantile.Q10) / s.Quantile.Q50 * 100
}
