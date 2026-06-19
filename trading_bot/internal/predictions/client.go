package predictions

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

type Client interface {
	Fetch(ctx context.Context) (*Snapshot, error)
}

type APIClient struct {
	baseURL string
	http    *http.Client
}

func NewAPIClient(baseURL string) *APIClient {
	return &APIClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 60 * time.Second},
	}
}

func (c *APIClient) Fetch(ctx context.Context) (*Snapshot, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/snapshot?bars=45", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("api request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("api status %d: %s", resp.StatusCode, string(body))
	}

	var snap Snapshot
	if err := json.Unmarshal(body, &snap); err != nil {
		return nil, fmt.Errorf("decode snapshot: %w", err)
	}
	return &snap, nil
}

type DBClient struct {
	conn   *pgx.Conn
	symbol string
	tf     string
}

func NewDBClient(ctx context.Context, databaseURL, symbol, timeframe string) (*DBClient, error) {
	conn, err := pgx.Connect(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("db connect: %w", err)
	}
	return &DBClient{conn: conn, symbol: symbol, tf: timeframe}, nil
}

func (c *DBClient) Close(ctx context.Context) error {
	return c.conn.Close(ctx)
}

func (c *DBClient) Fetch(ctx context.Context) (*Snapshot, error) {
	const q = `
SELECT
  q.target_ts,
  q.q10, q.q50, q.q90,
  q.log_return_q10, q.log_return_q50, q.log_return_q90,
  q.last_close,
  t.predicted_trend,
  t.prob_down, t.prob_up,
  t.logits,
  t.trend_last_ts
FROM quantile_predictions q
JOIN trend_predictions t
  ON q.symbol = t.symbol
 AND q.timeframe = t.timeframe
 AND q.target_ts = t.target_ts
WHERE q.symbol = $1 AND q.timeframe = $2
ORDER BY q.target_ts DESC
LIMIT 1`

	var targetTS time.Time
	var q10, q50, q90 float64
	var lr10, lr50, lr90 *float64
	var lastClose *float64
	var predictedTrend string
	var probDown, probUp *float64
	var logitsJSON *string
	var trendLastTS *time.Time

	err := c.conn.QueryRow(ctx, q, c.symbol, c.tf).Scan(
		&targetTS, &q10, &q50, &q90,
		&lr10, &lr50, &lr90,
		&lastClose,
		&predictedTrend,
		&probDown, &probUp,
		&logitsJSON,
		&trendLastTS,
	)
	if err != nil {
		return nil, fmt.Errorf("db query: %w", err)
	}

	snap := &Snapshot{
		Quantile: Quantile{
			NextTime: targetTS.UTC().Format(time.RFC3339),
			Q10:      q10,
			Q50:      q50,
			Q90:      q90,
		},
		Trend: Trend{
			PredictedTrend: predictedTrend,
			Probabilities:  map[string]float64{},
		},
		Meta: Meta{
			Symbol:   c.symbol,
			Interval: c.tf,
		},
	}

	if lr10 != nil && lr50 != nil && lr90 != nil {
		snap.Quantile.LogReturns = []float64{*lr10, *lr50, *lr90}
	}
	if probDown != nil {
		snap.Trend.Probabilities["DOWN"] = *probDown
	}
	if probUp != nil {
		snap.Trend.Probabilities["UP"] = *probUp
	}
	if logitsJSON != nil && *logitsJSON != "" {
		var logits []float64
		if err := json.Unmarshal([]byte(*logitsJSON), &logits); err == nil {
			snap.Trend.Logits = logits
		}
	}
	if trendLastTS != nil {
		snap.Meta.TrendLastTS = trendLastTS.UTC().Format(time.RFC3339)
	}
	if lastClose != nil {
		snap.Candles = []Candle{{Close: *lastClose, Time: targetTS.Add(-time.Hour).Unix()}}
	}

	return snap, nil
}

func NewClient(source, apiBaseURL, databaseURL, symbol, timeframe string) (Client, func(context.Context) error, error) {
	switch source {
	case "api":
		return NewAPIClient(apiBaseURL), func(context.Context) error { return nil }, nil
	case "database":
		ctx := context.Background()
		db, err := NewDBClient(ctx, databaseURL, symbol, timeframe)
		if err != nil {
			return nil, nil, err
		}
		return db, db.Close, nil
	default:
		return nil, nil, fmt.Errorf("unknown predictions source %q", source)
	}
}
