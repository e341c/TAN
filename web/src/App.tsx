import { useCallback, useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
  createChart,
} from "lightweight-charts";
import type { CandlestickData, IChartApi, UTCTimestamp } from "lightweight-charts";
import "./App.css";

type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
};

type Snapshot = {
  candles: Candle[];
  quantile: {
    next_time: string;
    q10: number;
    q50: number;
    q90: number;
    log_returns?: number[];
  };
  trend: {
    predicted_trend: string;
    probabilities: Record<string, number>;
    logits: number[];
  };
  meta: {
    server_time: string;
    symbol: string;
    interval: string;
    bars: number;
    trend_run_id?: string;
    trend_last_ts?: string | null;
  };
};

function getCssVar(name: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function parseNextTimeUnix(iso: string): number {
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) {
    throw new Error(`Invalid next_time: ${iso}`);
  }
  return Math.floor(ms / 1000);
}

function intervalToSeconds(interval?: string): number {
  if (!interval) return 3600;
  const m = interval.trim().match(/^(\d+)\s*([mhdw])$/i);
  if (!m) return 3600;
  const n = Number(m[1]);
  if (!Number.isFinite(n) || n <= 0) return 3600;
  const unit = m[2].toLowerCase();
  switch (unit) {
    case "m":
      return n * 60;
    case "h":
      return n * 60 * 60;
    case "d":
      return n * 24 * 60 * 60;
    case "w":
      return n * 7 * 24 * 60 * 60;
    default:
      return 3600;
  }
}

function formatPrice(n: number): string {
  return n.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export default function App() {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/snapshot?bars=200");
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const detail =
          typeof body.detail === "string"
            ? body.detail
            : Array.isArray(body.detail)
              ? body.detail.map((d: { msg?: string }) => d.msg).join("; ")
              : res.statusText;
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as Snapshot;
      setSnapshot(data);
      setUpdatedAt(new Date().toISOString());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSnapshot(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) {
      return;
    }
    if (!snapshot?.candles?.length) {
      chartRef.current?.remove();
      chartRef.current = null;
      return;
    }

    chartRef.current?.remove();
    chartRef.current = null;

    const chartBg = getCssVar("--chart-bg", "#0c0d12");
    const chartText = getCssVar("--chart-text", "#a1a1aa");
    const chartGrid = getCssVar("--chart-grid", "rgba(255,255,255,0.06)");
    const chartUp = getCssVar("--chart-up", "#34d399");
    const chartDown = getCssVar("--chart-down", "#f87171");
    const chartQ10 = getCssVar("--chart-q10", "#f87171");
    const chartQ50 = getCssVar("--chart-q50", "#a78bfa");
    const chartQ90 = getCssVar("--chart-q90", "#34d399");

    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: chartBg },
        textColor: chartText,
      },
      grid: {
        vertLines: { color: chartGrid },
        horzLines: { color: chartGrid },
      },
      width: el.clientWidth,
      height: el.clientHeight,
      timeScale: {
        borderColor: chartGrid,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: { borderColor: chartGrid },
    });

    chartRef.current = chart;

    const candle = chart.addSeries(CandlestickSeries, {
      upColor: chartUp,
      downColor: chartDown,
      borderVisible: false,
      wickUpColor: chartUp,
      wickDownColor: chartDown,
    });

    const rows: CandlestickData[] = snapshot.candles
      .slice()
      .sort((a, b) => a.time - b.time)
      .reduce<CandlestickData[]>((acc, c) => {
        const next: CandlestickData = {
          time: c.time as UTCTimestamp,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        };
        const prev = acc[acc.length - 1];
        if (!prev || prev.time !== next.time) {
          acc.push(next);
        } else {
          acc[acc.length - 1] = next;
        }
        return acc;
      }, []);
    candle.setData(rows);

    const lastRow = rows[rows.length - 1];
    const lastT = lastRow.time as UTCTimestamp;
    let nextT: number;
    try {
      nextT = parseNextTimeUnix(snapshot.quantile.next_time);
    } catch {
      nextT = (lastT as number) + intervalToSeconds(snapshot.meta?.interval);
    }
    const minNext = (lastT as number) + intervalToSeconds(snapshot.meta?.interval);
    const nextTime = (nextT <= (lastT as number) ? minNext : nextT) as UTCTimestamp;

    const q10 = snapshot.quantile.q10;
    const q50 = snapshot.quantile.q50;
    const q90 = snapshot.quantile.q90;

    const lineQ10 = chart.addSeries(LineSeries, {
      color: chartQ10,
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: true,
      priceLineVisible: false,
    });
    lineQ10.setData([
      { time: lastT, value: q10 },
      { time: nextTime, value: q10 },
    ]);

    const lineQ50 = chart.addSeries(LineSeries, {
      color: chartQ50,
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
    });
    lineQ50.setData([
      { time: lastT, value: q50 },
      { time: nextTime, value: q50 },
    ]);

    const lineQ90 = chart.addSeries(LineSeries, {
      color: chartQ90,
      lineWidth: 1,
      lineStyle: 2,
      lastValueVisible: true,
      priceLineVisible: false,
    });
    lineQ90.setData([
      { time: lastT, value: q90 },
      { time: nextTime, value: q90 },
    ]);

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (!hostRef.current || !chartRef.current) return;
      chartRef.current.applyOptions({
        width: hostRef.current.clientWidth,
        height: hostRef.current.clientHeight,
      });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [snapshot]);

  const trend = snapshot?.trend;
  const badgeClass =
    trend?.predicted_trend === "UP"
      ? "badge badge-up"
      : trend?.predicted_trend === "DOWN"
        ? "badge badge-down"
        : "badge badge-neutral";

  const symbol = snapshot?.meta?.symbol ?? "BTCUSDT";
  const interval = snapshot?.meta?.interval ?? "1h";
  const eyebrow = `${symbol} · ${interval}`;

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-name">TAN Forecast</span>
          <span className="brand-tagline">Quantile & trend models</span>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="btn-primary"
            onClick={() => void load()}
            disabled={loading}
          >
            {loading ? "Loading…" : "Refresh"}
          </button>
          {(updatedAt || snapshot?.meta.server_time) && (
            <div className="header-meta">
              {updatedAt && <span>UI updated: {updatedAt}</span>}
              {snapshot?.meta.server_time && (
                <span>Server: {snapshot.meta.server_time}</span>
              )}
            </div>
          )}
        </div>
      </header>

      <section className="hero">
        <span className="eyebrow">{eyebrow}</span>
        <h1>History and next-bar forecast</h1>
        <p className="hero-subtitle">
          Candlestick history with quantile bands (q10, median, q90) projected to
          the next bar open. Trend model probabilities summarize directional bias
          for the upcoming interval.
        </p>
      </section>

      <main>
        {error && <div className="alert-error">{error}</div>}

        <section className="chart-section">
          <div className="card chart-card">
            {loading && !snapshot && (
              <div className="chart-loading" aria-hidden="true">
                <div className="spinner" />
              </div>
            )}
            <div ref={hostRef} className="chart-wrap" />
          </div>
        </section>

        <div className="legend">
          <span className="chip">
            <i className="dot" style={{ background: "var(--chart-q10)" }} />
            Quantile q10
          </span>
          <span className="chip">
            <i className="dot" style={{ background: "var(--chart-q50)" }} />
            Quantile q50 (median)
          </span>
          <span className="chip">
            <i className="dot" style={{ background: "var(--chart-q90)" }} />
            Quantile q90
          </span>
          <span className="chip chip-muted">
            Lines span last closed bar → predicted next bar open
          </span>
        </div>

        {snapshot ? (
          <div className="forecast-grid">
            <div className="card forecast-card">
              <h2 className="card-title">Trend</h2>
              {trend ? (
                <>
                  <div className="trend-direction">
                    <span className="trend-label">Predicted</span>
                    <span className={badgeClass}>{trend.predicted_trend}</span>
                  </div>
                  {trend.probabilities &&
                    Object.entries(trend.probabilities).map(([k, v]) => (
                      <div key={k} className="stat-row">
                        <span>{k}</span>
                        <strong>{(v * 100).toFixed(2)}%</strong>
                      </div>
                    ))}
                </>
              ) : (
                <p className="empty-state">No trend data</p>
              )}
            </div>

            <div className="card forecast-card">
              <h2 className="card-title">Quantile</h2>
              <div className="stat-row">
                <span>Next bar</span>
                <strong className="meta-value">
                  {snapshot.quantile.next_time}
                </strong>
              </div>
              <div className="stat-row">
                <span>q10</span>
                <strong>{formatPrice(snapshot.quantile.q10)}</strong>
              </div>
              <div className="stat-row">
                <span>q50 (median)</span>
                <strong>{formatPrice(snapshot.quantile.q50)}</strong>
              </div>
              <div className="stat-row">
                <span>q90</span>
                <strong>{formatPrice(snapshot.quantile.q90)}</strong>
              </div>
            </div>

            <div className="card forecast-card">
              <h2 className="card-title">Meta</h2>
              <div className="stat-row">
                <span>Symbol</span>
                <strong>{snapshot.meta.symbol}</strong>
              </div>
              <div className="stat-row">
                <span>Interval</span>
                <strong>{snapshot.meta.interval}</strong>
              </div>
              <div className="stat-row">
                <span>Bars</span>
                <strong>{snapshot.meta.bars}</strong>
              </div>
              {snapshot.meta.trend_run_id && (
                <div className="stat-row">
                  <span>Run</span>
                  <strong className="meta-value">
                    {snapshot.meta.trend_run_id}
                  </strong>
                </div>
              )}
            </div>
          </div>
        ) : (
          !loading &&
          !error && (
            <div className="card empty-state">
              Connect the API and press Refresh to load a snapshot.
            </div>
          )
        )}
      </main>
    </div>
  );
}
