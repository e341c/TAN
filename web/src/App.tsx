import { useCallback, useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
  createChart,
  createSeriesMarkers,
} from "lightweight-charts";
import type {
  CandlestickData,
  IChartApi,
  LineData,
  SeriesMarker,
  UTCTimestamp,
  WhitespaceData,
} from "lightweight-charts";
import "./App.css";

type Candle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
};

type QuantileHistoryItem = {
  candle_time: string;
  target_time: string;
  candle_unix: number;
  target_unix: number;
  q10: number;
  q50: number;
  q90: number;
  last_close?: number;
};

type TrendHistoryItem = {
  candle_time: string;
  target_time: string;
  candle_unix: number;
  target_unix: number;
  predicted_trend: string;
  prob_up?: number;
  prob_down?: number;
  // Rows from different models answer different questions; the chart needs this
  // to avoid drawing a 24h barrier call and a 1h close call identically.
  target_mode?: string;
  horizon_hours?: number;
  model_run_id?: string;
  upper_mult?: number;
  lower_mult?: number;
  vol_window?: number;
  max_holding?: number;
};

type HorizonSegment = {
  startTime: number;
  endTime: number;
  startPrice: number;
  endPrice: number;
  isUp: boolean;
};

/**
 * Rolling sigma (std of log returns) indexed by candle time.
 *
 * Recomputed client-side rather than stored, using the same window the model
 * used, so the barrier levels drawn on the chart are the ones the model actually
 * targeted rather than an approximation.
 */
type ClosePoint = { time: unknown; close: number };

function buildSigmaLookup(candles: ClosePoint[], window: number): Map<number, number> {
  const out = new Map<number, number>();
  const logRet: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    logRet.push(Math.log(candles[i].close / candles[i - 1].close));
  }
  for (let i = window; i < logRet.length; i++) {
    const slice = logRet.slice(i - window, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const varc =
      slice.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (slice.length - 1);
    // logRet[i] spans candles[i] -> candles[i+1], so index i+1 is the anchor bar.
    out.set(candles[i + 1].time as number, Math.sqrt(varc));
  }
  return out;
}

/**
 * Turn barrier predictions into non-overlapping horizon segments.
 *
 * Predictions arrive hourly but each looks 24 bars ahead, so consecutive ones
 * share ~96% of their outcome window. Drawing all of them would imply far more
 * independent evidence than exists; stepping by the holding period shows only
 * predictions whose windows do not overlap.
 */
function buildHorizonSegments(
  history: TrendHistoryItem[],
  candles: ClosePoint[],
  barSeconds: number,
): HorizonSegment[] {
  const barrier = history.filter((h) => h.target_mode === "triple_barrier");
  if (barrier.length === 0 || candles.length < 2) {
    return [];
  }

  const closeAt = new Map<number, number>();
  for (const c of candles) {
    closeAt.set(c.time as number, c.close);
  }
  const volWindow = barrier[0].vol_window ?? 72;
  const sigmaAt = buildSigmaLookup(candles, volWindow);

  const segments: HorizonSegment[] = [];
  let nextFreeTime = -Infinity;

  for (const item of barrier) {
    const anchor = item.candle_unix;
    if (anchor < nextFreeTime) {
      continue; // window still covered by the previous segment
    }
    const close = closeAt.get(anchor);
    const sigma = sigmaAt.get(anchor);
    if (close === undefined || sigma === undefined || !Number.isFinite(sigma)) {
      continue;
    }
    const holding = item.max_holding ?? 24;
    const isUp = item.predicted_trend.toUpperCase() === "UP";
    const mult = isUp ? (item.upper_mult ?? 2.5) : -(item.lower_mult ?? 2.5);

    segments.push({
      startTime: anchor,
      endTime: anchor + holding * barSeconds,
      startPrice: close,
      endPrice: close * Math.exp(mult * sigma),
      isUp,
    });
    nextFreeTime = anchor + holding * barSeconds;
  }
  return segments;
}

/** Line data for one direction, with whitespace so segments never join up. */
function horizonSeriesData(
  segments: HorizonSegment[],
  isUp: boolean,
): (LineData<UTCTimestamp> | WhitespaceData<UTCTimestamp>)[] {
  const out: (LineData<UTCTimestamp> | WhitespaceData<UTCTimestamp>)[] = [];
  for (const s of segments) {
    if (s.isUp !== isUp) {
      continue;
    }
    if (out.length > 0) {
      out.push({ time: (s.startTime - 1) as UTCTimestamp });
    }
    out.push({ time: s.startTime as UTCTimestamp, value: s.startPrice });
    out.push({ time: s.endTime as UTCTimestamp, value: s.endPrice });
  }
  return out;
}

type Snapshot = {
  candles: Candle[];
  quantile: {
    quantiles?: Record<string, number>;
    quantile_levels?: number[];
    sigma?: number;
    next_time: string;
    q10: number;
    q50: number;
    q90: number;
    log_returns?: number[];
  };
  quantile_history?: QuantileHistoryItem[];
  trend_history?: TrendHistoryItem[];
  trend: {
    predicted_trend: string;
    probabilities: Record<string, number>;
    logits: number[];
    target?: TrendTarget;
    horizon_hours?: number;
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

// What a trend model was trained to predict. Sent with every forecast so the
// UI never has to guess the horizon behind an "UP"/"DOWN".
type TrendTarget = {
  mode: string;
  horizon_hours?: number;
  threshold?: number;
  upper_mult?: number;
  lower_mult?: number;
  max_holding?: number;
  description?: string;
};

function describeTarget(target?: TrendTarget): string {
  if (!target?.mode) {
    return "next 1h close";
  }
  if (target.mode === "triple_barrier") {
    const k = target.upper_mult ?? 2.5;
    const hours = target.horizon_hours ?? target.max_holding ?? 24;
    return `hits ±${k}σ within ${hours}h`;
  }
  const pct = ((target.threshold ?? 0.001) * 100).toFixed(2);
  const hours = target.horizon_hours ?? 1;
  return `next ${hours}h move > ±${pct}%`;
}

type StatsSeriesPoint = {
  target_time: string;
  target_unix: number;
  actual_close: number;
  q50_error_pct: number;
  in_band: boolean;
  breach_pct: number;
  predicted_trend: string;
  actual_trend: string | null;
  trend_correct: boolean | null;
};

type ModelStats = {
  meta: {
    symbol: string;
    interval: string;
    evaluated_count: number;
    neutral_skipped: number;
    period_start: string | null;
    period_end: string | null;
    trend_threshold: number;
    trend_target_modes?: Record<string, number>;
    trend_run_ids?: Record<string, number>;
    mixed_targets?: boolean;
  };
  quantile: {
    coverage_rate: number | null;
    coverage_nominal?: number;
    coverage_ci95?: [number, number] | null;
    breach_below_rate?: number | null;
    breach_above_rate?: number | null;
    mae_q50: number | null;
    mape_q50: number | null;
    mae_naive?: number | null;
    mape_naive?: number | null;
    mae_skill_vs_naive?: number | null;
    avg_breach_pct: number | null;
    avg_interval_width_pct?: number | null;
    pinball: { "q0.1": number | null; "q0.5": number | null; "q0.9": number | null };
  };
  trend: {
    accuracy: number | null;
    accuracy_ci95?: [number, number] | null;
    accuracy_high_confidence: number | null;
    high_confidence_count?: number;
    majority_baseline?: number | null;
    edge_vs_majority?: number | null;
    precision_up: number | null;
    recall_up: number | null;
    confusion: { tp: number; tn: number; fp: number; fn: number };
  };
  series: StatsSeriesPoint[];
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

function formatPctFraction(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || !Number.isFinite(n)) {
    return "—";
  }
  return `${(n * 100).toFixed(digits)}%`;
}

function formatNullable(
  n: number | null | undefined,
  formatter: (v: number) => string,
): string {
  if (n === null || n === undefined || !Number.isFinite(n)) {
    return "—";
  }
  return formatter(n);
}

function withAlpha(color: string, alpha: number): string {
  if (color.startsWith("#") && color.length === 7) {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  return color;
}

function filterHistoryForLive(
  history: QuantileHistoryItem[] | undefined,
  liveNextTimeIso: string,
): QuantileHistoryItem[] {
  if (!history?.length) {
    return [];
  }
  let liveTargetUnix: number;
  try {
    liveTargetUnix = parseNextTimeUnix(liveNextTimeIso);
  } catch {
    return history;
  }
  return history.filter((item) => item.target_unix !== liveTargetUnix);
}

function filterTrendHistoryForLive(
  history: TrendHistoryItem[] | undefined,
  liveNextTimeIso: string,
): TrendHistoryItem[] {
  if (!history?.length) {
    return [];
  }
  let liveTargetUnix: number;
  try {
    liveTargetUnix = parseNextTimeUnix(liveNextTimeIso);
  } catch {
    return history;
  }
  return history.filter((item) => item.target_unix !== liveTargetUnix);
}

function trendProbForItem(item: TrendHistoryItem): number | undefined {
  const trend = item.predicted_trend.toUpperCase();
  if (trend === "UP") {
    return item.prob_up;
  }
  if (trend === "DOWN") {
    return item.prob_down;
  }
  return undefined;
}

function trendMarkerText(
  predictedTrend: string,
  prob?: number,
  probabilities?: Record<string, number>,
  horizonSuffix?: string,
): string {
  const trend = predictedTrend.toUpperCase();
  let p = prob;
  if (p === undefined && probabilities) {
    p = probabilities[trend] ?? probabilities[trend.toLowerCase()];
  }
  const suffix = horizonSuffix ? ` ${horizonSuffix}` : "";
  if (p !== undefined && Number.isFinite(p)) {
    return `${trend} ${(p * 100).toFixed(0)}%${suffix}`;
  }
  return `${trend}${suffix}`;
}

function buildTrendMarkers(
  history: TrendHistoryItem[],
  liveTrend: Snapshot["trend"] | undefined,
  lastT: UTCTimestamp,
  candleTimes: Set<number>,
  chartUp: string,
  chartDown: string,
): SeriesMarker<UTCTimestamp>[] {
  const markers: SeriesMarker<UTCTimestamp>[] = [];

  for (const item of history) {
    if (!candleTimes.has(item.candle_unix)) {
      continue;
    }
    const trend = item.predicted_trend.toUpperCase();
    if (trend !== "UP" && trend !== "DOWN") {
      continue;
    }
    const isUp = trend === "UP";
    // Barrier predictions cover many bars, next-bar ones a single bar. Drawing
    // them identically implies the same horizon, so barrier markers get a circle
    // and carry their horizon in the label.
    const isBarrier = item.target_mode === "triple_barrier";
    markers.push({
      time: item.candle_unix as UTCTimestamp,
      position: isUp ? "belowBar" : "aboveBar",
      shape: isBarrier ? "circle" : isUp ? "arrowUp" : "arrowDown",
      color: withAlpha(isUp ? chartUp : chartDown, isBarrier ? 0.55 : 0.45),
      text: trendMarkerText(
        trend,
        trendProbForItem(item),
        undefined,
        isBarrier && item.horizon_hours ? `${item.horizon_hours}h` : undefined,
      ),
    });
  }

  if (liveTrend?.predicted_trend) {
    const trend = liveTrend.predicted_trend.toUpperCase();
    if (trend === "UP" || trend === "DOWN") {
      const isUp = trend === "UP";
      markers.push({
        time: lastT,
        position: isUp ? "belowBar" : "aboveBar",
        shape: isUp ? "arrowUp" : "arrowDown",
        color: isUp ? chartUp : chartDown,
        text: trendMarkerText(trend, undefined, liveTrend.probabilities),
      });
    }
  }

  markers.sort((a, b) => (a.time as number) - (b.time as number));
  return markers;
}

function buildHistoryLineData(
  history: QuantileHistoryItem[],
  pick: (item: QuantileHistoryItem) => number,
): (LineData<UTCTimestamp> | WhitespaceData<UTCTimestamp>)[] {
  const data: (LineData<UTCTimestamp> | WhitespaceData<UTCTimestamp>)[] = [];
  for (let i = 0; i < history.length; i++) {
    const item = history[i];
    const candleT = item.candle_unix as UTCTimestamp;
    const targetT = item.target_unix as UTCTimestamp;
    const value = pick(item);
    if (i > 0) {
      data.push({ time: candleT });
    }
    data.push({ time: candleT, value });
    data.push({ time: targetT, value });
  }
  return data;
}

export default function App() {
  const hostRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const statsHostRef = useRef<HTMLDivElement>(null);
  const statsChartRef = useRef<IChartApi | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [statsLoading, setStatsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setStatsLoading(true);
    setError(null);
    setStatsError(null);
    try {
      const [snapRes, statsRes] = await Promise.all([
        fetch("/api/snapshot?bars=200"),
        fetch("/api/model-stats?limit=200"),
      ]);

      if (!snapRes.ok) {
        const body = await snapRes.json().catch(() => ({}));
        const detail =
          typeof body.detail === "string"
            ? body.detail
            : Array.isArray(body.detail)
              ? body.detail.map((d: { msg?: string }) => d.msg).join("; ")
              : snapRes.statusText;
        throw new Error(detail || `HTTP ${snapRes.status}`);
      }
      const data = (await snapRes.json()) as Snapshot;
      setSnapshot(data);

      if (!statsRes.ok) {
        const body = await statsRes.json().catch(() => ({}));
        const detail =
          typeof body.detail === "string"
            ? body.detail
            : Array.isArray(body.detail)
              ? body.detail.map((d: { msg?: string }) => d.msg).join("; ")
              : statsRes.statusText;
        setStatsError(detail || `HTTP ${statsRes.status}`);
        setModelStats(null);
      } else {
        setModelStats((await statsRes.json()) as ModelStats);
      }

      setUpdatedAt(new Date().toISOString());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSnapshot(null);
    } finally {
      setLoading(false);
      setStatsLoading(false);
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

    const candleTimes = new Set(rows.map((r) => r.time as number));
    const trendHistory = filterTrendHistoryForLive(
      snapshot.trend_history,
      snapshot.quantile.next_time,
    );
    const trendMarkers = buildTrendMarkers(
      trendHistory,
      snapshot.trend,
      rows[rows.length - 1].time as UTCTimestamp,
      candleTimes,
      chartUp,
      chartDown,
    );
    if (trendMarkers.length > 0) {
      createSeriesMarkers(candle, trendMarkers);
    }

    // Barrier horizon: a line from the anchor close to the barrier level the
    // model expects to be touched first, spanning the full holding period. This
    // is what makes a 24h prediction legible on an hourly chart — the marker
    // alone sits on one candle and reads as a one-bar call.
    const barSeconds = intervalToSeconds(snapshot.meta?.interval);
    const horizonSegments = buildHorizonSegments(trendHistory, rows, barSeconds);
    if (horizonSegments.length > 0) {
      for (const isUp of [true, false]) {
        const data = horizonSeriesData(horizonSegments, isUp);
        if (data.length === 0) {
          continue;
        }
        const series = chart.addSeries(LineSeries, {
          color: withAlpha(isUp ? chartUp : chartDown, 0.5),
          lineWidth: 2,
          lineStyle: 2, // dashed: a projection, not observed price
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        series.setData(data);
      }
    }

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

    const history = filterHistoryForLive(
      snapshot.quantile_history,
      snapshot.quantile.next_time,
    );
    if (history.length > 0) {
      const histQ10 = chart.addSeries(LineSeries, {
        color: withAlpha(chartQ10, 0.35),
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      histQ10.setData(buildHistoryLineData(history, (item) => item.q10));

      const histQ50 = chart.addSeries(LineSeries, {
        color: withAlpha(chartQ50, 0.35),
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      histQ50.setData(buildHistoryLineData(history, (item) => item.q50));

      const histQ90 = chart.addSeries(LineSeries, {
        color: withAlpha(chartQ90, 0.35),
        lineWidth: 1,
        lineStyle: 2,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      histQ90.setData(buildHistoryLineData(history, (item) => item.q90));
    }

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

  useEffect(() => {
    const el = statsHostRef.current;
    if (!el) {
      return;
    }
    if (!modelStats?.series?.length) {
      statsChartRef.current?.remove();
      statsChartRef.current = null;
      return;
    }

    statsChartRef.current?.remove();
    statsChartRef.current = null;

    const chartBg = getCssVar("--chart-bg", "#0c0d12");
    const chartText = getCssVar("--chart-text", "#a1a1aa");
    const chartGrid = getCssVar("--chart-grid", "rgba(255,255,255,0.06)");
    const chartQ50 = getCssVar("--chart-q50", "#a78bfa");

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
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
      },
    });
    statsChartRef.current = chart;

    const line = chart.addSeries(LineSeries, {
      color: chartQ50,
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      title: "q50 error %",
    });
    line.setData(
      modelStats.series.map((p) => ({
        time: p.target_unix as UTCTimestamp,
        value: p.q50_error_pct * 100,
      })),
    );

    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (!statsHostRef.current || !statsChartRef.current) return;
      statsChartRef.current.applyOptions({
        width: statsHostRef.current.clientWidth,
        height: statsHostRef.current.clientHeight,
      });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      statsChartRef.current = null;
    };
  }, [modelStats]);

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
            disabled={loading || statsLoading}
          >
            {loading || statsLoading ? "Loading…" : "Refresh"}
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
            Quantile q10 (live)
          </span>
          <span className="chip">
            <i className="dot" style={{ background: "var(--chart-q50)" }} />
            Quantile q50 (live)
          </span>
          <span className="chip">
            <i className="dot" style={{ background: "var(--chart-q90)" }} />
            Quantile q90 (live)
          </span>
          <span className="chip chip-muted">
            Faded dashed lines: historical forecasts from DB
          </span>
          {/* Spells out the two different horizons on one chart: the quantile
              band is a 1-bar forecast, the trend horizon spans 24 bars. */}
          <span className="chip chip-muted">
            Quantile band = next 1h · trend horizon = 24h
          </span>
          <span className="chip chip-muted">
            Sloped dashed lines: barrier target over the holding window (only
            non-overlapping predictions shown — hourly ones overlap ~96%)
          </span>
          <span className="chip chip-muted">
            Solid lines: current forecast (last bar → next open)
          </span>
          <span className="chip">
            <i className="dot" style={{ background: "var(--chart-up)" }} />
            ▲ UP / ▼ DOWN (live)
          </span>
          <span className="chip chip-muted">
            Faded arrows: historical trend from DB
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
                  {/* UP/DOWN alone is ambiguous: the same label means "next hour
                      closes higher" under next_bar and "hits +k*sigma before
                      -k*sigma within N bars" under triple_barrier. Always show
                      which question was answered. */}
                  <div className="trend-target">{describeTarget(trend.target)}</div>
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
              {/* Show every level the model actually predicts. q10/q90 are
                  interpolated from this grid for backward compatibility, so
                  displaying only those hides half the distribution. */}
              {snapshot.quantile.quantiles ? (
                Object.entries(snapshot.quantile.quantiles).map(([k, v]) => {
                  const isMedian = k === "q50";
                  return (
                    <div
                      key={k}
                      className={`stat-row ${isMedian ? "stat-row-median" : ""}`}
                    >
                      <span>{isMedian ? "q50 (median)" : k}</span>
                      <strong>{formatPrice(v)}</strong>
                    </div>
                  );
                })
              ) : (
                <>
                  <div className="stat-row">
                    <span>q10</span>
                    <strong>{formatPrice(snapshot.quantile.q10)}</strong>
                  </div>
                  <div className="stat-row stat-row-median">
                    <span>q50 (median)</span>
                    <strong>{formatPrice(snapshot.quantile.q50)}</strong>
                  </div>
                  <div className="stat-row">
                    <span>q90</span>
                    <strong>{formatPrice(snapshot.quantile.q90)}</strong>
                  </div>
                </>
              )}
              <div className="stat-row stat-row-sub">
                <span>80% band</span>
                <strong className="stat-baseline">
                  {formatPrice(snapshot.quantile.q10)} –{" "}
                  {formatPrice(snapshot.quantile.q90)}
                </strong>
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

        <section className="stats-section">
          <div className="stats-heading">
            <h2>Model performance</h2>
            <p>
              Mature predictions scored against Binance actual closes. Quantile
              coverage and trend direction accuracy over the stored history.
            </p>
          </div>

          {statsError && <div className="alert-error">{statsError}</div>}

          {statsLoading && !modelStats && (
            <div className="card empty-state">Loading model statistics…</div>
          )}

          {modelStats && modelStats.meta.evaluated_count === 0 && !statsLoading && (
            <div className="card empty-state">
              Not enough mature predictions in the database yet.
            </div>
          )}

          {modelStats && modelStats.meta.evaluated_count > 0 && (
            <>
              <div className="stats-meta card forecast-card">
                <div className="stat-row">
                  <span>Evaluated</span>
                  <strong>{modelStats.meta.evaluated_count}</strong>
                </div>
                <div className="stat-row">
                  <span>Period</span>
                  <strong className="meta-value">
                    {modelStats.meta.period_start} → {modelStats.meta.period_end}
                  </strong>
                </div>
                <div className="stat-row">
                  <span>Unresolved / skipped</span>
                  <strong>{modelStats.meta.neutral_skipped}</strong>
                </div>
                {modelStats.meta.trend_target_modes && (
                  <div className="stat-row">
                    <span>Trend targets</span>
                    <strong className="meta-value">
                      {Object.entries(modelStats.meta.trend_target_modes)
                        .map(([m, n]) => `${m} ×${n}`)
                        .join(", ")}
                    </strong>
                  </div>
                )}
              </div>

              {/* Rows produced by different targets are not comparable: one asks
                  "does the next hour close higher", the other "does price hit
                  +k*sigma before -k*sigma within 24h". Averaging them into a
                  single accuracy is misleading, so say so plainly. */}
              {modelStats.meta.mixed_targets && (
                <div className="card stats-warning">
                  ⚠ These statistics mix predictions from different targets (
                  {Object.keys(modelStats.meta.trend_target_modes ?? {}).join(" + ")}
                  ). "UP"/"DOWN" means a different thing under each, so the
                  combined accuracy below is not a like-for-like measure.
                </div>
              )}

              <div className="stats-grid">
                <div className="card forecast-card">
                  <h2 className="card-title">Quantile stats</h2>
                  {/* Every figure is shown against the baseline that decides
                      whether it means anything. A coverage without its nominal,
                      or an MAE without persistence to compare to, reads as a
                      result when it may be none. */}
                  <div className="stat-row">
                    <span>Coverage</span>
                    <strong>
                      {formatPctFraction(modelStats.quantile.coverage_rate)}
                      <span className="stat-baseline">
                        {" "}
                        vs {formatPctFraction(modelStats.quantile.coverage_nominal ?? 0.8)} nominal
                      </span>
                    </strong>
                  </div>
                  {modelStats.quantile.coverage_ci95 && (
                    <div className="stat-row stat-row-sub">
                      <span>95% CI</span>
                      <strong className="stat-baseline">
                        {formatPctFraction(modelStats.quantile.coverage_ci95[0])} –{" "}
                        {formatPctFraction(modelStats.quantile.coverage_ci95[1])}
                      </strong>
                    </div>
                  )}
                  <div className="stat-row stat-row-sub">
                    <span>Breach below / above</span>
                    <strong className="stat-baseline">
                      {formatPctFraction(modelStats.quantile.breach_below_rate)} /{" "}
                      {formatPctFraction(modelStats.quantile.breach_above_rate)}
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>MAE q50</span>
                    <strong>
                      {formatNullable(modelStats.quantile.mae_q50, formatPrice)}
                      <span className="stat-baseline">
                        {" "}
                        vs {formatNullable(modelStats.quantile.mae_naive, formatPrice)} naive
                      </span>
                    </strong>
                  </div>
                  {/* The number that decides whether the point forecast is worth
                      anything: does it beat "price stays where it is"? */}
                  <div
                    className={`stat-row ${
                      (modelStats.quantile.mae_skill_vs_naive ?? 0) > 0.01
                        ? "stat-good"
                        : "stat-warn"
                    }`}
                  >
                    <span>Skill vs persistence</span>
                    <strong>
                      {formatNullable(
                        modelStats.quantile.mae_skill_vs_naive,
                        (v) => `${v > 0 ? "+" : ""}${v.toFixed(4)}`,
                      )}
                      {(modelStats.quantile.mae_skill_vs_naive ?? 0) <= 0.01 && (
                        <span className="stat-baseline"> (no edge)</span>
                      )}
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>MAPE q50</span>
                    <strong>
                      {formatPctFraction(modelStats.quantile.mape_q50, 3)}
                      <span className="stat-baseline">
                        {" "}
                        vs {formatPctFraction(modelStats.quantile.mape_naive, 3)} naive
                      </span>
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Avg breach / width</span>
                    <strong>
                      {formatPctFraction(modelStats.quantile.avg_breach_pct, 3)}
                      <span className="stat-baseline">
                        {" "}
                        / {formatNullable(
                          modelStats.quantile.avg_interval_width_pct,
                          (v) => `${v.toFixed(3)}%`,
                        )} wide
                      </span>
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Pinball q10 / q50 / q90</span>
                    <strong className="meta-value">
                      {formatNullable(modelStats.quantile.pinball["q0.1"], (v) =>
                        v.toExponential(2),
                      )}{" "}
                      /{" "}
                      {formatNullable(modelStats.quantile.pinball["q0.5"], (v) =>
                        v.toExponential(2),
                      )}{" "}
                      /{" "}
                      {formatNullable(modelStats.quantile.pinball["q0.9"], (v) =>
                        v.toExponential(2),
                      )}
                    </strong>
                  </div>
                </div>

                <div className="card forecast-card">
                  <h2 className="card-title">Trend stats</h2>
                  {/* Accuracy alone cannot be read: "always predict the more
                      common class" already scores the majority baseline, so the
                      edge over it is the real number. */}
                  <div className="stat-row">
                    <span>Accuracy</span>
                    <strong>
                      {formatPctFraction(modelStats.trend.accuracy)}
                      <span className="stat-baseline">
                        {" "}
                        vs {formatPctFraction(modelStats.trend.majority_baseline)} majority
                      </span>
                    </strong>
                  </div>
                  <div
                    className={`stat-row ${
                      (modelStats.trend.edge_vs_majority ?? 0) > 0.005
                        ? "stat-good"
                        : "stat-warn"
                    }`}
                  >
                    <span>Edge vs majority</span>
                    <strong>
                      {formatNullable(
                        modelStats.trend.edge_vs_majority,
                        (v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(2)} pp`,
                      )}
                    </strong>
                  </div>
                  {modelStats.trend.accuracy_ci95 && (
                    <div className="stat-row stat-row-sub">
                      <span>95% CI</span>
                      <strong className="stat-baseline">
                        {formatPctFraction(modelStats.trend.accuracy_ci95[0])} –{" "}
                        {formatPctFraction(modelStats.trend.accuracy_ci95[1])}
                      </strong>
                    </div>
                  )}
                  {/* Always paired with its count: a 75% from 8 samples and one
                      from 800 render identically without it. */}
                  <div className="stat-row">
                    <span>Accuracy (conf ≥ 60%)</span>
                    <strong>
                      {(modelStats.trend.high_confidence_count ?? 0) === 0 ? (
                        <span className="stat-baseline">no predictions ≥ 60%</span>
                      ) : (
                        <>
                          {formatPctFraction(
                            modelStats.trend.accuracy_high_confidence,
                          )}
                          <span className="stat-baseline">
                            {" "}
                            (n = {modelStats.trend.high_confidence_count})
                          </span>
                        </>
                      )}
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Precision UP</span>
                    <strong>
                      {formatPctFraction(modelStats.trend.precision_up)}
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Recall UP</span>
                    <strong>
                      {formatPctFraction(modelStats.trend.recall_up)}
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Confusion TP/TN/FP/FN</span>
                    <strong>
                      {modelStats.trend.confusion.tp}/
                      {modelStats.trend.confusion.tn}/
                      {modelStats.trend.confusion.fp}/
                      {modelStats.trend.confusion.fn}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="card chart-card stats-chart-card">
                <h2 className="card-title">q50 error over time</h2>
                <div className="legend stats-legend">
                  <span className="chip">
                    <i className="dot" style={{ background: "var(--chart-q50)" }} />
                    |actual − q50| / actual (%)
                  </span>
                  <span className="chip chip-muted">
                    Points outside [q10, q90] still appear on this error chart
                  </span>
                </div>
                <div ref={statsHostRef} className="chart-wrap stats-chart-wrap" />
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
