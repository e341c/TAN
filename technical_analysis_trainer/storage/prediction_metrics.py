"""
Live model statistics: compare stored predictions against Binance actuals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from technical_analysis_trainer.core.metrics import (
    bootstrap_ci,
    pinball_loss,
    skill_score,
)
from technical_analysis_trainer.models.trend.target import default_spec, spec_from_json
from technical_analysis_trainer.runtime.binance import (
    fetch_klines_range,
    interval_to_timedelta,
)
from technical_analysis_trainer.storage.prediction_store import (
    QuantilePrediction,
    TrendPrediction,
    _iso_utc_dt,
    get_database,
)

DEFAULT_TREND_THRESHOLD = 0.001


@dataclass(frozen=True)
class PredictionPair:
    candle_ts: datetime
    target_ts: datetime
    last_close: float | None
    q10: float
    q50: float
    q90: float
    predicted_trend: str
    prob_up: float | None
    prob_down: float | None
    target_spec: dict[str, Any] | None = None
    trend_run_id: str | None = None


def _to_utc_timestamp(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _dt_to_unix_seconds(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def build_close_lookup(candles: pd.DataFrame) -> dict[pd.Timestamp, float]:
    lookup: dict[pd.Timestamp, float] = {}
    for _, row in candles.iterrows():
        lookup[_to_utc_timestamp(row["ts"])] = float(row["close"])
    return lookup


def list_mature_prediction_pairs(
    symbol: str,
    timeframe: str,
    limit: int = 200,
) -> list[PredictionPair]:
    """JOIN quantile + trend rows where the target candle has closed."""
    if not os.environ.get("DATABASE_URL", "").strip():
        return []

    get_database()
    bar_delta = interval_to_timedelta(timeframe)
    mature_before = (pd.Timestamp.now(tz="UTC") - bar_delta).to_pydatetime()

    query = (
        QuantilePrediction.select(QuantilePrediction, TrendPrediction)
        .join(
            TrendPrediction,
            on=(
                (QuantilePrediction.symbol == TrendPrediction.symbol)
                & (QuantilePrediction.timeframe == TrendPrediction.timeframe)
                & (QuantilePrediction.target_ts == TrendPrediction.target_ts)
            ),
        )
        .where(
            (QuantilePrediction.symbol == symbol)
            & (QuantilePrediction.timeframe == timeframe)
            & (QuantilePrediction.target_ts <= mature_before)
        )
        .order_by(QuantilePrediction.target_ts.desc())
        .limit(limit)
    )

    pairs: list[PredictionPair] = []
    for q_row in query:
        t_row = q_row.trendprediction
        pairs.append(
            PredictionPair(
                candle_ts=q_row.candle_ts,
                target_ts=q_row.target_ts,
                last_close=float(q_row.last_close) if q_row.last_close is not None else None,
                q10=float(q_row.q10),
                q50=float(q_row.q50),
                q90=float(q_row.q90),
                predicted_trend=str(t_row.predicted_trend),
                prob_up=float(t_row.prob_up) if t_row.prob_up is not None else None,
                prob_down=float(t_row.prob_down) if t_row.prob_down is not None else None,
                # None here (rather than a default) so the scorer can fall back to
                # the artifact before assuming next_bar.
                target_spec=(
                    spec_from_json(t_row.target_spec)
                    if getattr(t_row, "target_spec", None)
                    else None
                ),
                trend_run_id=str(t_row.model_run_id) if t_row.model_run_id else None,
            )
        )
    return pairs


def _empty_stats(
    symbol: str,
    timeframe: str,
    *,
    trend_threshold: float,
) -> dict[str, Any]:
    return {
        "meta": {
            "symbol": symbol,
            "interval": timeframe,
            "evaluated_count": 0,
            "neutral_skipped": 0,
            "period_start": None,
            "period_end": None,
            "trend_threshold": trend_threshold,
            "trend_target_modes": {},
            "trend_run_ids": {},
            "mixed_targets": False,
        },
        "quantile": {
            "coverage_rate": None,
            "coverage_ci95": None,
            "coverage_nominal": 0.8,
            "breach_below_rate": None,
            "breach_above_rate": None,
            "mae_q50": None,
            "mape_q50": None,
            "mae_naive": None,
            "mape_naive": None,
            "mae_skill_vs_naive": None,
            "avg_breach_pct": None,
            "avg_interval_width_pct": None,
            "pinball": {"q0.1": None, "q0.5": None, "q0.9": None},
        },
        "trend": {
            "accuracy": None,
            "accuracy_ci95": None,
            "accuracy_high_confidence": None,
            "high_confidence_count": 0,
            "majority_baseline": None,
            "edge_vs_majority": None,
            "precision_up": None,
            "recall_up": None,
            "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        },
        "series": [],
    }


def _breach_pct(actual: float, q10: float, q90: float) -> float:
    if actual < q10:
        return (q10 - actual) / actual
    if actual > q90:
        return (actual - q90) / actual
    return 0.0


_SPEC_CACHE: dict[str, dict[str, Any]] = {}


def resolve_spec_for_run(run_id: str | None) -> dict[str, Any] | None:
    """
    Recover a row's target semantics from the model artifact that produced it.

    Rows written before `target_spec` existed have NULL, but their `model_run_id`
    still identifies the model — and the artifact records what it was trained to
    predict. Without this, barrier-model rows written by an older build fall back
    to next_bar and get graded against a question they never answered.
    """
    if not run_id:
        return None
    if run_id in _SPEC_CACHE:
        return _SPEC_CACHE[run_id]

    spec: dict[str, Any] | None = None
    try:
        import pickle

        from technical_analysis_trainer.core.artifacts import ArtifactStore
        from technical_analysis_trainer.models._paths import package_models_dir
        from technical_analysis_trainer.models.trend.target import spec_from_bundle

        run = ArtifactStore(package_models_dir("trend")).get_run(run_id)
        with open(run.scaler_path, "rb") as f:
            spec = spec_from_bundle(pickle.load(f))
    except Exception:  # noqa: BLE001 - a missing artifact must not break stats
        spec = None

    _SPEC_CACHE[run_id] = spec  # cache misses too; artifacts do not appear mid-run
    return spec


def _trend_label(ret: float, threshold: float) -> str | None:
    if abs(ret) <= threshold:
        return None
    return "UP" if ret > threshold else "DOWN"


def _barrier_label(
    candles: pd.DataFrame,
    start_ts: pd.Timestamp,
    upper_mult: float,
    lower_mult: float,
    max_holding: int,
    vol_window: int,
) -> str | None:
    """
    Realised triple-barrier outcome for a prediction made at `start_ts`.

    A barrier model answers "does price touch +k*sigma before -k*sigma within N
    bars?". Scoring it against the next bar's sign — as the next_bar path does —
    measures a completely different question and makes the accuracy meaningless.

    Returns "UP"/"DOWN", or None when the outcome is undetermined: not enough
    history for sigma, not enough forward bars, or both barriers touched inside
    the same candle (OHLC cannot order them).
    """
    ts = pd.to_datetime(candles["ts"], utc=True, errors="coerce")
    idx = ts.searchsorted(start_ts)
    if idx <= 0 or idx >= len(candles):
        return None

    close = candles["close"].astype(float).to_numpy()
    high = candles["high"].astype(float).to_numpy() if "high" in candles else close
    low = candles["low"].astype(float).to_numpy() if "low" in candles else close

    if idx < vol_window + 1 or idx + max_holding >= len(candles):
        return None

    lr = np.diff(np.log(close[idx - vol_window : idx + 1]))
    sigma = float(np.std(lr, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    up = close[idx] * np.exp(upper_mult * sigma)
    dn = close[idx] * np.exp(-lower_mult * sigma)

    for j in range(idx + 1, min(idx + max_holding, len(candles) - 1) + 1):
        hit_up, hit_dn = high[j] >= up, low[j] <= dn
        if hit_up and hit_dn:
            return None
        if hit_up:
            return "UP"
        if hit_dn:
            return "DOWN"
    return None  # vertical barrier: no directional resolution


def compute_stats_from_pairs(
    pairs: list[PredictionPair],
    candles: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    trend_target_mode: str = "next_bar",
    trend_tb_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not pairs:
        return _empty_stats(symbol, timeframe, trend_threshold=trend_threshold)
    tb = trend_tb_cfg or {}

    close_lookup = build_close_lookup(candles)
    bar_delta = interval_to_timedelta(timeframe)
    series: list[dict[str, Any]] = []

    actuals: list[float] = []
    q10s: list[float] = []
    q50s: list[float] = []
    q90s: list[float] = []
    base_closes: list[float] = []
    in_band_flags: list[bool] = []
    breach_pcts: list[float] = []
    mae_q50_vals: list[float] = []
    mape_q50_vals: list[float] = []
    below_flags: list[bool] = []
    above_flags: list[bool] = []
    trend_correct_flags: list[bool] = []
    actual_up_flags: list[bool] = []

    tp = tn = fp = fn = 0
    neutral_skipped = 0
    trend_evaluated = 0
    high_conf_correct = 0
    high_conf_total = 0
    mode_counts: dict[str, int] = {}
    run_counts: dict[str, int] = {}

    for pair in sorted(pairs, key=lambda p: p.target_ts):
        target_key = _to_utc_timestamp(pair.target_ts)
        candle_key = _to_utc_timestamp(pair.candle_ts)
        actual_close = close_lookup.get(target_key)
        if actual_close is None:
            continue

        base_close = pair.last_close
        if base_close is None:
            base_close = close_lookup.get(candle_key)
        if base_close is None or base_close <= 0:
            continue

        in_band = pair.q10 <= actual_close <= pair.q90
        breach = _breach_pct(actual_close, pair.q10, pair.q90)
        q50_err_frac = abs(actual_close - pair.q50) / actual_close

        actuals.append(actual_close)
        q10s.append(pair.q10)
        q50s.append(pair.q50)
        q90s.append(pair.q90)
        base_closes.append(base_close)
        in_band_flags.append(in_band)
        breach_pcts.append(breach)
        below_flags.append(actual_close < pair.q10)
        above_flags.append(actual_close > pair.q90)
        mae_q50_vals.append(abs(actual_close - pair.q50))
        mape_q50_vals.append(q50_err_frac)

        ret = (actual_close - base_close) / base_close

        # Score each row against the target ITS OWN model was trained on. Using
        # one globally-configured mode would grade the pre-2026-08-16 next_bar
        # rows as if they were barrier predictions, which is simply a different
        # question and yields a meaningless accuracy.
        # Precedence: the row's own spec > the producing model's artifact >
        # the caller-supplied mode > next_bar.
        spec = pair.target_spec or resolve_spec_for_run(pair.trend_run_id) or {}
        if not spec.get("mode"):
            spec = (
                {"mode": trend_target_mode, **tb}
                if trend_target_mode and trend_target_mode != "next_bar"
                else default_spec()
            )
        mode = str(spec.get("mode", "next_bar"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        if mode == "triple_barrier":
            # Enter at the bar whose close is `last_close` — the anchor the model
            # actually predicted from. Rows store candle_ts on the *predicted*
            # bar, so the anchor is one bar earlier; starting at candle_ts would
            # measure the barrier from an hour after the decision was made.
            actual_trend = _barrier_label(
                candles,
                candle_key - bar_delta,
                upper_mult=float(spec.get("upper_mult", 2.5)),
                lower_mult=float(spec.get("lower_mult", 2.5)),
                max_holding=int(spec.get("max_holding", 24)),
                vol_window=int(spec.get("vol_window", 72)),
            )
        else:
            actual_trend = _trend_label(ret, float(spec.get("threshold", trend_threshold)))
        predicted = pair.predicted_trend.upper()

        trend_correct: bool | None = None
        if actual_trend is None:
            neutral_skipped += 1
        else:
            trend_evaluated += 1
            pred_up = predicted == "UP"
            actual_up = actual_trend == "UP"
            trend_correct = predicted == actual_trend
            trend_correct_flags.append(trend_correct)
            actual_up_flags.append(actual_up)
            if pred_up and actual_up:
                tp += 1
            elif not pred_up and not actual_up:
                tn += 1
            elif pred_up and not actual_up:
                fp += 1
            else:
                fn += 1

            max_prob = max(
                pair.prob_up if pair.prob_up is not None else 0.0,
                pair.prob_down if pair.prob_down is not None else 0.0,
            )
            if max_prob >= 0.6:
                high_conf_total += 1
                if trend_correct:
                    high_conf_correct += 1

        point: dict[str, Any] = {
            "target_time": _iso_utc_dt(pair.target_ts),
            "target_unix": _dt_to_unix_seconds(pair.target_ts),
            "actual_close": actual_close,
            "q50_error_pct": q50_err_frac,
            "in_band": in_band,
            "breach_pct": breach,
            "predicted_trend": predicted,
            "actual_trend": actual_trend,
            "trend_correct": trend_correct,
            "target_mode": mode,
        }
        series.append(point)
        rid = pair.trend_run_id or "unknown"
        run_counts[rid] = run_counts.get(rid, 0) + 1

    evaluated_count = len(series)
    if evaluated_count == 0:
        return _empty_stats(symbol, timeframe, trend_threshold=trend_threshold)

    actual_arr = np.asarray(actuals, dtype=np.float64)
    q10_arr = np.asarray(q10s, dtype=np.float64)
    q50_arr = np.asarray(q50s, dtype=np.float64)
    q90_arr = np.asarray(q90s, dtype=np.float64)

    base_arr = np.asarray(base_closes, dtype=np.float64)

    breach_nonzero = [b for b in breach_pcts if b > 0]
    avg_breach = float(np.mean(breach_nonzero)) if breach_nonzero else 0.0

    accuracy = (tp + tn) / trend_evaluated if trend_evaluated else None
    precision_up = tp / (tp + fp) if (tp + fp) > 0 else None
    recall_up = tp / (tp + fn) if (tp + fn) > 0 else None
    accuracy_high_conf = (
        high_conf_correct / high_conf_total if high_conf_total > 0 else None
    )

    # ── Baselines ────────────────────────────────────────────────────────────
    # Persistence: "next close = last close". If q50 does not beat this, the
    # MAE/MAPE above is measuring the asset's volatility, not the model.
    naive_abs_err = np.abs(base_arr - actual_arr)
    mae_naive = float(np.mean(naive_abs_err))
    mape_naive = float(np.mean(naive_abs_err / actual_arr))
    mae_skill = skill_score(float(np.mean(mae_q50_vals)), mae_naive)

    # Majority class: always predict the more common direction.
    if actual_up_flags:
        up_rate = float(np.mean(actual_up_flags))
        majority = max(up_rate, 1.0 - up_rate)
        edge = accuracy - majority if accuracy is not None else None
    else:
        majority = edge = None

    acc_ci = (
        bootstrap_ci(np.asarray(trend_correct_flags, dtype=np.float64))
        if trend_correct_flags
        else None
    )
    cov_ci = bootstrap_ci(np.asarray(in_band_flags, dtype=np.float64))

    return {
        "meta": {
            "symbol": symbol,
            "interval": timeframe,
            "evaluated_count": evaluated_count,
            "neutral_skipped": neutral_skipped,
            "period_start": series[0]["target_time"],
            "period_end": series[-1]["target_time"],
            "trend_threshold": trend_threshold,
            # More than one entry here means the panel is averaging predictions
            # that answered different questions — read the numbers accordingly.
            "trend_target_modes": mode_counts,
            "trend_run_ids": run_counts,
            "mixed_targets": len(mode_counts) > 1,
        },
        "quantile": {
            "coverage_rate": float(np.mean(in_band_flags)),
            "coverage_ci95": list(cov_ci),
            "coverage_nominal": 0.8,
            "breach_below_rate": float(np.mean(below_flags)),
            "breach_above_rate": float(np.mean(above_flags)),
            "mae_q50": float(np.mean(mae_q50_vals)),
            "mape_q50": float(np.mean(mape_q50_vals)),
            "mae_naive": mae_naive,
            "mape_naive": mape_naive,
            "mae_skill_vs_naive": mae_skill,
            "avg_breach_pct": avg_breach,
            "avg_interval_width_pct": float(np.mean((q90_arr - q10_arr) / actual_arr)),
            "pinball": {
                "q0.1": pinball_loss(actual_arr, q10_arr, 0.1),
                "q0.5": pinball_loss(actual_arr, q50_arr, 0.5),
                "q0.9": pinball_loss(actual_arr, q90_arr, 0.9),
            },
        },
        "trend": {
            "accuracy": accuracy,
            "accuracy_ci95": list(acc_ci) if acc_ci else None,
            "accuracy_high_confidence": accuracy_high_conf,
            # The denominator behind accuracy_high_confidence. Without it a
            # "75%" from 8 samples reads the same as one from 800.
            "high_confidence_count": high_conf_total,
            "majority_baseline": majority,
            "edge_vs_majority": edge,
            "precision_up": precision_up,
            "recall_up": recall_up,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        },
        "series": series,
    }


async def compute_model_stats(
    symbol: str,
    timeframe: str,
    *,
    limit: int = 200,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    candles: pd.DataFrame | None = None,
    trend_target_mode: str = "next_bar",
    trend_tb_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pairs = list_mature_prediction_pairs(symbol, timeframe, limit=limit)
    if not pairs:
        return _empty_stats(symbol, timeframe, trend_threshold=trend_threshold)

    if candles is None:
        bar_delta = interval_to_timedelta(timeframe)
        tb = trend_tb_cfg or {}
        # Barrier labels need sigma history before the prediction and the full
        # holding window after it, or every outcome resolves to "undetermined".
        pad_before = int(tb.get("vol_window", 72)) + 5
        pad_after = int(tb.get("max_holding", 24)) + 5
        if trend_target_mode != "triple_barrier":
            pad_before, pad_after = 0, 1
        start = min(p.candle_ts for p in pairs) - pad_before * bar_delta.to_pytimedelta()
        end = max(p.target_ts for p in pairs) + pad_after * bar_delta.to_pytimedelta()
        candles = await fetch_klines_range(symbol, timeframe, start, end)

    return compute_stats_from_pairs(
        pairs,
        candles,
        symbol=symbol,
        timeframe=timeframe,
        trend_threshold=trend_threshold,
        trend_target_mode=trend_target_mode,
        trend_tb_cfg=trend_tb_cfg,
    )
