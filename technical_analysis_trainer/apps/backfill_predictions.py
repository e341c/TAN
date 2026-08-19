"""
Fill prediction gaps for the candles the web app displays.

The dashboard overlays stored forecasts onto the candle chart, so any candle
without a row in `quantile_predictions` / `trend_predictions` shows up as a hole
— typically because the scheduler was down, the container restarted, or the
range was never covered at all.

This module regenerates those missing forecasts **as they would have been made
at the time**, which is the whole difficulty: a naive backfill that feeds the
model the full history produces forecasts that quietly saw their own future, and
they then look far too good in the accuracy panel.

The guarantee here is structural rather than trusted:

  * every input frame (BTC and each cross-market benchmark) is trimmed to
    `ts <= T` before any feature is computed, so future rows do not exist at all
    during the call — no reliance on rolling/merge_asof being backward-looking;
  * the trim is asserted, not assumed;
  * the window is anchored so its *content* matches what the live path would
    have used at T (see `last_row_is_closed` in trend/predict.py and `drop_last`
    in quantile/predict.py). Getting this wrong shifts every backfilled
    prediction one bar against the live ones and makes the two incomparable.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from technical_analysis_trainer.assembly.orchestrator import (
    _run_id_for,
    load_assembly_config,
)
from technical_analysis_trainer.core.artifacts import ArtifactStore
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.quantile.predict import predict_from_candles
from technical_analysis_trainer.models.trend.plugin import load_trend_config
from technical_analysis_trainer.models.trend.predict import (
    build_trend_x_window_from_frames,
    run_trend_onnx,
)
from technical_analysis_trainer.runtime.binance import (
    fetch_klines_range_sync,
    interval_to_timedelta,
)
from technical_analysis_trainer.storage.prediction_store import (
    QuantilePrediction,
    TrendPrediction,
    get_database,
    save_prediction_run,
)

logger = logging.getLogger(__name__)

# History kept before each anchor bar. Matches REALTIME_CSV_TAIL so backfilled
# features see the same amount of history the live path does — EWM-based
# indicators (MACD) never fully forget, so a shorter window would drift.
DEFAULT_LOOKBACK = int(os.environ.get("BACKFILL_LOOKBACK", "1000"))
# Minimum bars before an anchor for the slowest feature to be defined:
# a 168-bar rolling level plus the 168-bar reference window that de-trends it
# (~350 in practice), then seq_len and margin. Was 260, which silently produced
# a shorter effective history than the live path once de-trending was added.
MIN_HISTORY = 420


@dataclass
class BackfillResult:
    scanned: int = 0
    already_present: int = 0
    filled: int = 0
    skipped_no_history: int = 0
    errors: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    error_samples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "already_present": self.already_present,
            "filled": self.filled,
            "skipped_no_history": self.skipped_no_history,
            "errors": self.errors,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "error_samples": self.error_samples[:5],
        }


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _trim_to(df: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    """Drop every row after `cutoff`. This is the leak barrier."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out[out["ts"] <= cutoff].sort_values("ts").reset_index(drop=True)
    return out


def existing_prediction_ts(
    symbol: str, timeframe: str, lo: datetime, hi: datetime
) -> set[pd.Timestamp]:
    """candle_ts values that already have BOTH a quantile and a trend row."""
    get_database()
    q = {
        _utc(r.candle_ts)
        for r in QuantilePrediction.select(QuantilePrediction.candle_ts).where(
            (QuantilePrediction.symbol == symbol)
            & (QuantilePrediction.timeframe == timeframe)
            & (QuantilePrediction.candle_ts >= lo)
            & (QuantilePrediction.candle_ts <= hi)
        )
    }
    t = {
        _utc(r.candle_ts)
        for r in TrendPrediction.select(TrendPrediction.candle_ts).where(
            (TrendPrediction.symbol == symbol)
            & (TrendPrediction.timeframe == timeframe)
            & (TrendPrediction.candle_ts >= lo)
            & (TrendPrediction.candle_ts <= hi)
        )
    }
    # A candle with only one of the two is a gap: the stats join needs both.
    return q & t


def predict_as_of(
    anchor_ts: pd.Timestamp,
    *,
    btc: pd.DataFrame,
    eth: pd.DataFrame | None,
    sol: pd.DataFrame | None,
    quantile_store: ArtifactStore,
    trend_store: ArtifactStore,
    quantile_run: str | None,
    trend_run: str | None,
    trend_cfg: dict,
) -> tuple[dict, dict, dict]:
    """
    Produce the forecast that would have been made at `anchor_ts`.

    Every frame is trimmed to <= anchor_ts first, and the trim is verified before
    a single feature is computed.
    """
    btc_t = _trim_to(btc, anchor_ts)
    eth_t = _trim_to(eth, anchor_ts)
    sol_t = _trim_to(sol, anchor_ts)

    if btc_t is None or len(btc_t) == 0:
        raise ValueError(f"no BTC history at or before {anchor_ts}")
    actual_last = _utc(btc_t["ts"].iloc[-1])
    if actual_last != anchor_ts:
        raise ValueError(f"anchor mismatch: expected {anchor_ts}, frame ends {actual_last}")
    for name, frame in (("eth", eth_t), ("sol", sol_t)):
        if frame is not None and len(frame) and _utc(frame["ts"].iloc[-1]) > anchor_ts:
            raise ValueError(f"{name} frame leaks past {anchor_ts}")
    if len(btc_t) < MIN_HISTORY:
        raise LookupError(f"only {len(btc_t)} bars before {anchor_ts}, need {MIN_HISTORY}")

    # drop_last=False: the frame's final row IS the closed anchor bar.
    q_payload, q_run = predict_from_candles(
        btc_t, quantile_store, run_id=quantile_run, drop_last=False
    )

    x_window, t_meta = build_trend_x_window_from_frames(
        trend_cfg,
        trend_store,
        run_id=trend_run,
        btc=btc_t,
        eth=eth_t,
        sol=sol_t,
        last_row_is_closed=True,
    )
    t_payload = run_trend_onnx(
        x_window, str(t_meta["onnx_path"]), target_spec=t_meta.get("target_spec")
    )

    q_payload = dict(q_payload)
    q_payload["_run_id"] = q_run.run_id
    return q_payload, t_payload, t_meta


def backfill_sync(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    bars: int = 200,
    lookback: int = DEFAULT_LOOKBACK,
    *,
    dry_run: bool = False,
    force: bool = False,
    max_fill: int | None = None,
    log_every: int = 25,
) -> BackfillResult:
    """Scan the displayed candle range and fill every candle missing a forecast."""
    result = BackfillResult()
    if not os.environ.get("DATABASE_URL", "").strip():
        raise RuntimeError("DATABASE_URL is not set; backfill needs PostgreSQL")

    cfg = load_assembly_config()
    trend_cfg = load_trend_config(cfg.get("trend_config_path"))
    quantile_run = _run_id_for("quantile", cfg)
    trend_run = _run_id_for("trend", cfg)
    quantile_store = ArtifactStore(package_models_dir("quantile"))
    trend_store = ArtifactStore(package_models_dir("trend"))

    bar = interval_to_timedelta(interval)
    now = pd.Timestamp.now(tz="UTC")
    start = now - (bars + lookback + 5) * bar

    logger.info("fetching %s %s candles from %s", symbol, interval, start)
    btc = fetch_klines_range_sync(symbol, interval, start, now)

    cross = trend_cfg.get("cross_market") or {}
    eth = sol = None
    if cross.get("enabled", False):
        bench = cross.get("benchmarks") or {}
        if ((bench.get("eth") or {}).get("path") or "").strip():
            eth = fetch_klines_range_sync("ETHUSDT", interval, start, now)
        if ((bench.get("sol") or {}).get("path") or "").strip():
            sol = fetch_klines_range_sync("SOLUSDT", interval, start, now)

    btc["ts"] = pd.to_datetime(btc["ts"], utc=True)
    # The final Binance bar is still forming — it has no settled close, so it is
    # not a valid anchor and never gets a backfilled prediction.
    closed = btc.iloc[:-1].reset_index(drop=True)
    if len(closed) == 0:
        return result

    display = closed.tail(bars).reset_index(drop=True)
    lo = _utc(display["ts"].iloc[0]).to_pydatetime()
    hi = _utc(display["ts"].iloc[-1]).to_pydatetime()
    result.first_ts, result.last_ts = str(lo), str(hi)

    # Rows are keyed by the predicted bar (anchor + one bar), so an anchor is
    # "already covered" when anchor+bar is present — comparing anchors directly
    # against stored keys would mark every candle as missing and refill forever.
    have = (
        set()
        if force
        else existing_prediction_ts(symbol, interval, lo, hi + bar)
    )
    anchors = [_utc(x) for x in display["ts"]]
    targets = [t for t in anchors if (t + bar) not in have]
    result.scanned = len(display)
    result.already_present = len(display) - len(targets)

    if max_fill is not None:
        targets = targets[:max_fill]

    logger.info(
        "range %s .. %s | %d candles, %d already have forecasts, %d to fill%s",
        lo, hi, result.scanned, result.already_present, len(targets),
        " (dry run)" if dry_run else "",
    )

    for n, anchor in enumerate(targets, 1):
        try:
            q_payload, t_payload, t_meta = predict_as_of(
                anchor,
                btc=btc, eth=eth, sol=sol,
                quantile_store=quantile_store, trend_store=trend_store,
                quantile_run=quantile_run, trend_run=trend_run, trend_cfg=trend_cfg,
            )
        except LookupError as e:
            result.skipped_no_history += 1
            logger.debug("skip %s: %s", anchor, e)
            continue
        except Exception as e:  # noqa: BLE001 - one bad bar must not stop the run
            result.errors += 1
            msg = f"{anchor}: {type(e).__name__}: {e}"
            if len(result.error_samples) < 5:
                result.error_samples.append(msg)
            logger.warning("backfill failed for %s", msg)
            continue

        if not dry_run:
            try:
                # Store candle_ts on the *predicted* bar, matching the live path
                # (which passes the frame's last row, i.e. the forming candle)
                # and every row already in the table. The chart filters markers
                # by candle_unix, so storing the anchor here instead would draw
                # backfilled forecasts one bar to the left of live ones.
                save_prediction_run(
                    q_payload,
                    t_payload,
                    t_meta,
                    symbol=symbol,
                    timeframe=interval,
                    candle_ts=_utc(q_payload["ts"]),
                    quantile_run_id=str(q_payload.get("_run_id") or ""),
                )
            except Exception as e:  # noqa: BLE001
                result.errors += 1
                if len(result.error_samples) < 5:
                    result.error_samples.append(f"save {anchor}: {e}")
                logger.warning("save failed for %s: %s", anchor, e)
                continue

        result.filled += 1
        if log_every and n % log_every == 0:
            logger.info("  %d/%d filled (%s)", n, len(targets), anchor)

    logger.info("backfill done: %s", result.as_dict())
    return result


async def backfill(**kwargs) -> BackfillResult:
    import asyncio

    return await asyncio.to_thread(lambda: backfill_sync(**kwargs))


def main() -> None:
    p = argparse.ArgumentParser(description="Fill missing predictions for displayed candles")
    p.add_argument("--symbol", default=os.environ.get("PREDICTION_SYMBOL", "BTCUSDT"))
    p.add_argument("--interval", default=os.environ.get("PREDICTION_INTERVAL", "1h"))
    p.add_argument("--bars", type=int, default=200, help="candles the dashboard shows")
    p.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    p.add_argument("--dry-run", action="store_true", help="compute but do not write")
    p.add_argument("--force", action="store_true", help="recompute even where rows exist")
    p.add_argument("--max-fill", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    res = backfill_sync(
        symbol=a.symbol, interval=a.interval, bars=a.bars, lookback=a.lookback,
        dry_run=a.dry_run, force=a.force, max_fill=a.max_fill,
    )
    print(res.as_dict())


if __name__ == "__main__":
    main()
