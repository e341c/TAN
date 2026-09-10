from __future__ import annotations

import copy
import os
import tempfile
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from technical_analysis_trainer.core.artifacts import ArtifactStore, resolve_path
from technical_analysis_trainer.core.calibration import Calibrator
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.validation import dead_features
from technical_analysis_trainer.models.trend.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
)
from technical_analysis_trainer.models.trend.target import (
    ANCHOR_LAST_OBSERVED,
    default_spec,
    spec_anchor,
    spec_from_bundle,
)
from technical_analysis_trainer.runtime.binance import fetch_last_candles_sync

TAIL_ROWS = int(os.environ.get("REALTIME_CSV_TAIL", "1000"))
# 1000 is Binance's per-request maximum and is deliberately taken in full. The
# CSVs under data/predict/ are not on any refresh schedule (only the macro feeds
# are, see deploy/docker/crontab), so everything after their last row comes from
# this fetch. Feature warm-up now runs ~336 bars (168-bar rolling level + its
# 168-bar de-trending reference) before the 24-bar window; at the previous 400
# the reference window reached back across the CSV/live seam into months-old
# rows and normalised today's flow against them.
BINANCE_LIMIT = int(os.environ.get("REALTIME_BINANCE_LIMIT", "1000"))


def _ensure_schema_for_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Live Binance payload often does not include `trades` and `funding_rate`.
    # After stitching dummy/base CSV with live rows we may have NaNs; trend features
    # (e.g. avg_trade_size = volume / trades) will then drop rows (via dropna),
    # producing empty samples for scaler.transform.
    if "trades" not in out.columns:
        out["trades"] = 1.0
    out["trades"] = out["trades"].fillna(1.0)
    if "funding_rate" not in out.columns:
        out["funding_rate"] = 0.0
    out["funding_rate"] = out["funding_rate"].fillna(0.0)
    return out


def backfill_missing_features_with_scaler_means(
    df_feat: pd.DataFrame,
    features: list[str],
    scaler: object,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in features if c not in df_feat.columns]
    if not missing:
        return df_feat, []
    out = df_feat.copy()
    mean = getattr(scaler, "mean_", None)
    feature_index = {f: i for i, f in enumerate(features)}
    for col in missing:
        i = feature_index.get(col)
        if i is not None and mean is not None and i < len(mean):
            fill_value = float(mean[i])
        else:
            fill_value = 0.0
        out[col] = fill_value
    return out, missing


def _overlay_csv_tail_with_live(csv_path: str, live: pd.DataFrame, tail_n: int) -> pd.DataFrame:
    path = str(resolve_path(csv_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Predict CSV not found: {path}")
    base = load_dataset(path)
    if len(base) > tail_n:
        base = base.tail(tail_n).reset_index(drop=True)
    base = base.copy()
    base["ts"] = pd.to_datetime(base["ts"], utc=True, errors="coerce")
    live = live.copy()
    live["ts"] = pd.to_datetime(live["ts"], utc=True, errors="coerce")
    live = live.dropna(subset=["ts"])
    # Union of columns, not a projection onto the CSV's.
    # Projecting onto `base.columns` silently drops any column the live feed has
    # and the on-disk CSV does not — which is every newly-added input until the
    # CSV is regenerated. `taker_buy_base` hit exactly this: fetched live, absent
    # from the older predict CSVs, dropped here, then mean-filled downstream.
    for c in base.columns:
        if c not in live.columns:
            live[c] = np.nan
    for c in live.columns:
        if c not in base.columns:
            base[c] = np.nan
    live = live[base.columns]
    merged = pd.concat([base, live], ignore_index=True)
    return merged.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)


def _write_temp_csv(df: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(prefix="rt_", suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def _deep_cross_cfg_with_paths(cross_cfg: dict | None, path_overrides: dict[str, str]) -> dict | None:
    if not cross_cfg:
        return cross_cfg
    out = copy.deepcopy(cross_cfg)
    bench = dict(out.get("benchmarks") or {})
    for key, p in path_overrides.items():
        ent = bench.get(key)
        if isinstance(ent, dict):
            bench[key] = {**ent, "path": p}
        else:
            bench[key] = {"path": p}
    out["benchmarks"] = bench
    return out


def load_target_spec(store: ArtifactStore, run_id: str | None = None) -> dict[str, Any]:
    """Read a run's target semantics from its saved bundle."""
    import pickle

    try:
        run = store.get_run(run_id)
        with open(run.scaler_path, "rb") as f:
            return spec_from_bundle(pickle.load(f))
    except Exception:  # noqa: BLE001 - a missing spec must not break inference
        return default_spec()


def load_calibrator(store: ArtifactStore, run_id: str | None = None) -> Calibrator:
    """
    Read a run's probability calibration, or the identity for artifacts without one.

    Identity is the correct fallback: a model exported before calibration existed
    was never corrected, and silently applying someone else's coefficients would
    be worse than applying none.
    """
    import pickle

    try:
        run = store.get_run(run_id)
        with open(run.scaler_path, "rb") as f:
            bundle = pickle.load(f)
        return Calibrator.from_dict((bundle or {}).get("calibrator"))
    except Exception:  # noqa: BLE001 - a missing calibrator must not break inference
        return Calibrator.identity()


def trailing_prior_from_candles(
    df: pd.DataFrame, spec: dict[str, Any], window: int = 720
) -> np.ndarray | None:
    """
    Base rate of recently RESOLVED barrier outcomes, per bar of `df`.

    This is the serving-side counterpart of the prior used in training: the
    trend model's decision deficit was almost entirely a base-rate mismatch
    (calibrated where UP ran at 51.9%, served where it ran at 60.7%), and
    correcting it needs to know the rate prevailing now.

    The labels are recomputed here from the same `triple_barrier_labels` that
    made the training targets, and each is credited to the bar it RESOLVES on,
    not the bar it was anchored to — so the value at bar t uses only outcomes a
    live system would already have seen at t. Returns None when the frame is too
    short for the volatility window to produce anything.
    """
    from technical_analysis_trainer.core.calibration import trailing_prior
    from technical_analysis_trainer.models.trend.features import triple_barrier_labels

    if spec.get("mode") != "triple_barrier" or "close" not in df.columns:
        return None
    vol_window = int(spec.get("vol_window", 72))
    if len(df) < vol_window + 2:
        return None
    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    sigma = lr.rolling(vol_window, min_periods=vol_window).std().to_numpy()
    labels, spans, _ = triple_barrier_labels(
        close.to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), sigma,
        upper_mult=float(spec.get("upper_mult", 2.5)),
        lower_mult=float(spec.get("lower_mult", 2.5)),
        max_holding=int(spec.get("max_holding", 24)),
    )
    return trailing_prior(labels, np.arange(len(labels)) + np.maximum(spans, 1), len(df), window)


def _prior_at_anchor(df_feat: pd.DataFrame, spec: dict[str, Any], anchor_idx: int) -> float | None:
    """Trailing resolved base rate at the anchor bar, or None if unavailable."""
    series = trailing_prior_from_candles(df_feat, spec)
    if series is None or anchor_idx >= len(series):
        return None
    return float(series[anchor_idx])


def build_trend_x_window(
    cfg: dict[str, Any],
    store: ArtifactStore,
    run_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Live path: fetch current candles from Binance, stitch onto the CSV tail."""
    cross_cfg = cfg.get("cross_market")

    btc_live = fetch_last_candles_sync(str(cfg.get("symbol", "BTCUSDT")), "1h", BINANCE_LIMIT)
    eth_live = fetch_last_candles_sync("ETHUSDT", "1h", BINANCE_LIMIT)
    sol_live = fetch_last_candles_sync("SOLUSDT", "1h", BINANCE_LIMIT)

    btc_merged = _overlay_csv_tail_with_live(str(cfg["data_path"]), btc_live, TAIL_ROWS)

    eth_merged = sol_merged = None
    if (cross_cfg or {}).get("enabled", False):
        bench = (cross_cfg or {}).get("benchmarks") or {}
        eth_entry = bench.get("eth") or {}
        sol_entry = bench.get("sol") or {}
        eth_csv = (eth_entry.get("path") if isinstance(eth_entry, dict) else None) or ""
        sol_csv = (sol_entry.get("path") if isinstance(sol_entry, dict) else None) or ""
        if eth_csv.strip():
            eth_merged = _overlay_csv_tail_with_live(eth_csv, eth_live, TAIL_ROWS)
        if sol_csv.strip():
            sol_merged = _overlay_csv_tail_with_live(sol_csv, sol_live, TAIL_ROWS)

    # Live frames end on the still-forming candle, so the window must stop one
    # bar earlier to stay on closed data.
    return build_trend_x_window_from_frames(
        cfg, store, run_id=run_id, btc=btc_merged, eth=eth_merged, sol=sol_merged,
        last_row_is_closed=False,
    )


def build_trend_x_window_from_frames(
    cfg: dict[str, Any],
    store: ArtifactStore,
    run_id: str | None = None,
    *,
    btc: pd.DataFrame,
    eth: pd.DataFrame | None = None,
    sol: pd.DataFrame | None = None,
    last_row_is_closed: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build the model window from explicitly supplied frames — no network calls.

    `last_row_is_closed` selects the anchor:

      * False (live): the final row is the forming candle, so the window ends on
        the row before it.
      * True (backfill): the caller trimmed the frames so the final row IS the
        closed bar being predicted from, and it belongs in the window.

    Both produce the same window *content* for the same anchor bar, which is what
    makes a backfilled prediction comparable to the live one it stands in for.
    """
    cross_cfg = cfg.get("cross_market")
    mtf_cfg = cfg.get("mtf")

    path_overrides: dict[str, str] = {}
    temp_paths: list[str] = []

    if (cross_cfg or {}).get("enabled", False):
        if eth is not None and len(eth):
            p = _write_temp_csv(eth)
            temp_paths.append(p)
            path_overrides["eth"] = p
        if sol is not None and len(sol):
            p = _write_temp_csv(sol)
            temp_paths.append(p)
            path_overrides["sol"] = p

    btc_temp = _write_temp_csv(btc)
    temp_paths.append(btc_temp)
    cross_eff = _deep_cross_cfg_with_paths(cross_cfg, path_overrides)

    try:
        d = load_dataset(btc_temp)
        if len(d) == 0:
            raise ValueError("Empty BTC series after stitch")
        # Important: don't attach cross/MTF features here.
        # `add_features()` will run `build_features_df()` with the same cfg and is
        # the single place that should decide whether features are present and
        # safely drop NaN rows. Doing it twice can leave columns present but
        # fully-NaN (especially after stitching with live data), which later
        # causes `dropna()` to drop the whole frame.
        d = _ensure_schema_for_features(d)

        scaler, features, scale_lo, scale_hi, run = store.load_trend_scaler_bundle(run_id)
        # Loaded before the feature build, not after: the label-autocorrelation
        # columns need the artifact's own barrier geometry to be reconstructed.
        spec = load_target_spec(store, run.run_id)
        window_cfg = cfg.get("window") or {}
        seq_len = int(window_cfg.get("long", window_cfg.get("mid", 24)))
        threshold = float(cfg["threshold"])
        horizon = int(cfg.get("horizon", 1))

        *_, df_feat = add_features(
            d,
            seq_len=seq_len,
            threshold=threshold,
            horizon=horizon,
            scaler=scaler,
            mtf_cfg=mtf_cfg,
            cross_market_cfg=cross_eff,
            return_df_feat=True,
            clip_low=scale_lo,
            clip_high=scale_hi,
            # Build every available column and let the artifact pick: its
            # scaler bundle is the only authority on what it was fitted with.
            order_flow=True,
            features_override=features,
            # Built unconditionally at inference: `features_override` decides
            # what actually reaches the model, so building a column the artifact
            # does not use is free, while failing to build one it does need is not.
            target_mode=str(spec.get("mode", "next_bar")),
            tb_cfg=spec,
            label_autocorr=True,
        )
        df_feat, filled = backfill_missing_features_with_scaler_means(df_feat, features, scaler)
        if filled:
            # Never silent. A feature the model trained on but the live frame
            # cannot supply gets the training mean, which keeps inference running
            # while quietly moving the input off the distribution the model was
            # fitted to. That is how `trades` and the ETH cross-market block both
            # shipped broken; the fix is upstream (supply the column), not here.
            print(f"  ⚠ trend inference: {len(filled)} feature(s) missing from the live "
                  f"frame, filled with training means: {', '.join(filled)}")

        min_rows = seq_len + max(horizon, 1) + 1
        if len(df_feat) < min_rows:
            raise ValueError(
                f"After features {len(df_feat)} rows, need >= {min_rows} "
                f"(seq_len={seq_len}, horizon={horizon})"
            )

        X_scaled = np.clip(
            scaler.transform(df_feat[features].values.astype(np.float32)),
            scale_lo,
            scale_hi,
        ).astype(np.float32)

        # A feature that was constant while the scaler was fitted has scale_ = 1.0,
        # so `transform` leaves its live value essentially raw — the model would be
        # handed a number it never saw during training (a correlation of 0.9 where
        # it only ever saw 0.0). Pin those columns to the training constant, which
        # is what the model was actually fitted against.
        dead = dead_features(scaler, features)
        if dead:
            X_scaled[:, [features.index(f) for f in dead]] = 0.0

        # `i` is one past the last bar that belongs in the window. The window
        # itself is the same slice under both anchor conventions — the last
        # `seq_len` closed bars — so nothing here changes when the convention
        # does. What changes is where the barriers sit relative to it, which is
        # why that is reported explicitly below rather than left to the caller.
        i = len(X_scaled) if last_row_is_closed else len(X_scaled) - 1
        if i < seq_len:
            raise ValueError(f"Not enough rows for window: i={i}, seq_len={seq_len}")

        x_input = X_scaled[i - seq_len : i].reshape(seq_len, len(features))
        anchor_idx = i - 1  # last bar actually inside the window
        ts_col = df_feat["ts"] if "ts" in df_feat.columns else None
        last_ts = ts_col.iloc[anchor_idx] if ts_col is not None else None
        # Under `last_observed` the barriers start at the close of the last bar
        # in the window — a price that exists now. Under the legacy `next_close`
        # they start at the following bar, which has not closed yet, so anything
        # placing real levels has to substitute the current price and is
        # measuring something slightly different from the label.
        if spec_anchor(spec) == ANCHOR_LAST_OBSERVED:
            barrier_anchor_ts = last_ts
            barrier_anchor_close = float(df_feat["close"].iloc[anchor_idx])
        else:
            barrier_anchor_ts = None  # the bar after `last_ts`; does not exist yet
            barrier_anchor_close = None
        meta = {
            "trend_run_id": run.run_id,
            "onnx_path": str(run.onnx_path),
            "seq_len": seq_len,
            "n_features": len(features),
            "last_ts": last_ts,
            "barrier_anchor_ts": barrier_anchor_ts,
            "barrier_anchor_close": barrier_anchor_close,
            # Taken from the artifact, not from cfg: the deployed model defines
            # what its own output means.
            "target_spec": spec,
            "calibrator": load_calibrator(store, run.run_id),
            # Base rate of barrier outcomes that had already resolved by the
            # anchor bar. None when the artifact does not use the correction or
            # the frame is too short; run_trend_onnx treats that as "no shift".
            "prior_now": _prior_at_anchor(df_feat, spec, anchor_idx),
        }
        return x_input, meta
    finally:
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (logits - np.max(logits)) / max(temperature, 1e-9)
    e = np.exp(z)
    return e / np.sum(e)


def run_trend_onnx(
    X_window: np.ndarray,
    onnx_path: str,
    target_spec: dict[str, Any] | None = None,
    calibrator: Calibrator | None = None,
    prior_now: float | None = None,
) -> dict[str, Any]:
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    x = X_window[np.newaxis, ...].astype(np.float32)
    logits = np.asarray(ort_session.run(None, {input_name: x})[0][0], dtype=np.float64).ravel()
    raw = softmax_np(logits, temperature=1.0)
    class_names = ["DOWN", "UP"] if len(raw) == 2 else [f"C{i}" for i in range(len(raw))]

    # Calibration and the decision cut both come from the artifact. Binary only:
    # the Platt fit is on p(UP), and a multi-class head has no such pair.
    cal = calibrator or Calibrator.identity()
    if len(raw) == 2:
        p_up = float(cal.shift(np.array([raw[1]]), prior_now)[0])
        probs = np.array([1.0 - p_up, p_up])
        # Argmax is a 0.5 cut. After calibration the accuracy-optimal cut is a
        # fitted number, not 0.5 — the base rate the model was trained on and the
        # one it is served in do not have to agree. `cut()` returns 0.5 when the
        # prior shift already re-centred the probabilities, so the correction is
        # never applied twice.
        cut = cal.cut(prior_now)
        pred_i = int(p_up >= cut)
    else:
        probs, pred_i, cut = raw, int(np.argmax(raw)), 0.5

    display = {class_names[i]: f"{probs[i] * 100:.2f}%" for i in range(len(probs))}
    spec = target_spec or default_spec()
    return {
        "predicted_trend": class_names[pred_i],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))},
        # Kept alongside so a calibration change is auditable after the fact.
        "probabilities_raw": {class_names[i]: float(raw[i]) for i in range(len(raw))},
        "decision_threshold": float(cut),
        "calibrated": cal.n > 0,
        "prior_now": None if prior_now is None else float(prior_now),
        "prior_ref": float(cal.prior_ref),
        "prior_shift_applied": bool(cal.prior_shift and prior_now is not None),
        "logits": logits.tolist(),
        # What this label actually means. Consumers must read this rather than
        # assuming a 1-bar horizon — the same "UP"/"DOWN" covers both targets.
        "target": spec,
        "horizon_hours": spec.get("horizon_hours"),
        **display,
    }
