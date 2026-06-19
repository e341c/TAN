from __future__ import annotations

import copy
import os
import tempfile
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd

from technical_analysis_trainer.core.artifacts import ArtifactStore, resolve_path
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.models.trend.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
)
from technical_analysis_trainer.runtime.binance import fetch_last_candles_sync

TAIL_ROWS = int(os.environ.get("REALTIME_CSV_TAIL", "1000"))
BINANCE_LIMIT = int(os.environ.get("REALTIME_BINANCE_LIMIT", "400"))


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
    for c in base.columns:
        if c not in live.columns:
            live[c] = np.nan
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


def build_trend_x_window(
    cfg: dict[str, Any],
    store: ArtifactStore,
    run_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cross_cfg = cfg.get("cross_market")
    mtf_cfg = cfg.get("mtf")

    btc_live = fetch_last_candles_sync(str(cfg.get("symbol", "BTCUSDT")), "1h", BINANCE_LIMIT)
    eth_live = fetch_last_candles_sync("ETHUSDT", "1h", BINANCE_LIMIT)
    sol_live = fetch_last_candles_sync("SOLUSDT", "1h", BINANCE_LIMIT)

    btc_merged = _overlay_csv_tail_with_live(str(cfg["data_path"]), btc_live, TAIL_ROWS)
    path_overrides: dict[str, str] = {}
    temp_paths: list[str] = []

    if (cross_cfg or {}).get("enabled", False):
        bench = (cross_cfg or {}).get("benchmarks") or {}
        eth_entry = bench.get("eth") or {}
        sol_entry = bench.get("sol") or {}
        eth_csv = (eth_entry.get("path") if isinstance(eth_entry, dict) else None) or ""
        sol_csv = (sol_entry.get("path") if isinstance(sol_entry, dict) else None) or ""
        if eth_csv.strip():
            eth_merged = _overlay_csv_tail_with_live(eth_csv, eth_live, TAIL_ROWS)
            p = _write_temp_csv(eth_merged)
            temp_paths.append(p)
            path_overrides["eth"] = p
        if sol_csv.strip():
            sol_merged = _overlay_csv_tail_with_live(sol_csv, sol_live, TAIL_ROWS)
            p = _write_temp_csv(sol_merged)
            temp_paths.append(p)
            path_overrides["sol"] = p

    btc_temp = _write_temp_csv(btc_merged)
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
        )
        df_feat, _ = backfill_missing_features_with_scaler_means(df_feat, features, scaler)

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

        i = len(X_scaled) - 1
        if i < seq_len:
            raise ValueError(f"Not enough rows for window: i={i}, seq_len={seq_len}")

        x_input = X_scaled[i - seq_len : i].reshape(seq_len, len(features))
        meta = {
            "trend_run_id": run.run_id,
            "onnx_path": str(run.onnx_path),
            "seq_len": seq_len,
            "n_features": len(features),
            "last_ts": df_feat["ts"].iloc[i] if "ts" in df_feat.columns else None,
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


def run_trend_onnx(X_window: np.ndarray, onnx_path: str) -> dict[str, Any]:
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    x = X_window[np.newaxis, ...].astype(np.float32)
    logits = np.asarray(ort_session.run(None, {input_name: x})[0][0], dtype=np.float64).ravel()
    probs = softmax_np(logits, temperature=1.0)
    class_names = ["DOWN", "UP"] if len(probs) == 2 else [f"C{i}" for i in range(len(probs))]
    pred_i = int(np.argmax(probs))
    display = {class_names[i]: f"{probs[i] * 100:.2f}%" for i in range(len(probs))}
    return {
        "predicted_trend": class_names[pred_i],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))},
        "logits": logits.tolist(),
        **display,
    }
