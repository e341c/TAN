"""
Real-time: quantile ONNX (30×7 фич, как раньше) + trend ONNX как в trend_model/test.py.
"""
from __future__ import annotations

import asyncio
import copy
import os
import pickle
import tempfile
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from binance.client import Client

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.features import compute_rsi
from technical_analysis_trainer.trend_model.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
)


def _ensure_schema_for_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trades" not in out.columns:
        out["trades"] = 1.0
    if "funding_rate" not in out.columns:
        out["funding_rate"] = 0.0
    return out


def latest_model_run(models_root: str) -> str:
    best_t: float = -1.0
    best_name: str | None = None
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"Нет каталога моделей: {models_root}")
    for name in os.listdir(models_root):
        sub = os.path.join(models_root, name)
        if not os.path.isdir(sub):
            continue
        scaler_glob = os.path.join(sub, f"scaler_{name}.pkl")
        if not os.path.isfile(scaler_glob):
            continue
        mtime = os.path.getmtime(scaler_glob)
        if mtime > best_t:
            best_t = mtime
            best_name = name
    if best_name is None:
        raise FileNotFoundError(f"Не найден ни один scaler_*.pkl в {models_root}")
    return best_name


def load_scaler_bundle(path: str) -> tuple[object, list[str], float, float]:
    with open(path, "rb") as f:
        scaler_data = pickle.load(f)
    if isinstance(scaler_data, dict):
        scaler = scaler_data["scaler"]
        features = list(scaler_data["features"])
        lo = float(scaler_data.get("scale_clip_min", -5.0))
        hi = float(scaler_data.get("scale_clip_max", 5.0))
        return scaler, features, lo, hi
    raise ValueError(
        "Ожидался dict из train.py: {scaler, features, scale_clip_min, scale_clip_max}. "
        "Переобучите модель текущим train.py."
    )


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (logits - np.max(logits)) / max(temperature, 1e-9)
    e = np.exp(z)
    return e / np.sum(e)


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

# ── Quantile (логика без изменений; артефакты в quantile_model/models) ─────────
_QUANT_RUN = os.environ.get("QUANTILE_MODEL_RUN", "2026-05-08T01-04-55")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(_repo_root(), p)


SCALER_PATH = _resolve_path(
    f"technical_analysis_trainer/quantile_model/models/{_QUANT_RUN}/scaler_{_QUANT_RUN}.pkl"
)
ONNX_MODEL_PATH = _resolve_path(
    f"technical_analysis_trainer/quantile_model/models/{_QUANT_RUN}/model_{_QUANT_RUN}.onnx"
)

# Trend: run id или пусто → последний scaler в output_dir из YAML
_TREND_RUN = os.environ.get("TREND_MODEL_RUN", "").strip()
_TREND_CONFIG = os.environ.get(
    "TREND_REALTIME_CONFIG",
    os.path.join(os.path.dirname(__file__), "trend_model", "test_config.yaml"),
)

TAIL_ROWS = int(os.environ.get("REALTIME_CSV_TAIL", "1000"))
BINANCE_LIMIT = int(os.environ.get("REALTIME_BINANCE_LIMIT", "400"))

SEQ_LEN = 24
FEATURES = ["open", "high", "low", "close", "log_return", "volume_log", "rsi_scaled"]


def _load_quantile_scaler(scaler_path: str):
    with open(scaler_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "scaler" in data:
        return data["scaler"]
    return data


def fetch_last_candles_sync(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not klines:
        raise RuntimeError("No data returned from Binance")
    df = pd.DataFrame(
        klines,
        columns=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        }
    )
    return df


async def fetch_last_candles(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_last_candles_sync, symbol, interval, limit)


def _overlay_csv_tail_with_live(csv_path: str, live: pd.DataFrame, tail_n: int) -> pd.DataFrame:
    path = _resolve_path(csv_path)
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
    merged = merged.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    return merged


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


def build_trend_x_window(cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Как test.py: cross → mtf → add_features(return_df_feat) → backfill → clip → последнее окно.
    Возвращает X_window формы (seq_len, n_features) и метаданные для отладки.
    """
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
            raise ValueError("Пустой BTC ряд после stitch")
        d = _ensure_schema_for_features(d)
        if (cross_eff or {}).get("enabled", False):
            d = attach_cross_market_features(d, cross_eff)
        if (mtf_cfg or {}).get("enabled", True):
            d = attach_mtf_features(d, mtf_cfg)

        models_root = _resolve_path(str(cfg["output_dir"]))
        run_id = _TREND_RUN or latest_model_run(models_root)
        run_dir = os.path.join(models_root, run_id)
        scaler_path = os.path.join(run_dir, f"scaler_{run_id}.pkl")
        onnx_path = os.path.join(run_dir, f"model_{run_id}.onnx")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(scaler_path)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(onnx_path)

        scaler, features, scale_lo, scale_hi = load_scaler_bundle(scaler_path)
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

        df_feat, _miss = backfill_missing_features_with_scaler_means(
            df_feat=df_feat,
            features=features,
            scaler=scaler,
        )

        min_rows = seq_len + max(horizon, 1) + 1
        if len(df_feat) < min_rows:
            raise ValueError(
                f"После фичей {len(df_feat)} строк, нужно >= {min_rows} "
                f"(seq_len={seq_len}, horizon={horizon}). Увеличьте TAIL_ROWS / predict CSV."
            )

        X_scaled = np.clip(
            scaler.transform(df_feat[features].values.astype(np.float32)),
            scale_lo,
            scale_hi,
        ).astype(np.float32)

        i = len(X_scaled) - 1
        if i < seq_len:
            raise ValueError(f"Недостаточно строк для окна: i={i}, seq_len={seq_len}")

        x_input = X_scaled[i - seq_len : i].reshape(seq_len, len(features))
        meta = {
            "trend_run_id": run_id,
            "onnx_path": onnx_path,
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


def scaleData(df: pd.DataFrame, scaler_path: str):
    print("Starting get data (quantile)")

    df = df.iloc[:-1].reset_index(drop=True)

    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_log"] = np.log(df["volume"] + 1)
    df["rsi_scaled"] = df["rsi14"] / 100
    df = df.dropna().reset_index(drop=True)

    scaler = _load_quantile_scaler(scaler_path)
    print("✅ Quantile scaler loaded")

    X_raw = df[FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    X_scaled = X_scaled[-SEQ_LEN:]
    df = df.iloc[-SEQ_LEN:].reset_index(drop=True)

    return X_scaled, df


def quantilePredict(X_window: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
    print("Starting quantile predict")

    ort_session = ort.InferenceSession(_resolve_path(ONNX_MODEL_PATH))
    input_name = ort_session.get_inputs()[0].name

    X_window = X_window[np.newaxis, ...]

    pred_log_returns = ort_session.run(None, {input_name: X_window})[0][0]

    last_close = df["close"].iloc[-1]

    price_q1 = last_close * np.exp(pred_log_returns[0])
    price_q2 = last_close * np.exp(pred_log_returns[1])
    price_q3 = last_close * np.exp(pred_log_returns[2])

    next_ts = pd.to_datetime(df["ts"].iloc[-1]) + pd.Timedelta(hours=1)

    return {
        "ts": next_ts,
        "q10": price_q1,
        "q50": price_q2,
        "q90": price_q3,
        "log_returns": pred_log_returns,
    }


def trendPredict(X_window: np.ndarray, onnx_path: str) -> dict[str, Any]:
    print("Starting trend predict")

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    x = X_window[np.newaxis, ...].astype(np.float32)
    logits = np.asarray(ort_session.run(None, {input_name: x})[0][0], dtype=np.float64).ravel()
    probs = softmax_np(logits, temperature=1.0)

    class_names = ["DOWN", "UP"] if len(probs) == 2 else [f"C{i}" for i in range(len(probs))]
    pred_i = int(np.argmax(probs))

    result: dict[str, Any] = {
        "predicted_trend": class_names[pred_i],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))},
        "logits": logits.tolist(),
    }
    for i, name in enumerate(class_names):
        result[name] = f"{probs[i] * 100:.2f}%"
    print(result)
    return result


def load_trend_config() -> dict[str, Any]:
    cfg_path = _resolve_path(_TREND_CONFIG)
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


async def compute_predictions_from_df(
    data: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Quantile + trend predictions from one BTCUSDT 1h candle DataFrame (Binance schema).
    Trend path still uses build_trend_x_window (CSV overlay + cross/MTF) internally.

    Returns:
        quantile_result, trend_result, trend_meta (onnx_path, run id, last_ts, etc.)
    """
    if data is None or len(data) == 0:
        raise ValueError("Empty or missing candle dataframe")

    trend_cfg = load_trend_config()

    X_window, df = scaleData(data, SCALER_PATH)
    X_window_trend, trend_meta = await asyncio.to_thread(build_trend_x_window, trend_cfg)

    print("Quantile df tail ts:", df["ts"].iloc[-1] if len(df) else None)
    print("Trend meta:", trend_meta)

    preds = quantilePredict(X_window, df)
    trend_res = trendPredict(X_window_trend, str(trend_meta["onnx_path"]))

    return preds, trend_res, trend_meta


async def getPreds():
    print("Starting fetching data and prediction")

    data = await fetch_last_candles("BTCUSDT", "1h", 45)

    if data is None:
        print("No data fetched")
        return "No data fetched"
    if len(data) == 0:
        print("No data fetched")
        return "No data fetched"

    preds, trend_res, _meta = await compute_predictions_from_df(data)
    return preds, trend_res
