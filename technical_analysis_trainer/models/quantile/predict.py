from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pandas as pd

from technical_analysis_trainer.core.artifacts import ArtifactStore, ModelRun
from technical_analysis_trainer.models.quantile.features import prepare_inference_window


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (logits - np.max(logits)) / max(temperature, 1e-9)
    e = np.exp(z)
    return e / np.sum(e)


def run_quantile_onnx(
    X_window: np.ndarray,
    df: pd.DataFrame,
    onnx_path: str,
) -> dict:
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    X_batch = X_window[np.newaxis, ...]
    pred_log_returns = ort_session.run(None, {input_name: X_batch})[0][0]

    last_close = float(df["close"].iloc[-1])
    price_q1 = last_close * np.exp(pred_log_returns[0])
    price_q2 = last_close * np.exp(pred_log_returns[1])
    price_q3 = last_close * np.exp(pred_log_returns[2])
    next_ts = pd.to_datetime(df["ts"].iloc[-1]) + pd.Timedelta(hours=1)

    return {
        "ts": next_ts,
        "q10": float(price_q1),
        "q50": float(price_q2),
        "q90": float(price_q3),
        "log_returns": [float(x) for x in pred_log_returns],
        "last_close": last_close,
    }


def predict_from_candles(
    candles: pd.DataFrame,
    store: ArtifactStore,
    seq_len: int = 24,
) -> tuple[dict, ModelRun]:
    scaler, run = store.load_scaler_raw()
    X_window, df = prepare_inference_window(candles, scaler, seq_len=seq_len)
    preds = run_quantile_onnx(X_window, df, str(run.onnx_path))
    return preds, run
