from __future__ import annotations

import pickle

import numpy as np
import onnxruntime as ort
import pandas as pd

from technical_analysis_trainer.core.artifacts import ArtifactStore, ModelRun
from technical_analysis_trainer.models.quantile.features import (
    DEFAULT_VOL_WINDOW,
    prepare_inference_window,
)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (logits - np.max(logits)) / max(temperature, 1e-9)
    e = np.exp(z)
    return e / np.sum(e)


def load_bundle(store: ArtifactStore, run_id: str | None = None) -> tuple[dict, ModelRun]:
    """
    Load the training-time bundle {scaler, features, seq_len, vol_window, ...}.

    Older runs saved a bare scaler; those are wrapped with defaults so existing
    artifacts keep loading.
    """
    run = store.get_run(run_id)
    with open(run.scaler_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "scaler" not in data:
        data = {"scaler": data, "target": "log_return"}
    return data, run


def run_quantile_onnx(
    X_window: np.ndarray,
    sigma: float,
    last_close: float,
    df: pd.DataFrame,
    onnx_path: str,
    horizon_hours: int = 1,
    target: str = "vol_normalized_log_return",
    quantiles: list[float] | None = None,
) -> dict:
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    X_batch = X_window[np.newaxis, ...].astype(np.float32)
    raw = ort_session.run(None, {input_name: X_batch})[0][0]

    # The network predicts quantiles of the volatility-normalised return;
    # multiply by the current sigma to get back to log-return space.
    if target == "vol_normalized_log_return":
        pred_log_returns = np.asarray(raw, dtype=np.float64) * sigma
    else:
        pred_log_returns = np.asarray(raw, dtype=np.float64)

    prices = last_close * np.exp(pred_log_returns)
    next_ts = pd.to_datetime(df["ts"].iloc[-1]) + pd.Timedelta(hours=horizon_hours)

    quantiles = list(quantiles or [0.1, 0.5, 0.9])
    by_q = {q: float(p) for q, p in zip(quantiles, prices)}

    def at_level(target_q: float) -> float:
        """
        Interpolate the quantile function to an exact level.

        Snapping to the nearest trained level would silently relabel the band —
        with a [0.05..0.95] grid, "q10" would return q05 and the payload would
        advertise an 80% interval that is really 90% wide.
        """
        lr = float(np.interp(target_q, quantiles, pred_log_returns))
        return float(last_close * np.exp(lr))

    out = {
        "ts": next_ts,
        # q10/q50/q90 stay in the payload for the API, the bot and the web UI,
        # interpolated from whatever grid the model was trained on.
        "q10": at_level(0.1),
        "q50": at_level(0.5),
        "q90": at_level(0.9),
        "quantiles": {f"q{int(q * 100):02d}": v for q, v in by_q.items()},
        "quantile_levels": quantiles,
        "log_returns": [float(x) for x in pred_log_returns],
        "sigma": float(sigma),
        "last_close": float(last_close),
    }
    return out


def predict_from_candles(
    candles: pd.DataFrame,
    store: ArtifactStore,
    seq_len: int | None = None,
    run_id: str | None = None,
    drop_last: bool = True,
) -> tuple[dict, ModelRun]:
    """
    drop_last=True  (live): the final row is the still-forming candle, so it is
                    dropped and the window anchors on the last closed bar.
    drop_last=False (backfill): the caller has already trimmed the frame so its
                    final row *is* the closed bar to anchor on. Dropping it there
                    would silently shift every backfilled prediction one bar
                    earlier than the live path.
    """
    bundle, run = load_bundle(store, run_id)
    scaler = bundle["scaler"]
    seq_len = seq_len or int(bundle.get("seq_len", 24))
    vol_window = int(bundle.get("vol_window", DEFAULT_VOL_WINDOW))
    clip = float(bundle.get("feature_scale_clip", 5.0))
    horizon = int(bundle.get("horizon", 1))

    X_window, sigma, last_close, df = prepare_inference_window(
        candles, scaler, seq_len=seq_len, vol_window=vol_window, clip=clip,
        drop_last=drop_last,
    )
    preds = run_quantile_onnx(
        X_window,
        sigma,
        last_close,
        df,
        str(run.onnx_path),
        horizon_hours=horizon,
        target=str(bundle.get("target", "vol_normalized_log_return")),
        quantiles=bundle.get("quantiles"),
    )
    return preds, run
