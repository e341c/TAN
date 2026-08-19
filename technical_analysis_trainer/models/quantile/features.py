from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from technical_analysis_trainer.core.metrics import har_rv_forecast
from technical_analysis_trainer.features.atr import compute_atr
from technical_analysis_trainer.features.rsi import compute_rsi

# Stationary features only.
#
# Raw OHLC levels are deliberately absent: StandardScaler learns mean/std of the
# price over the training period, so at inference every new price high produces
# feature values the network never saw during training. Everything below is a
# ratio, a return, or a bounded oscillator, so its distribution is stable as the
# price level drifts.
QUANTILE_FEATURES = [
    "log_return",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "body_ratio",
    "roc_3",
    "roc_6",
    "roc_12",
    "rv_24",
    "rv_72",
    "vol_ratio",
    "atr_norm",
    "atr_ratio",
    "volume_norm",
    "vol_z_24",
    "rsi_scaled",
    "bb_position",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# Window for the conditioning volatility sigma_t used to normalise the target.
DEFAULT_VOL_WINDOW = 72

# Longest rolling window in the feature set: the HAR-RV weekly component.
# Everything else (rv_72, vol_z_24, bb_20, atr_14) is shorter, so this is what
# actually sets how much raw history an inference call needs.
LONGEST_FEATURE_WINDOW = 168


def required_history(
    seq_len: int = 24,
    vol_window: int = DEFAULT_VOL_WINDOW,
    horizon: int = 1,
    margin: int = 10,
) -> int:
    """
    Raw bars needed before the model can produce a single prediction.

    Callers that fetch candles must respect this. The old 7-feature pipeline ran
    on ~40 bars, so fetch limits sized for it silently yield zero usable rows
    here — the failure surfaces as "0 usable bars", not as a fetch problem.
    """
    return max(LONGEST_FEATURE_WINDOW, vol_window) + seq_len + horizon + margin


def build_features_df(
    df: pd.DataFrame,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.DataFrame:
    """
    Add the stationary feature set plus `sigma` (conditioning volatility).

    `sigma` at row t uses only returns up to and including t, so it is known at
    prediction time and introduces no lookahead.
    """
    df = df.copy()

    close = df["close"].astype(float)
    df["log_return"] = np.log(close / close.shift(1))

    # Candle shape, all relative to close
    df["open_ratio"] = df["open"] / close - 1
    df["high_ratio"] = df["high"] / close - 1
    df["low_ratio"] = df["low"] / close - 1
    candle_range = df["high"] - df["low"]
    df["body_ratio"] = (df["close"] - df["open"]) / (candle_range + 1e-9)

    # Momentum
    df["roc_3"] = close.pct_change(3)
    df["roc_6"] = close.pct_change(6)
    df["roc_12"] = close.pct_change(12)

    # Realised volatility at two horizons + the short/long vol ratio.
    # Volatility clustering is the dominant structure at 1H, and it is what an
    # interval forecast actually needs to condition on.
    df["rv_24"] = df["log_return"].rolling(24, min_periods=24).std()
    df["rv_72"] = df["log_return"].rolling(72, min_periods=72).std()
    df["vol_ratio"] = df["rv_24"] / (df["rv_72"] + 1e-12)

    atr14 = compute_atr(df, period=14)
    df["atr_norm"] = atr14 / close
    df["atr_ratio"] = atr14 / (atr14.rolling(24, min_periods=24).mean() + 1e-12)

    # Volume, normalised against its own recent distribution
    vol_ma = df["volume"].rolling(24, min_periods=24).mean()
    vol_sd = df["volume"].rolling(24, min_periods=24).std()
    df["volume_norm"] = df["volume"] / (vol_ma + 1e-9)
    df["vol_z_24"] = (df["volume"] - vol_ma) / (vol_sd + 1e-12)

    df["rsi_scaled"] = compute_rsi(close, period=14) / 100

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    df["bb_position"] = (close - bb_mid) / (4.0 * bb_std + 1e-9)

    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    else:
        df["hour_sin"] = df["hour_cos"] = df["dow_sin"] = df["dow_cos"] = 0.0

    # Conditioning volatility: std of log returns over the trailing window.
    df["sigma"] = df["log_return"].rolling(vol_window, min_periods=vol_window).std()

    # HAR-RV sigma, carried as a column so it survives the same dropna and
    # windowing as everything else — computing it separately and slicing by
    # length silently misaligns it by the forecast horizon.
    df["har_sigma"] = har_rv_forecast(df["log_return"].to_numpy())

    return df


def add_features(
    df: pd.DataFrame,
    seq_len: int = 24,
    horizon: int = 1,
    vol_window: int = DEFAULT_VOL_WINDOW,
    scaler: StandardScaler | None = None,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    Build (X, y_z, sigma, last_close, features, scaler).

    The target is the *volatility-normalised* forward log return:

        z_t = log(close_{t+horizon} / close_t) / sigma_t

    Predicting z rather than the raw return is what fixes interval coverage under
    volatility clustering: the network learns the shape of the return
    distribution once, and sigma_t supplies the scale that varies bar to bar. A
    fixed-width interval cannot be simultaneously right in a calm regime and a
    violent one; a sigma-scaled one can.

    `scaler` must be passed (fitted on train only) for val/test splits.
    """
    feat_df = build_features_df(df, vol_window=vol_window)

    fwd_log_return = np.log(
        feat_df["close"].shift(-horizon) / feat_df["close"]
    )
    feat_df["target_z"] = fwd_log_return / feat_df["sigma"]

    needed = list(QUANTILE_FEATURES) + ["sigma", "har_sigma", "target_z", "close"]
    feat_df = feat_df.dropna(subset=needed).reset_index(drop=True)

    features = list(QUANTILE_FEATURES)
    data = feat_df[features].to_numpy(dtype=np.float64)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data)

    data_scaled = np.clip(scaler.transform(data), -clip, clip)

    target_z = feat_df["target_z"].to_numpy(dtype=np.float64)
    sigma = feat_df["sigma"].to_numpy(dtype=np.float64)
    har = feat_df["har_sigma"].to_numpy(dtype=np.float64)
    close = feat_df["close"].to_numpy(dtype=np.float64)

    X, y, sig, har_sig, last_close = [], [], [], [], []
    # Window i..i+seq_len-1 predicts the target anchored at bar t = i+seq_len-1.
    for i in range(len(data_scaled) - seq_len + 1):
        t = i + seq_len - 1
        X.append(data_scaled[i : t + 1])
        y.append(target_z[t])
        sig.append(sigma[t])
        har_sig.append(har[t])
        last_close.append(close[t])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(sig, dtype=np.float64),
        np.asarray(last_close, dtype=np.float64),
        np.asarray(har_sig, dtype=np.float64),
        features,
        scaler,
    )


def prepare_inference_window(
    df: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int = 24,
    vol_window: int = DEFAULT_VOL_WINDOW,
    clip: float = 5.0,
    drop_last: bool = True,
) -> tuple[np.ndarray, float, float, pd.DataFrame]:
    """
    Build the scaled window for ONNX inference.

    Returns (X_scaled, sigma_t, last_close, tail_df). Uses exactly the same
    feature construction as training — the only difference is dropping the
    still-forming last candle.
    """
    if drop_last:
        df = df.iloc[:-1]
    feat_df = build_features_df(df.reset_index(drop=True), vol_window=vol_window)

    needed = list(QUANTILE_FEATURES) + ["sigma"]
    feat_df = feat_df.dropna(subset=needed).reset_index(drop=True)


    if len(feat_df) < seq_len:
        need = required_history(seq_len, vol_window)
        raise ValueError(
            f"Need at least {seq_len} usable bars after feature warm-up, got {len(feat_df)}. "
            f"Supply at least {need} raw bars (got {len(df)}); the binding constraint is "
            f"the {LONGEST_FEATURE_WINDOW}-bar HAR-RV window."
        )

    X_raw = feat_df[QUANTILE_FEATURES].to_numpy(dtype=np.float64)
    X_scaled = np.clip(scaler.transform(X_raw), -clip, clip)[-seq_len:]

    tail = feat_df.iloc[-seq_len:].reset_index(drop=True)
    sigma_t = float(tail["sigma"].iloc[-1])
    last_close = float(tail["close"].iloc[-1])

    return X_scaled.astype(np.float32), sigma_t, last_close, tail
