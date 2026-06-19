import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from technical_analysis_trainer.features.rsi import compute_rsi

QUANTILE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "log_return",
    "volume_log",
    "rsi_scaled",
]


def add_features(
    df: pd.DataFrame,
    seq_len: int = 30,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    df = df.copy()
    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)
    df["volume_log"] = np.log(df["volume"] + 1)
    df["rsi_scaled"] = df["rsi14"] / 100

    features = list(QUANTILE_FEATURES)
    data = df[features].values
    target = df["log_return"].values

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data[:-seq_len])

    data_scaled = scaler.transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i : i + seq_len])
        y.append(target[i + seq_len])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        features,
        scaler,
    )


def prepare_inference_window(
    df: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int = 24,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build scaled window for ONNX inference (drops last bar before RSI, as in production).
    """
    df = df.iloc[:-1].reset_index(drop=True)
    df = df.copy()
    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_log"] = np.log(df["volume"] + 1)
    df["rsi_scaled"] = df["rsi14"] / 100
    df = df.dropna().reset_index(drop=True)

    X_raw = df[QUANTILE_FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)
    X_scaled = X_scaled[-seq_len:]
    df = df.iloc[-seq_len:].reset_index(drop=True)
    return X_scaled, df
