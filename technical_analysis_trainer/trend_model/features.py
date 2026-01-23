# features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean().fillna(0)
    avg_loss = loss.rolling(window=period, min_periods=period).mean().fillna(0)

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(
    df: pd.DataFrame,
    seq_len: int = 30,
    horizon: int = 10,
    threshold: float = 0.002,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    Feature engineering for time series data.:
        - open, high, low, close, volume
        - returns (log-доходность 1 бара)
        - close_lag_1..N, volume_lag_1..N (N <= 24)AndServelish
    """
    # dataset has ts,open,high,low,close,volume
    df = df.copy()
    df["rsi14"] = compute_rsi(df["close"], period=14)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)  # dont know what this code do

    df["volume_log"] = np.log(df["volume"] + 1)
    df["rsi_scaled"] = df["rsi14"] / 100

    features = [
        "open",
        "high",
        "low",
        "close",
        "log_return",
        "volume_log",
        "rsi_scaled",
    ]

    X_raw = df[features].values

    # Scale
    if scaler is None:
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(X_raw)
    else:
        X_raw = scaler.transform(X_raw)

    # Target (Trend)
    future_return = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    y = np.zeros(len(df), dtype=np.int64)
    y[future_return > threshold] = 2  # UP
    y[future_return < -threshold] = 0  # DOWN
    y[(future_return >= -threshold) & (future_return <= threshold)] = 1  # FLAT

    # Sequences
    X, Y = [], []
    for i in range(seq_len, len(df) - horizon):
        X.append(X_raw[i - seq_len : i])
        Y.append(y[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.int64),
        features,
        scaler,
    )
