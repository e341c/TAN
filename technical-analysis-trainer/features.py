# features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def add_features(
    df: pd.DataFrame,
    seq_len: int = 30,
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

    data = df[features].values
    target = df["log_return"].values  # future return

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(data[:-seq_len])  # no future leakage

    data_scaled = scaler.transform(data)

    X = []
    y = []

    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i : i + seq_len])
        y.append(target[i + seq_len])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        features,
        scaler,
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean().fillna(0)
    avg_loss = loss.rolling(window=period, min_periods=period).mean().fillna(0)

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    return rsi
