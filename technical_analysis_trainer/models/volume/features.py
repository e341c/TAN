import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from technical_analysis_trainer.features.rsi import compute_rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def compute_average_volume(df: pd.DataFrame, period: int = 24) -> pd.Series:
    return df["volume"].rolling(window=period, min_periods=period).mean()


def add_volume_window_features(
    df: pd.DataFrame,
    seq_len: int,
    volume_spike_mult: float = 2.0,
) -> None:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 for volume window features")
    vol = df["volume"]
    prior_mean = vol.shift(1).rolling(
        window=seq_len - 1, min_periods=seq_len - 1
    ).mean()
    df["volume_last_vs_prior_mean"] = vol / (prior_mean + 1e-9)
    df["most_volume"] = (df["volume_last_vs_prior_mean"] >= volume_spike_mult).astype(
        np.float64
    )


def build_features_df(
    df: pd.DataFrame,
    seq_len: int = 24,
    volume_spike_mult: float = 2.0,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["atr14"] = compute_atr(df, period=14)
    df["price_roc"] = df["close"].pct_change()
    df["average_volume"] = compute_average_volume(df, period=24)

    bb_period = 20
    bb_k = 2.0
    bb_mid = df["close"].rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std = df["close"].rolling(window=bb_period, min_periods=bb_period).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + bb_k * bb_std
    df["bb_lower"] = bb_mid - bb_k * bb_std

    m = df["price_roc"]
    df["M_signal"] = np.where(m > 0, 1, np.where(m < 0, -1, 0))
    df["RSI_signal"] = np.where(df["rsi14"] < 30, 1, np.where(df["rsi14"] > 70, -1, 0))
    df["BB_signal"] = np.where(
        df["close"] < df["bb_lower"],
        1,
        np.where(df["close"] > df["bb_upper"], -1, 0),
    )
    df["score"] = df["M_signal"] + df["RSI_signal"] + df["BB_signal"]
    df["score_norm"] = df["score"] / 3.0

    df["roc_3"] = df["close"].pct_change(3)
    df["roc_6"] = df["close"].pct_change(6)
    df["roc_12"] = df["close"].pct_change(12)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["macd_hist_norm"] = df["macd_hist"] / (df["close"] + 1e-9)

    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_mid"]) / (bb_range + 1e-9)

    body = df["close"] - df["open"]
    candle_range = df["high"] - df["low"]
    df["body_ratio"] = body / (candle_range + 1e-9)

    atr_ma24 = df["atr14"].rolling(window=24, min_periods=1).mean()
    df["atr_ratio"] = df["atr14"] / (atr_ma24 + 1e-9)

    roll_high_24 = df["high"].rolling(window=24, min_periods=1).max()
    roll_low_24 = df["low"].rolling(window=24, min_periods=1).min()
    df["dist_from_high"] = (df["close"] - roll_high_24) / (df["atr14"] + 1e-9)
    df["dist_from_low"] = (df["close"] - roll_low_24) / (df["atr14"] + 1e-9)

    df["high_ratio"] = df["high"] / df["close"] - 1
    df["low_ratio"] = df["low"] / df["close"] - 1
    df["open_ratio"] = df["open"] / df["close"] - 1
    df["atr_norm"] = df["atr14"] / df["close"]
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_norm"] = df["volume"] / (df["average_volume"] + 1e-9)
    df["rsi_scaled"] = df["rsi14"] / 100

    add_volume_window_features(df, seq_len=seq_len, volume_spike_mult=volume_spike_mult)

    has_funding = "funding_rate" in df.columns
    if has_funding:
        df["funding_rate_scaled"] = df["funding_rate"] * 100
        df["funding_rate_cum3"] = (
            df["funding_rate"].rolling(window=3, min_periods=1).sum() * 100
        )
        funding_features = ["funding_rate_scaled", "funding_rate_cum3"]
    else:
        funding_features = []

    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"])
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        time_features = []

    df = df.dropna().reset_index(drop=True)

    features = [
        "open_ratio",
        "high_ratio",
        "low_ratio",
        "price_roc",
        "log_return",
        "volume_norm",
        "rsi_scaled",
        "atr_norm",
        "score_norm",
        "roc_3",
        "roc_6",
        "roc_12",
        "macd_hist_norm",
        "bb_position",
        "body_ratio",
        "atr_ratio",
        "dist_from_high",
        "dist_from_low",
        "volume_last_vs_prior_mean",
        "most_volume",
        *funding_features,
        *time_features,
    ]

    return df, features


def add_features(
    df: pd.DataFrame,
    seq_len: int = 30,
    threshold: float = 0.002,
    horizon: int = 3,
    volume_spike_mult: float = 2.0,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    df_feat, features = build_features_df(
        df, seq_len=seq_len, volume_spike_mult=volume_spike_mult
    )

    future_return = (df_feat["close"].shift(-horizon) - df_feat["close"]) / df_feat["close"]
    X_raw = df_feat[features].values

    if scaler is None:
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(X_raw)
    else:
        X_raw = scaler.transform(X_raw)

    X_raw = np.clip(X_raw, -5, 5)

    X, Y = [], []
    for i in range(seq_len, len(df_feat) - horizon):
        fr = future_return.iloc[i]
        if abs(fr) <= threshold:
            continue
        window = X_raw[i - seq_len : i]
        if np.isnan(window).any():
            continue
        X.append(window)
        Y.append(1 if fr > threshold else 0)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.int64),
        features,
        scaler,
    )
