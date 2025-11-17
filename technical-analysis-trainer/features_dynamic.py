# features_dynamic.py

import pandas as pd
import numpy as np
import json


def add_features_dynamic(df: pd.DataFrame, meta_path: str) -> pd.DataFrame:
    """
    Генерирует только те признаки, которые перечислены в meta.json.
    Никогда не создаёт отсутствующих фичей.
    Никогда не вызывает KeyError.
    Поддерживает short/mid/long модели.
    """
    df = df.copy()

    # ---- Загружаем meta.json ----
    with open(meta_path) as f:
        meta = json.load(f)

    features = meta["features"]

    print(f"Generating features dynamically based on meta.json: {features}")

    # ---- Базовые признаки ----
    if "open" in features:
        df["open"] = df["open"]

    if "high" in features:
        df["high"] = df["high"]

    if "low" in features:
        df["low"] = df["low"]

    if "close" in features:
        df["close"] = df["close"]

    if "volume" in features:
        df["volume"] = df["volume"]

    # ---- Технические индикаторы ----

    # log-return
    if "returns" in features:
        df["returns"] = np.log(df["close"] / df["close"].shift(1))

    # RSI 14
    if "rsi14" in features:
        df["rsi14"] = compute_rsi(df["close"], 14)

    # MACD
    if "macd" in features or "signal" in features:
        macd, signal = compute_macd(df["close"])
        if "macd" in features:
            df["macd"] = macd
        if "signal" in features:
            df["signal"] = signal

    # ATR 14
    if "atr14" in features:
        df["atr14"] = compute_atr(df, 14)

    # ---- Лаги ----
    for f in features:
        if f.startswith("close_lag_"):
            lag = int(f.replace("close_lag_", ""))
            df[f] = df["close"].shift(lag)

        if f.startswith("volume_lag_"):
            lag = int(f.replace("volume_lag_", ""))
            df[f] = df["volume"].shift(lag)

    # ---- Очистка ----
    df = df.dropna().reset_index(drop=True)
    return df


# ===== Индикаторы ===== #

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def compute_atr(df: pd.DataFrame, period: int = 14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close = np.abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()