# features.py

import pandas as pd
import numpy as np

# MAX_LAG = 24  # <= 24 как ты и хотел
MAX_LAG = {
    "short": 6,
    "mid": 12,
    "long": 24,
}

def add_features(df: pd.DataFrame, window: int = 24, model_name: str = "none") -> pd.DataFrame:
    """
    Простые фичи:
      - open, high, low, close, volume
      - returns (log-доходность 1 бара)
      - close_lag_1..N, volume_lag_1..N (N <= 24)
    Без RSI/MACD/ATR — убираем потенциальный шум.
    """
    df = df.copy()

    # Базовые
    df["open"] = df["open"]
    df["high"] = df["high"]
    df["low"] = df["low"]
    df["close"] = df["close"]
    # df["volume"] = df["volume"]
    df["volume"] = np.log(df["volume"] + 1)

    # Лог-доходность последнего бара
    # df["returns"] = np.log(df["close"] / df["close"].shift(1))
    df["returns"] = np.log(df["close"].pct_change() + 1)

    # Лаги (ограничены 24)
    lag_window = min(window, MAX_LAG[model_name])
    for i in range(1, lag_window + 1):
        df[f"close_lag_{i}"] = df["close"].shift(i)
        df[f"volume_lag_{i}"] = df["volume"].shift(i)

    df = df.dropna().reset_index(drop=True)
    return df

# # features.py
# import pandas as pd
# import numpy as np

# def add_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
#     """
#     Добавляет стандартные технические индикаторы + лаги за последние N свечей.
#     """
#     df = df.copy()
    
#     # --- Базовые индикаторы ---
#     df["returns"] = np.log(df["close"] / df["close"].shift(1))
#     df["rsi14"] = compute_rsi(df["close"], 14)
#     df["macd"], df["signal"] = compute_macd(df["close"])
#     df["atr14"] = compute_atr(df, 14)

#     # --- Лаги (контекст последних N свечей) ---
#     for i in range(1, window + 1):
#         df[f"close_lag_{i}"] = df["close"].shift(i)
#         df[f"volume_lag_{i}"] = df["volume"].shift(i)

#     # --- Очистка ---
#     df = df.dropna().reset_index(drop=True)
#     return df


# def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))


# def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
#     ema_fast = series.ewm(span=fast, adjust=False).mean()
#     ema_slow = series.ewm(span=slow, adjust=False).mean()
#     macd = ema_fast - ema_slow
#     signal_line = macd.ewm(span=signal, adjust=False).mean()
#     return macd, signal_line


# def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
#     high_low = df["high"] - df["low"]
#     high_close = np.abs(df["high"] - df["close"].shift())
#     low_close = np.abs(df["low"] - df["close"].shift())
#     tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#     return tr.rolling(period).mean()