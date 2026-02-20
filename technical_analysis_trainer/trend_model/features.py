import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI using exponential moving average."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def add_features(
    df: pd.DataFrame,
    seq_len: int = 30,
    threshold: float = 0.002,
    horizon: int = 3,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    Feature engineering pipeline for OHLCV time series (BTCUSDT 1H).

    Входные колонки: ts, open, high, low, close, volume, funding_rate (опционально)
    Выходные данные:
        X        — массив последовательностей формы (N, seq_len, n_features)
        Y        — метки классов: 1=UP, 0=DOWN (flat-бары исключены)
        features — список имён фичей
        scaler   — обученный StandardScaler (fit только на train-данных)

    ВАЖНО: передавайте scaler=None только для train-сплита.
    Для val/test передавайте обученный scaler, чтобы не было data leakage.
    """
    df = df.copy()

    # ── Технические индикаторы ───────────────────────────────────────────────
    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["atr14"] = compute_atr(df, period=14)

    # ── Нормализованные ценовые фичи ─────────────────────────────────────────
    df["high_ratio"]  = df["high"] / df["close"] - 1
    df["low_ratio"]   = df["low"]  / df["close"] - 1
    df["open_ratio"]  = df["open"] / df["close"] - 1
    df["atr_norm"]    = df["atr14"] / df["close"]
    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))

    vol_ma = df["volume"].rolling(window=24, min_periods=1).mean()
    df["volume_norm"] = df["volume"] / (vol_ma + 1e-9)
    df["rsi_scaled"]  = df["rsi14"] / 100

    # ── Funding rate ─────────────────────────────────────────────────────────
    # Выходит каждые 8H — уже замёрджен через forward fill в fetch_data.py
    # Типичный диапазон: -0.01..+0.01, умножаем на 100 → -1..+1
    has_funding = "funding_rate" in df.columns
    if has_funding:
        df["funding_rate_scaled"] = df["funding_rate"] * 100
        # Накопленное давление за последние 3 выплаты (= 24H)
        # Положительное значение = лонги переплачивают шортам (медвежий сигнал)
        df["funding_rate_cum3"] = df["funding_rate"].rolling(window=3, min_periods=1).sum() * 100
        funding_features = ["funding_rate_scaled", "funding_rate_cum3"]
    else:
        funding_features = []

    # ── Временны́е фичи ───────────────────────────────────────────────────────
    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"])
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        time_features = []

    # Удаляем строки с NaN (первые ~14 баров из-за RSI/ATR)
    df = df.dropna().reset_index(drop=True)

    features = [
        "open_ratio",
        "high_ratio",
        "low_ratio",
        "log_return",
        "volume_norm",
        "rsi_scaled",
        "atr_norm",
        *funding_features,
        *time_features,
    ]

    # ── Target ───────────────────────────────────────────────────────────────
    future_return = (df["close"].shift(-horizon) - df["close"]) / df["close"]

    # ── Масштабирование ───────────────────────────────────────────────────────
    X_raw = df[features].values

    if scaler is None:
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(X_raw)
    else:
        X_raw = scaler.transform(X_raw)

    X_raw = np.clip(X_raw, -5, 5)

    # ── Построение последовательностей ───────────────────────────────────────
    X, Y = [], []
    for i in range(seq_len, len(df) - horizon):
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


# ── Пример правильного использования (train/val/test split) ──────────────────
#
#   train_df, val_df, test_df = split(df)  # хронологически, без перемешивания
#
#   X_train, y_train, features, scaler = add_features(train_df)
#   X_val,   y_val,   _,        _      = add_features(val_df,  scaler=scaler)
#   X_test,  y_test,  _,        _      = add_features(test_df, scaler=scaler)
#
# scaler обучается ТОЛЬКО на train_df — никакого data leakage.