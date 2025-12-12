# features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# MAX_LAG = 24  # <= 24 как ты и хотел
MAX_LAG = {
    "short": 6,
    "mid": 12,
    "long": 24,
}

def add_features(df: pd.DataFrame, seq_len: int = 30, model_name: str = "none", scaler=None) -> tuple[np.ndarray, np.ndarray, list[str], MinMaxScaler]:
    """
    Простые фичи:
      - open, high, low, close, volume
      - returns (log-доходность 1 бара)
      - close_lag_1..N, volume_lag_1..N (N <= 24)AndServelish
    Без RSI/MACD/ATR — убираем потенциальный шум.
    """
    df = df.copy()

    features = ["open", "high", "low", "close", "volume"]

    data = df[features]

    # scaler = MinMaxScaler()
    # data_scaled = scaler.fit_transform(data)

    if scaler is None:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)


    X = []
    y = []
    for i in range(len(data_scaled) - seq_len):
        seq = data_scaled[i:i + seq_len]
        X.append(seq)
        y.append(data_scaled[i + seq_len][3]) # target: close price of the next time step
    return np.array(X), np.array(y).reshape(-1, 1), features, scaler
