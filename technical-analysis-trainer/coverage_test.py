import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
import plotly.graph_objects as go
from features import compute_rsi

timestamp = "2026-01-17T23-33-20"

DATA_PATH = "./data/btcusdt_1h.csv"
SCALER_PATH = f"models/{timestamp}/scaler_{timestamp}.pkl"
ONNX_MODEL_PATH = f"models/{timestamp}/model_{timestamp}.onnx"

SEQ_LEN = 30
QUANTILES = [0.1, 0.5, 0.9]
FEATURES = [
    "open", "high", "low", "close", "log_return", "volume_log", "rsi_scaled"
]

# =============================
# 1. Load data and features
# =============================
df = pd.read_csv(DATA_PATH).copy()
df["rsi14"] = compute_rsi(df["close"], period=14)
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["volume_log"] = np.log(df["volume"] + 1)
df["rsi_scaled"] = df["rsi14"] / 100
df = df.dropna().reset_index(drop=True)

# =============================
# 2. Load scaler
# =============================
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded")

# =============================
# 3. Prepare windows
# =============================
X_raw = df[FEATURES].values.astype(np.float32)
X_scaled = scaler.transform(X_raw)

# Фактические log_return для следующей свечи
y_true = df["log_return"].values[SEQ_LEN:]  # t+1

# Формируем все окна для предсказания
X_windows = np.array([X_scaled[i:i+SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)], dtype=np.float32)

# =============================
# 4. Predict with ONNX (batch=1)
# =============================
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

y_pred_quantiles = []
for i in range(len(X_windows)):
    x = X_windows[i:i+1]  # batch=1
    pred = ort_session.run(None, {input_name: x})[0]  # (1, n_quantiles)
    y_pred_quantiles.append(pred[0])
y_pred_quantiles = np.array(y_pred_quantiles)  # (n_samples, n_quantiles)

# =============================
# 5. Coverage test
# =============================
def coverage_test(y_true, y_pred_quantiles, quantiles=[0.1, 0.5, 0.9]):
    y_lower = y_pred_quantiles[:, 0]
    y_upper = y_pred_quantiles[:, -1]
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    print(f"Coverage Q{int(quantiles[0]*100)}-Q{int(quantiles[-1]*100)}: {coverage*100:.2f}%")
    return coverage

coverage_test(y_true, y_pred_quantiles, QUANTILES)

# =============================
# 6. Plot Q10/Q50/Q90 vs price
# =============================
q1 = y_pred_quantiles[:, 0]  # все строки, первый столбец
q2 = y_pred_quantiles[:, 1]  # все строки, второй столбец
q3 = y_pred_quantiles[:, 2]  # все строки, третий столбец

print(y_pred_quantiles)
print(q1)

close_prices = df["close"].values[SEQ_LEN-1:-1]  # Последняя свеча каждого окна

# Переводим предсказания log_return в реальные цены
price_q1 = close_prices * np.exp(q1)
price_q2 = close_prices * np.exp(q2)
price_q3 = close_prices * np.exp(q3)

# Проверка
print(price_q1)
print(price_q2)
print(price_q3)

ts = df["ts"].values[SEQ_LEN:]

last_window = X_scaled[-SEQ_LEN:]  # (SEQ_LEN, n_features)
last_window = last_window[np.newaxis, ...]  # batch=1

# Предсказание через ONNX
pred_next = ort_session.run(None, {input_name: last_window})[0][0]  # (n_quantiles,)

# Переводим в реальные цены
last_close = df["close"].values[-1]
price_next_q1 = last_close * np.exp(pred_next[0])
price_next_q2 = last_close * np.exp(pred_next[1])
price_next_q3 = last_close * np.exp(pred_next[2])

print("Next predicted prices:")
print(price_next_q1, price_next_q2, price_next_q3)

next_ts = pd.to_datetime(df["ts"].values[-1]) + pd.Timedelta(hours=1)

ts_extended = np.append(ts, next_ts) # type: ignore
price_q1_ext = np.append(price_q1, price_next_q1)
price_q2_ext = np.append(price_q2, price_next_q2)
price_q3_ext = np.append(price_q3, price_next_q3)

# =============================
# 9. Plot candles + quantile predictions
# =============================
fig = go.Figure()

# Candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df["ts"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="BTCUSDT",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        opacity=0.9,
    )
)

# Quantile fan с будущей свечой
fig.add_trace(
    go.Scatter(
        x=ts_extended,
        y=price_q3_ext,
        name="Q90",
        line=dict(color="rgba(0,255,0,0.8)"),
    )
)

fig.add_trace(
    go.Scatter(
        x=ts_extended,
        y=price_q1_ext,
        name="Q10",
        line=dict(color="rgba(255,0,0,0.8)"),
        fill="tonexty",
        fillcolor="rgba(0,255,0,0.15)",
    )
)

fig.add_trace(
    go.Scatter(
        x=ts_extended,
        y=price_q2_ext,
        name="Q50 (Median)",
        line=dict(color="rgba(255,255,255,0.9)", dash="dash"),
    )
)

fig.update_layout(
    title="BTCUSDT — ONNX Predictions vs Real Prices + Next Candle",
    template="plotly_dark",
    yaxis=dict(fixedrange=False),
)

fig.show()