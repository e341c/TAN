import pandas as pd
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go

# -----------------------------
# 1️⃣ Load new BTCUSDT dataset
# -----------------------------
df_new = pd.read_csv("./data/btcusdt_1h_predict.csv")  # columns: open, high, low, close, volume
features = ["open", "high", "low", "close", "volume"]
seq_len = 30

# -----------------------------
# 2️⃣ Prepare rolling windows (raw, without scaling)
# -----------------------------
X_windows = []
y_real = []
for i in range(len(df_new) - seq_len):
    seq = df_new[features].values[i:i+seq_len]
    X_windows.append(seq)
    y_real.append(df_new["close"].values[i+seq_len])

X_windows = np.array(X_windows, dtype=np.float32)
y_real = np.array(y_real, dtype=np.float32)

# -----------------------------
# 3️⃣ Load ONNX model
# -----------------------------
onnx_model_path = "./models/model_2025-12-12T20-28-52.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# -----------------------------
# 4️⃣ Predict prices
# -----------------------------
y_pred = []
for x in X_windows:
    x = x[np.newaxis, :, :]  # add batch dimension
    y_out = ort_session.run(None, {"X": x})[0]
    y_pred.append(y_out[0,0])
y_pred = np.array(y_pred)

# -----------------------------
# 5️⃣ Calculate percentage difference
# -----------------------------
percent_diff = (y_pred - y_real) / y_real * 100

# -----------------------------
# 6️⃣ Save results
# -----------------------------
results = pd.DataFrame({
    "predicted": y_pred,
    "real": y_real,
    "percent_diff": percent_diff
})

results.to_csv("btcusdt_predictions_from_onnx.csv", index=False)
print(results.head())

# -----------------------------
# 7️⃣ Visualization
# -----------------------------
fig = go.Figure()

# Historical candles
fig.add_trace(go.Candlestick(
    x=df_new["ts"][:len(y_real)],
    open=df_new["open"].values[seq_len:],
    high=df_new["high"].values[seq_len:],
    low=df_new["low"].values[seq_len:],
    close=y_real,
    name="Real"
))

# Predicted prices line
fig.add_trace(go.Scatter(
    x=df_new["ts"][seq_len:],
    y=y_pred,
    mode="lines+markers",
    name="Predicted",
    line=dict(color="cyan", width=2, dash="dot")
))

fig.update_layout(
    title="BTCUSDT — ONNX Predictions vs Real Prices",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    width=1300,
    height=650
)

fig.show()