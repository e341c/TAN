import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# -----------------------------
# 1. Load new BTCUSDT dataset
# -----------------------------
df_new = pd.read_csv("./data/btcusdt_1h_predict.csv")  # columns: open, high, low, close, volume
features = ["open", "high", "low", "close", "volume"]
seq_len = 30

# -----------------------------
# 2. Scale features like training
# -----------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_new[features])  # fit on new data

# -----------------------------
# 3. Prepare rolling windows
# -----------------------------
X_windows = []
y_real = []
for i in range(len(data_scaled) - seq_len):
    X_windows.append(data_scaled[i:i+seq_len])
    y_real.append(data_scaled[i+seq_len][3])  # target: close price

X_windows = np.array(X_windows, dtype=np.float32)  # shape: (num_samples, seq_len, features)
y_real = np.array(y_real, dtype=np.float32)

# -----------------------------
# 4. Load ONNX model
# -----------------------------
onnx_model_path = "./models/model_2025-12-12T19-30-45.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# -----------------------------
# 5. Predict prices
# -----------------------------
# ONNXRuntime expects inputs as a dictionary
y_pred_scaled = []
for x in X_windows:
    x = x[np.newaxis, :, :]  # add batch dimension
    ort_inputs = {"X": x}
    y_pred_scaled.append(ort_session.run(None, ort_inputs)[0][0, 0])

y_pred_scaled = np.array(y_pred_scaled)

# -----------------------------
# 6. Convert predictions back to original scale
# -----------------------------
# Only inverse-transform close price
y_pred = scaler.inverse_transform(
    np.concatenate([
        np.zeros((len(y_pred_scaled), 3)),  # dummy columns for open, high, low
        y_pred_scaled.reshape(-1, 1),
        np.zeros((len(y_pred_scaled), 1))   # dummy volume
    ], axis=1)
)[:, 3]

y_real_original = df_new["close"].values[seq_len:]  # real close prices

# -----------------------------
# 7. Calculate percentage difference
# -----------------------------
percent_diff = (y_pred - y_real_original) / y_real_original * 100

# -----------------------------
# 8. Save results
# -----------------------------
results = pd.DataFrame({
    "predicted": y_pred,
    "real": y_real_original,
    "percent_diff": percent_diff
})

print(results.head())
results.to_csv("btcusdt_predictions_from_onnx.csv", index=False)

# -----------------------------
# 9. Visualization
# -----------------------------
fig = go.Figure()

# Historical candles
fig.add_trace(go.Candlestick(
    x=df_new["ts"][:len(y_real_original)],
    open=df_new["open"].values[seq_len:],
    high=df_new["high"].values[seq_len:],
    low=df_new["low"].values[seq_len:],
    close=y_real_original,
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

# Layout
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