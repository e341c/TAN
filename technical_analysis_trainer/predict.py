import pandas as pd
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import pickle
from features import compute_rsi
from check_no_leakage import check_no_leakage

# =============================
# CONFIG
# =============================
DATA_PATH = "./data/btcusdt_1h_predict.csv"
PIPELINE_PATH = "models/2026-01-16T11-36-11/scaler-2026-01-16T11-36-11.pkl"
ONNX_MODEL_PATH = "models/2026-01-16T11-36-11/model_2026-01-16T11-36-11.onnx"

FEATURES = ["open_rel", "high_rel", "low_rel", "log_return", "volume_log", "rsi_scaled"]
SEQ_LEN = 30
QUANTILES = [0.1, 0.5, 0.9]

# =============================
# 1. Load data
# =============================
df = pd.read_csv(DATA_PATH)
df = df.copy()

# Compute RSI
df["rsi14"] = compute_rsi(df["close"], period=14)

# Compute feature transformations
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["open_rel"] = (df["open"] - df["close"]) / df["close"]
df["high_rel"] = (df["high"] - df["close"]) / df["close"]
df["low_rel"] = (df["low"] - df["close"]) / df["close"]
df["volume_log"] = np.log(df["volume"] + 1)
df["rsi_scaled"] = df["rsi14"] / 100

df = df.dropna().reset_index(drop=True)  # drop first row due to log_return

# =============================
# 2. Load scaler
# =============================
with open(PIPELINE_PATH, "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded", scaler)

# =============================
# 3. Scale features
# =============================
X_raw = df[FEATURES].values.astype(np.float32)
X_scaled = scaler.transform(X_raw)

# =============================
# 4. Build sequences
# =============================
X_seq = np.stack(
    [X_scaled[i : i + SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]
).astype(np.float32)

print(X_seq[0])
print(X_seq[1])

target_idx = 3  # log_return column, as in your FEATURES
pred_t0 = df["log_return"].values[SEQ_LEN]  # prediction target for X_seq[0]
pred_t1 = df["log_return"].values[SEQ_LEN + 1]  # target for X_seq[1]

check_no_leakage(X_seq, df, target_feature="log_return", features_list=FEATURES, seq_len=30)

# check if target is inside sequences
print("Target t0 in X_seq[0] log_return column:", pred_t0 in X_seq[0][:, target_idx])
print("Target t1 in X_seq[1] log_return column:", pred_t1 in X_seq[1][:, target_idx])

last_close = df["close"].values[SEQ_LEN:]
ts = df["ts"].values[SEQ_LEN:]

print("X_seq shape:", X_seq.shape)

# =============================
# 5. Load ONNX model
# =============================
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

# =============================
# 6. Predict
# =============================
preds = []

for i in range(len(X_seq)):
    x_input = X_seq[i : i + 1]  # shape (1, seq_len, features)
    out = ort_session.run(None, {input_name: x_input})[0]  # shape (1, n_quantiles)
    preds.append(out[0])

preds = np.array(preds)
print("preds shape:", preds.shape)

# =============================
# 7. Inverse scale predictions (log-returns)
# =============================
target_idx = FEATURES.index("log_return")  # target in features
y_pred_real = preds * scaler.scale_[target_idx] + scaler.mean_[target_idx]

# Convert log-return to price for each quantile
price_q = {q: last_close * np.exp(y_pred_real[:, i]) for i, q in enumerate(QUANTILES)}

# Also keep log-return predictions for reference
pred_q = {q: y_pred_real[:, i] for i, q in enumerate(QUANTILES)}

# =============================
# 8. Create result dataframe
# =============================
result = pd.DataFrame(
    {
        "ts": ts,
        "close_t": last_close,
        "q10_logret": pred_q[0.1],
        "q50_logret": pred_q[0.5],
        "q90_logret": pred_q[0.9],
        "q10_price": price_q[0.1],
        "q50_price": price_q[0.5],
        "q90_price": price_q[0.9],
    }
)

print("RESULT sample:\n", result.head())

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

# Quantile fan
fig.add_trace(
    go.Scatter(
        x=ts,
        y=result["q90_price"],
        name="Q90",
        line=dict(color="rgba(0,255,0,0.8)"),
    )
)

fig.add_trace(
    go.Scatter(
        x=ts,
        y=result["q10_price"],
        name="Q10",
        line=dict(color="rgba(255,0,0,0.8)"),
        fill="tonexty",
        fillcolor="rgba(0,255,0,0.15)",
    )
)

fig.add_trace(
    go.Scatter(
        x=ts,
        y=result["q50_price"],
        name="Q50 (Median)",
        line=dict(color="rgba(255,255,255,0.9)", dash="dash"),
    )
)

fig.update_layout(
    title="BTCUSDT — ONNX Predictions vs Real Prices",
    template="plotly_dark",
    yaxis=dict(fixedrange=False),  # allow zooming on y-axis
)

fig.show()