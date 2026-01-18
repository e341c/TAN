import pandas as pd
import numpy as np
import onnxruntime as ort
import pickle
from features import compute_rsi

# =============================
# CONFIG
# =============================
DATA_PATH = "./data/btcusdt_1h_predict.csv"
PIPELINE_PATH = "models/2026-01-16T11-36-11/scaler-2026-01-16T11-36-11.pkl"
ONNX_MODEL_PATH = "models/2026-01-16T11-36-11/model_2026-01-16T11-36-11.onnx"
SEQ_LEN = 30
QUANTILES = [0.1, 0.5, 0.9]
FEATURES = [
    "open_rel",
    "high_rel",
    "low_rel",
    "log_return",
    "volume_log",
    "rsi_scaled",
]

# =============================
# 1. Load data
# =============================
df = pd.read_csv(DATA_PATH).copy()

# Compute features
df["rsi14"] = compute_rsi(df["close"], period=14)
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["open_rel"] = (df["open"] - df["close"]) / df["close"]
df["high_rel"] = (df["high"] - df["close"]) / df["close"]
df["low_rel"] = (df["low"] - df["close"]) / df["close"]
df["volume_log"] = np.log(df["volume"] + 1)
df["rsi_scaled"] = df["rsi14"] / 100
df = df.dropna().reset_index(drop=True)

# =============================
# 2. Load scaler
# =============================
with open(PIPELINE_PATH, "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded")

# =============================
# 3. Prepare last sequence for prediction
# =============================
X_raw = df[FEATURES].values.astype(np.float32)
X_scaled = scaler.transform(X_raw)

# Take last SEQ_LEN rows
last_sequence = X_scaled[-SEQ_LEN:].copy()

# MASK log_return for last row (unknown future candle)
log_return_idx = FEATURES.index("log_return")
last_sequence[-1, log_return_idx] = 0.0  # simulate unknown future return

x_input = last_sequence[None, :, :]  # add batch dimension

# =============================
# 4. Load ONNX model
# =============================
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

# =============================
# 5. Predict next candle
# =============================
pred_next = ort_session.run(None, {input_name: x_input})[0]  # shape (1, n_quantiles)

# Inverse scale
y_pred_real = pred_next * scaler.scale_[log_return_idx] + scaler.mean_[log_return_idx]

# Convert log-return to price for each quantile
last_close = df["close"].iloc[-1]
price_q = {q: last_close * np.exp(y_pred_real[0, i]) for i, q in enumerate(QUANTILES)}

last_ts = df["ts"].iloc[-1]
print(f"Last close for {last_ts}:", last_close)
# Print results
for q in QUANTILES:
    print(f"Predicted next candle (Q{int(q*100)}): {price_q[q]:.2f}")

# TODO: Find where data leakage is happening
# TODO: Remove data leakage
# TODO: Make a test that will show real predcitions