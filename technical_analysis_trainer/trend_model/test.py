import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
import torch
import plotly.graph_objects as go

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.trend_model.features import add_features, compute_rsi

TIMESTAMP = "2026-01-23T13-14-54"

DATA_PATH = "technical_analysis_trainer/data/btcusdt_1h_predict.csv"
SCALER_PATH = (
    f"technical_analysis_trainer/trend_model/models/{TIMESTAMP}/scaler-{TIMESTAMP}.pkl"
)
ONNX_MODEL_PATH = (
    f"technical_analysis_trainer/trend_model/models/{TIMESTAMP}/model_{TIMESTAMP}.onnx"
)


SEQ_LEN = 30
QUANTILES = [0.1, 0.5, 0.9]
FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "log_return",
    "volume_log",
    "rsi_scaled",
]


print("Init test")

# =============================
# 1. Load data
# =============================
df = pd.read_csv(DATA_PATH).copy()

df = df.iloc[:-1].reset_index(drop=True)

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

X_seq = np.stack(
    [X_scaled[i : i + SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]
).astype(np.float32)

print(X_seq[0])

ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

preds = []

for i in range(len(X_seq)):
    x_input = X_seq[i : i + 1]  # shape (1, seq_len, features)
    out = ort_session.run(None, {input_name: x_input})[0][0]  # shape (1, n_quantiles)

    T = 3.0  # попробуй 2, 3, 5

    logits = torch.tensor(out)
    probs = torch.softmax(logits / T, dim=0).numpy()

    classes = ["DOWN", "FLAT", "UP"]
    pred_class = classes[np.argmax(probs)]

    preds.append(classes[np.argmax(probs)])


start_idx = len(df) - len(preds)

df_preds = df.iloc[start_idx:].copy()
df_preds["pred"] = preds

up_mask = df_preds["pred"] == "UP"
down_mask = df_preds["pred"] == "DOWN"
flat_mask = df_preds["pred"] == "FLAT"


fig = go.Figure()

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

fig.add_trace(
    go.Scatter(
        x=df_preds.loc[up_mask, "ts"],
        y=df_preds.loc[up_mask, "low"] * 0.995,
        mode="markers",
        name="Prediction UP",
        marker=dict(
            symbol="triangle-up",
            size=10,
            color="lime",
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_preds.loc[down_mask, "ts"],
        y=df_preds.loc[down_mask, "high"] * 1.005,
        mode="markers",
        name="Prediction DOWN",
        marker=dict(
            symbol="triangle-down",
            size=10,
            color="red",
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_preds.loc[flat_mask, "ts"],
        y=df_preds.loc[flat_mask, "close"],
        mode="markers",
        name="Prediction FLAT",
        marker=dict(
            symbol="circle",
            size=6,
            color="gray",
            opacity=0.6,
        ),
    )
)

fig.update_layout(
    title="BTCUSDT — ONNX Predictions vs Real Prices",
    template="plotly_dark",
    yaxis=dict(fixedrange=False),  # allow zooming on y-axis
)

fig.show()
