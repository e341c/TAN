import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
import torch
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.trend_model.features import compute_rsi

# =============================
# Paths and parameters
# =============================
TIMESTAMP_FOLDER = "2026-02-19T14-35-34"
TIMESTAMP = "2026-02-19T14-35-34"

DATA_PATH = "technical_analysis_trainer/data/btcusdt_1h_predict.csv"
SCALER_PATH = (
    f"technical_analysis_trainer/trend_model/models/{TIMESTAMP_FOLDER}/scaler_{TIMESTAMP}.pkl"
)
ONNX_MODEL_PATH = (
    f"technical_analysis_trainer/trend_model/models/{TIMESTAMP_FOLDER}/model_{TIMESTAMP}.onnx"
)

SEQ_LEN = 30
FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "log_return",
    "volume_log",
    "rsi_scaled",
]

THRESHOLD = 0.001  # тот же, что при обучении

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

X_seq = np.stack([X_scaled[i : i + SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]).astype(np.float32)

# =============================
# 4. Load ONNX and predict
# =============================
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

preds = []
classes = ["DOWN", "FLAT", "UP"]
T = 1.0  # температура

for i in range(len(X_seq)):
    x_input = X_seq[i : i + 1]  # (1, seq_len, features)
    out = ort_session.run(None, {input_name: x_input})[0][0]  # (n_classes)

    logits = torch.tensor(out)
    probs = torch.softmax(logits / T, dim=0).numpy()
    pred_class = classes[np.argmax(probs)]
    preds.append(pred_class)

# =============================
# 5. Align preds with df
# =============================
start_idx = len(df) - len(preds)
df_preds = df.iloc[start_idx:].copy()
df_preds["pred"] = preds

# =============================
# 6. Compute actual
# =============================
future_return = (df_preds["close"].shift(-1) - df_preds["close"]) / df_preds["close"]

actual = np.full(len(df_preds), "FLAT", dtype=object)
actual[future_return > THRESHOLD] = "UP"
actual[future_return < -THRESHOLD] = "DOWN"

df_preds["actual"] = actual

# Отбрасываем последнюю строку (нет следующей свечи)
df_preds = df_preds.iloc[:-1].reset_index(drop=True)

# =============================
# 7. Metrics
# =============================
print("Confusion matrix:")
print(confusion_matrix(df_preds["actual"], df_preds["pred"], labels=classes))
print("\nClassification report:")
print(classification_report(df_preds["actual"], df_preds["pred"], digits=4))

# =============================
# 8. Plot
# =============================
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
        marker=dict(symbol="triangle-up", size=10, color="lime"),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_preds.loc[down_mask, "ts"],
        y=df_preds.loc[down_mask, "high"] * 1.005,
        mode="markers",
        name="Prediction DOWN",
        marker=dict(symbol="triangle-down", size=10, color="red"),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_preds.loc[flat_mask, "ts"],
        y=df_preds.loc[flat_mask, "close"],
        mode="markers",
        name="Prediction FLAT",
        marker=dict(symbol="circle", size=6, color="gray", opacity=0.6),
    )
)

fig.update_layout(
    title="BTCUSDT — ONNX Predictions vs Real Prices",
    template="plotly_dark",
    yaxis=dict(fixedrange=False),
)

fig.show()