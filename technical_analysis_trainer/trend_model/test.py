import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
import torch
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

from technical_analysis_trainer.trend_model.features import build_features_df

# =============================
# Paths and parameters
# =============================
# Папка и метка модели должны совпадать с экспортом (папка может быть с суффиксом _best)
TIMESTAMP_FOLDER = "2026-03-14T01-10-21"
TIMESTAMP = "2026-03-14T01-10-21"

DATA_PATH = "technical_analysis_trainer/data/btcusdt_1h_predict.csv"
SCALER_PATH = (
    f"technical_analysis_trainer/trend_model/models/{TIMESTAMP_FOLDER}/scaler_{TIMESTAMP}.pkl"
)
ONNX_MODEL_PATH = (
    f"technical_analysis_trainer/trend_model/models/{TIMESTAMP_FOLDER}/model_{TIMESTAMP}.onnx"
)

SEQ_LEN = 24
HORIZON = 1  # должен совпадать с horizon в add_features
THRESHOLD = 0.003  # тот же, что при обучении (порог для UP/DOWN)
MIN_CONFIDENCE = 0.55  # не показывать и не учитывать в метриках сигналы слабее этого (max prob)

# =============================
# 1. Load data and build features
# =============================
df_raw = pd.read_csv(DATA_PATH).copy()
df_raw = df_raw.iloc[:-1].reset_index(drop=True)

df, FEATURES = build_features_df(df_raw)

# =============================
# 2. Load scaler и список фичей (как при обучении)
# =============================
with open(SCALER_PATH, "rb") as f:
    scaler_data = pickle.load(f)
if isinstance(scaler_data, dict):
    scaler = scaler_data["scaler"]
    FEATURES = scaler_data["features"]
else:
    scaler = scaler_data
    FEATURES = [f for f in FEATURES if f in df.columns]
    if getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError(
            f"Сохранённый scaler ожидает {scaler.n_features_in_} фичей, в данных {len(FEATURES)}. "
            "Переобучите модель (train) — тогда сохранится список фичей и тест будет использовать те же."
        )
print("✅ Scaler loaded")

# =============================
# 3. Prepare windows
# =============================
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"В данных нет фичей, на которых обучен scaler: {missing}")
X_raw = df[FEATURES].values.astype(np.float32)
X_scaled = scaler.transform(X_raw)

X_seq = np.stack([X_scaled[i : i + SEQ_LEN] for i in range(len(X_scaled) - SEQ_LEN)]).astype(np.float32)

# =============================
# 4. Load ONNX and predict (бинарная модель: [DOWN, UP])
# =============================
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

pred_labels = []  # 0=DOWN, 1=UP
pred_confidences = []  # max(prob) — уверенность модели
T = 1.0  # температура (не влияет на argmax, только на "остроту")

for i in range(len(X_seq)):
    x_input = X_seq[i : i + 1]  # (1, seq_len, features)
    out = ort_session.run(None, {input_name: x_input})[0][0]  # (2,) -> [DOWN, UP]

    logits = torch.tensor(out)
    probs = torch.softmax(logits / T, dim=0).numpy()
    pred_label = int(np.argmax(probs))  # 0=DOWN, 1=UP
    confidence = float(np.max(probs))
    pred_labels.append(pred_label)
    pred_confidences.append(confidence)

classes = [0, 1]
class_names = {0: "DOWN", 1: "UP"}

# =============================
# 5. Align preds with df
# =============================
start_idx = len(df) - len(pred_labels)
df_preds = df.iloc[start_idx:].copy()
df_preds["pred_label"] = np.array(pred_labels, dtype=int)
df_preds["pred_confidence"] = np.array(pred_confidences, dtype=np.float32)
# Только уверенные сигналы: для слабых — пусто (NaN) в pred
confident_mask = df_preds["pred_confidence"] >= MIN_CONFIDENCE
df_preds["pred"] = df_preds["pred_label"].map(class_names)
df_preds.loc[~confident_mask, "pred"] = np.nan

# =============================
# 6. Compute actual (как при обучении: бинарно, без FLAT)
# =============================
future_return = (
    df_preds["close"].shift(-HORIZON) - df_preds["close"]
) / df_preds["close"]

# Маска только для баров, которые вообще попадали бы в train (нет FLAT)
mask = (future_return > THRESHOLD) | (future_return < -THRESHOLD)
# Метрики считаем только по уверенным сигналам (confident_mask уже есть выше)
eval_mask = mask & confident_mask
df_eval = df_preds[eval_mask].copy()
future_return_eval = future_return[eval_mask]

y_true = np.where(future_return_eval > THRESHOLD, 1, 0)  # 1=UP, 0=DOWN
y_pred = df_eval["pred_label"].to_numpy()

n_confident = confident_mask.sum()
print(f"Signals: {len(df_preds)} total, {n_confident} with confidence >= {MIN_CONFIDENCE} (shown on chart)")
print("Pred distribution (all):", dict(Counter(df_preds["pred_label"].map(class_names))))
print("Pred distribution (confident only, on chart):", dict(Counter(df_preds["pred"].dropna())))
unique, counts = np.unique(y_true, return_counts=True)
actual_dist = {class_names[int(k)]: int(v) for k, v in zip(unique, counts)}
print("Actual distribution (eval, UP/DOWN only):", actual_dist)

# =============================
# 7. Metrics (бинарные: 0=DOWN, 1=UP)
# =============================
print("Confusion matrix (0=DOWN, 1=UP):")
print(confusion_matrix(y_true, y_pred, labels=classes))
print("\nClassification report:")
print(
    classification_report(
        y_true, y_pred, digits=4, target_names=[class_names[c] for c in classes]
    )
)

# =============================
# 8. Plot (только уверенные сигналы: confidence >= MIN_CONFIDENCE)
# =============================
up_mask = (df_preds["pred_label"] == 1) & confident_mask
down_mask = (df_preds["pred_label"] == 0) & confident_mask
flat_mask = np.zeros(len(df_preds), dtype=bool)  # больше нет отдельного FLAT-класса

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
        name=f"UP (→ t+{HORIZON})",
        marker=dict(symbol="triangle-up", size=10, color="lime"),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_preds.loc[down_mask, "ts"],
        y=df_preds.loc[down_mask, "high"] * 1.005,
        mode="markers",
        name=f"DOWN (→ t+{HORIZON})",
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
    title=f"BTCUSDT — Predictions (marker at t = forecast for next {HORIZON} candles)",
    template="plotly_dark",
    yaxis=dict(fixedrange=False),
)

fig.show()