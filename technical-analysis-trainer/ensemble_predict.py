import os
import yaml
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import onnxruntime as rt
from features import add_features

# === ⚙️ Настройки ===
CFG_PATH = "config.yaml"
MODEL_DIR = "./models"
MODELS = {
    "short": "2025-11-04T22-38-52",  # horizon=1
    "mid":   "2025-11-04T22-38-54",  # horizon=12
    "long":  "2025-11-04T22-38-56",  # horizon=48
}
WEIGHTS = {"short": 0.2, "mid": 0.3, "long": 0.5}  # ensemble веса

# === 1️⃣ Загрузка данных ===
cfg = yaml.safe_load(open(CFG_PATH))
df_raw = pd.read_csv(cfg["data_path"])
df = add_features(df_raw).dropna().reset_index(drop=True)

# Последние свечи для прогноза
context = 100
df_hist = df.tail(context).copy()
df_pred = df_hist.copy()

# === 2️⃣ Загрузка моделей ===
sessions = {}
metas = {}
for name, version in MODELS.items():
    meta_path = os.path.join(MODEL_DIR, name, version, "meta.json")
    print(f"Loading {name} model from {version}... {meta_path}")
    model_path = os.path.join(MODEL_DIR, name, version, "model.onnx")
    metas[name] = json.load(open(meta_path))
    sessions[name] = rt.InferenceSession(model_path)

# === 3️⃣ Прогноз по каждой модели ===
preds = {}
for name in MODELS.keys():
    features = metas[name]["features"]
    sess = sessions[name]

    # 🧩 Проверим, хватает ли lag-фич
    missing = [f for f in features if f not in df_pred.columns]
    if missing:
        print(f"⚠️ {name} model expects {len(missing)//2} extra lag steps → generating them...")
        max_lag = max(int(f.split("_")[-1]) for f in missing if "close_lag" in f)
        for i in range(1, max_lag + 1):
            df_pred[f"close_lag_{i}"] = df_pred["close"].shift(i)
            df_pred[f"volume_lag_{i}"] = df_pred["volume"].shift(i)
        df_pred = df_pred.dropna().reset_index(drop=True)

    # ✅ теперь все нужные фичи есть
    X = df_pred[features].tail(1).to_numpy().astype(np.float32)
    pred = float(sess.run(None, {"float_input": X})[0][0][0])
    preds[name] = pred
    print(f"{name}: log-ret={pred:.6f} → Δ≈{(np.exp(pred)-1)*100:.3f}%")

# === 4️⃣ Комбинированный прогноз ===
ensemble_pred = sum(preds[name] * WEIGHTS[name] for name in preds)
print("\n🧠 Ensemble forecast:")
print(f"  Weighted log-return = {ensemble_pred:.6f}")
print(f"  Δ≈{(np.exp(ensemble_pred)-1)*100:.3f}%")

# === 5️⃣ Генерация прогнозных свечей ===
HORIZON = cfg["horizon"]
last = df_pred.iloc[-1]
start_close = float(last["close"])
end_close = start_close * np.exp(ensemble_pred)
price_path = np.linspace(start_close, end_close, HORIZON + 1)[1:]

for i, next_close in enumerate(price_path):
    next_ts = pd.to_datetime(last["ts"]) + pd.Timedelta(hours=i + 1)
    prev_close = df_pred.iloc[-1]["close"]
    df_pred.loc[len(df_pred)] = {
        **last.to_dict(),
        "ts": next_ts,
        "open": prev_close,
        "high": next_close * 1.002,
        "low": next_close * 0.998,
        "close": next_close,
        "volume": last["volume"],
    }

# === 6️⃣ Построение графика ===
fig = go.Figure()

# Исторические свечи
fig.add_trace(go.Candlestick(
    x=df_hist["ts"],
    open=df_hist["open"],
    high=df_hist["high"],
    low=df_hist["low"],
    close=df_hist["close"],
    name="History"
))

# Прогноз (ensemble)
fig.add_trace(go.Candlestick(
    x=df_pred.tail(HORIZON)["ts"],
    open=df_pred.tail(HORIZON)["open"],
    high=df_pred.tail(HORIZON)["high"],
    low=df_pred.tail(HORIZON)["low"],
    close=df_pred.tail(HORIZON)["close"],
    name="Ensemble Forecast",
    increasing_line_color="deepskyblue",
    decreasing_line_color="deepskyblue",
    opacity=0.8
))

# Линии для каждой модели
colors = {"short": "yellow", "mid": "orange", "long": "red"}
for name in preds:
    pred_close = start_close * np.exp(preds[name])
    x0 = pd.to_datetime(df_hist["ts"].iloc[-1])  # 🧩 fix
    fig.add_trace(go.Scatter(
        x=[x0, x0 + pd.Timedelta(hours=cfg["horizon"])],
        y=[start_close, pred_close],
        mode="lines",
        name=f"{name} forecast (Δ{(np.exp(preds[name])-1)*100:.2f}%)",
        line=dict(color=colors[name], dash="dot")
    ))

# Разделительная линия
forecast_start = pd.to_datetime(df_hist["ts"].iloc[-1])
fig.add_shape(
    type="line",
    x0=forecast_start,
    x1=forecast_start,
    y0=df_pred["low"].min(),
    y1=df_pred["high"].max(),
    line=dict(color="orange", width=2, dash="dash")
)

fig.update_layout(
    title=f"{cfg['symbol']} — Ensemble Forecast (Short + Mid + Long)",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    width=1200,
    height=600
)

fig.show()