import pandas as pd
import numpy as np
import plotly.graph_objects as go
import onnxruntime as rt
import yaml, json
from features import add_features

# --- ⚙️ Конфигурация ---
MODELS = {
    "short": "./models/short/2025-11-04T22-30-06/model.onnx",
    "mid": "./models/mid/2025-11-04T22-30-08/model.onnx",
    "long": "./models/long/2025-11-04T22-30-09/model.onnx",
}
META = {
    "short": "./models/short/2025-11-04T22-30-06/meta.json",
    "mid": "./models/mid/2025-11-04T22-30-08/meta.json",
    "long": "./models/long/2025-11-04T22-30-09/meta.json",
}
CFG_PATH = "config.yaml"

# --- 1️⃣ Загрузка данных ---
cfg = yaml.safe_load(open(CFG_PATH))
df_raw = pd.read_csv(cfg["data_path"])
df = add_features(df_raw).dropna().reset_index(drop=True)

# --- 2️⃣ Подготовка данных ---
context = 100
df_hist = df.tail(context).copy()
df_pred = df_hist.copy()
last = df_pred.iloc[-1]
start_close = float(last["close"])

# --- 3️⃣ Функция предсказания одной модели ---
def predict_single(model_path, meta_path):
    meta = json.load(open(meta_path))
    features = meta["features"]
    sess = rt.InferenceSession(model_path)
    X = df_pred[features].tail(1).to_numpy().astype(np.float32)
    pred = float(sess.run(None, {"float_input": X})[0][0][0])
    return pred, features

# --- 4️⃣ Получение прогнозов всех моделей ---
preds = {}
for name, model_path in MODELS.items():
    pred, _ = predict_single(model_path, META[name])
    preds[name] = pred
    print(f"🧩 {name.upper()} → log-ret={pred:.6f} → Δ={(np.exp(pred)-1)*100:.3f}%")

# --- 5️⃣ Генерация синтетических свечей для каждого горизонта ---
colors = {"short": "cyan", "mid": "blue", "long": "purple"}
for name, pred in preds.items():
    if name == "short":
        steps = 1
    elif name == "mid":
        steps = 12
    elif name == "long":
        steps = 48
    else:
        continue

    end_close = start_close * np.exp(pred)
    price_path = np.linspace(start_close, end_close, steps + 1)[1:]

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

    # --- Добавляем визуализацию линии прогноза ---
    fig = go.Figure() if name == "short" else fig
    fig.add_trace(go.Scatter(
        x=[last["ts"], next_ts],
        y=[start_close, end_close],
        mode="lines+markers",
        name=f"{name.upper()} forecast ({steps}h)",
        line=dict(color=colors[name], width=2, dash="dot")
    ))

# --- 6️⃣ Добавляем исторические свечи ---
fig.add_trace(go.Candlestick(
    x=df_hist["ts"],
    open=df_hist["open"],
    high=df_hist["high"],
    low=df_hist["low"],
    close=df_hist["close"],
    name="History"
))

# --- 7️⃣ Разделительная линия ---
forecast_start = pd.to_datetime(df_hist["ts"].iloc[-1])
fig.add_shape(
    type="line",
    x0=forecast_start,
    x1=forecast_start,
    y0=df_pred["low"].min(),
    y1=df_pred["high"].max(),
    line=dict(color="orange", width=2, dash="dash")
)
fig.add_annotation(
    x=forecast_start,
    y=df_pred["high"].max(),
    text="Forecast start →",
    showarrow=False,
    font=dict(color="orange", size=12),
)

# --- 8️⃣ Настройки графика ---
fig.update_layout(
    title=f"{cfg['symbol']} — Multi-Horizon Forecasts (1h / 12h / 48h)",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    width=1300,
    height=650
)

fig.show()