# py-trainer/predict_batch.py
import numpy as np
import pandas as pd
import onnxruntime as rt
import json, yaml
from features import add_features

# === 1. Загрузка модели и конфига ===
cfg = yaml.safe_load(open("config.yaml"))
meta = json.load(open("./models/2025-11-04T18-42-37/meta.json"))
features = meta["features"]

# === 2. Подготовка данных ===
df = pd.read_csv(cfg["data_path"])
df = add_features(df).dropna()

X = df[features].to_numpy().astype(np.float32)
last_price = df["close"].iloc[-1]

# === 3. Загрузка модели ===
sess = rt.InferenceSession("./models/2025-11-04T18-42-37/model.onnx")

# === 4. Предсказания для всех строк ===
preds = sess.run(None, {"float_input": X})[0].flatten()
df["pred"] = preds
df["confidence"] = np.abs(preds)
df["signal"] = np.where(df["pred"] > 0.001, 1,
                 np.where(df["pred"] < -0.001, -1, 0))

# === 5. Сохранение ===
np.save("./predictions.npy", preds)
df.to_csv("./predictions.csv", index=False)
print(f"✅ Saved {len(preds)} predictions to ./predictions.npy and ./predictions.csv")

# === 6. Последняя строка для справки ===
pred_last = preds[-1]
ratio = np.exp(pred_last)
next_price = last_price * ratio
delta_pct = (ratio - 1.0) * 100

print(f"🧠 last pred (log-return): {pred_last:.6f}")
print(f"💵 last close: {last_price:.2f}")
print(f"🎯 next price ≈ {next_price:.2f} ({delta_pct:.3f}%)")