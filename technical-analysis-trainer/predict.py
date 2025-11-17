# import onnxruntime as rt
# import pandas as pd
# import numpy as np
# import json
# import yaml

# # 1️⃣ Загружаем метаданные модели
# with open("./models/2025-11-04T18-42-37/meta.json") as f:
#     meta = json.load(f)

# features = meta["features"]

# # 2️⃣ Загружаем датасет и берём последние строки
# with open("config.yaml") as f:
#     cfg = yaml.safe_load(f)

# df = pd.read_csv(cfg["data_path"])
# df = df.dropna()

# # ⚙️ При желании — добавь сюда те же фичи, что и в training
# from features import add_features
# df = add_features(df).dropna()

# # 3️⃣ Подготовим вход для модели
# X = df[features].tail(1).to_numpy().astype(np.float32)

# # 4️⃣ Загружаем модель
# sess = rt.InferenceSession("./models/2025-11-04T18-42-37/model.onnx")

# # 5️⃣ Предсказание
# pred = sess.run(None, {"float_input": X})[0][0][0]

# # 6️⃣ Вывод
# print("🧠 Model prediction (log-return):", pred)
# print("💰 Predicted close ratio:", np.exp(pred))
# if pred > 0:
#     print("📈 Expecting upward move")
# elif pred < 0:
#     print("📉 Expecting downward move")
# else:
#     print("➖ Neutral signal")

import numpy as np
import pandas as pd
import onnxruntime as rt
import json, yaml
from features import add_features

# meta + config
meta = json.load(open("./models/2025-11-04T18-42-37/meta.json"))
features = meta["features"]
cfg = yaml.safe_load(open("config.yaml"))

# данные и фичи
df_raw = pd.read_csv(cfg["data_path"])
df = add_features(df_raw).dropna()

X = df[features].tail(1).to_numpy().astype(np.float32)
C_last = float(df["close"].iloc[-1])

# инференс
sess = rt.InferenceSession("./models/2025-11-04T18-42-37/model.onnx")
pred = float(sess.run(None, {"float_input": X})[0][0][0])

# перевод в цену
ratio = np.exp(pred)
next_price = C_last * ratio
delta_abs = next_price - C_last
delta_pct = (ratio - 1.0) * 100.0

np.save("./predictions.npy", pred)
print("✅ Saved predictions to ./predictions.npy")

print(f"🧠 pred (log-return): {pred:.8f}")
print(f"💵 last close: {C_last:.2f}")
print(f"🎯 next price ≈ {next_price:.2f}")
print(f"Δ ≈ {delta_abs:.2f} ({delta_pct:.5f}%)")