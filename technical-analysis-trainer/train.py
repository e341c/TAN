# train.py
import numpy as np
from dataset import load_dataset
from features import add_features
from model import train_model
from export_onnx import export_model
from eval import evaluate_model
import yaml, datetime, os

def main():
    # --- 1️⃣ Конфиг ---
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    horizon = int(cfg["horizon"])
    model_dir = cfg["output_dir"]
    window = cfg.get("window")  # 👈 Новая настройка
    if window is None:
        raise ValueError("Window size must be specified in the config file under 'window' key.")
    os.makedirs(model_dir, exist_ok=True)

    # --- 2️⃣ Данные ---
    df = load_dataset(cfg["data_path"])

    # --- 3️⃣ Фичи с историей ---
    df = add_features(df, window)
    print(f"✅ Added {window}-step historical context")

    # --- 4️⃣ Целевая переменная (лог-доходность через H часов) ---
    df["target"] = np.log(df["close"].shift(-horizon) / df["close"])
    df = df.dropna()

    # --- 5️⃣ Модель ---
    features = [c for c in df.columns if c not in ["ts", "target"]]
    model, metrics = train_model(df[features], df["target"], cfg)
    eval_results = evaluate_model(model, df[features], df["target"])

    # --- 6️⃣ Экспорт ---
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    export_model(model, features, eval_results, model_dir, version=timestamp)

    print(f"✅ {symbol} model trained with {window}-step window and horizon={horizon}")
    print(f"📈 Metrics: {eval_results}")

if __name__ == "__main__":
    main()