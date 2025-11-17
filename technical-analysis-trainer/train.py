# train.py
import numpy as np
from dataset import load_dataset
from features import add_features
from model import train_model
from export_onnx import export_model
from eval import evaluate_model
import yaml, datetime, os


# TARGET_SCALE = 1000.0  # масштаб для таргета
TARGET_SCALE = 100.0  # масштаб для таргета


def main():
    # --- 1️⃣ Конфиг ---
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    horizon = int(cfg["horizon"])
    model_dir = cfg["output_dir"]
    window = cfg.get("window")
    if window is None:
        raise ValueError("Window size must be specified in the config file under 'window' key.")
    os.makedirs(model_dir, exist_ok=True)

    # --- 2️⃣ Данные ---
    df = load_dataset(cfg["data_path"])

    # --- 3️⃣ Фичи ---
    df = add_features(df, window=window, model_name=cfg.get("model_name"))
    print(f"✅ Added {window}-step historical context (lag_window <= 24)")

    # --- 4️⃣ Цель (лог-доходность через H баров, масштабируем) ---
    raw_target = np.log(df["close"].shift(-horizon) / df["close"])
    df["target"] = TARGET_SCALE * raw_target
    df = df.dropna().reset_index(drop=True)

    # --- 5️⃣ Модель ---
    all_features = [c for c in df.columns if c not in ["ts", "target"]]
    X = df[all_features]
    y = df["target"]

    model, train_metrics, used_features = train_model(X, y, cfg)

    # --- 6️⃣ Оценка (на всех данных, уже с отфильтрованными фичами) ---
    from eval import evaluate_model
    eval_results = evaluate_model(model, df[used_features], y)

    # --- 7️⃣ Экспорт ---
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    # export_model(model, used_features, eval_results, model_dir, version=timestamp)
    # export_model(model, features=used_features, eval_results, model_dir, version=timestamp, horizon=horizon)
    export_model(
        model=model,
        features=used_features,
        eval_results=eval_results,
        output_dir=model_dir,
        version=timestamp,
        horizon=horizon,
    )

    print(f"✅ {symbol} model trained with window={window} and horizon={horizon}")
    print(f"📈 Train CV metrics: {train_metrics}")
    print(f"📈 Eval metrics: {eval_results}")


if __name__ == "__main__":
    main()

# # train.py
# import numpy as np
# from dataset import load_dataset
# from features import add_features
# from model import train_model
# from export_onnx import export_model
# from eval import evaluate_model
# import yaml, datetime, os

# def main():
#     # --- 1️⃣ Конфиг ---
#     cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
#     with open(cfg_path) as f:
#         cfg = yaml.safe_load(f)

#     symbol = cfg["symbol"]
#     horizon = int(cfg["horizon"])
#     model_dir = cfg["output_dir"]
#     window = cfg.get("window")  # 👈 Новая настройка
#     if window is None:
#         raise ValueError("Window size must be specified in the config file under 'window' key.")
#     os.makedirs(model_dir, exist_ok=True)

#     # --- 2️⃣ Данные ---
#     df = load_dataset(cfg["data_path"])

#     # --- 3️⃣ Фичи с историей ---
#     df = add_features(df, window)
#     print(f"✅ Added {window}-step historical context")

#     # --- 4️⃣ Целевая переменная (лог-доходность через H часов) ---
#     df["target"] = np.log(df["close"].shift(-horizon) / df["close"])
#     df = df.dropna()

#     # --- 5️⃣ Модель ---
#     features = [c for c in df.columns if c not in ["ts", "target"]]
#     model, metrics = train_model(df[features], df["target"], cfg)
#     eval_results = evaluate_model(model, df[features], df["target"])

#     # --- 6️⃣ Экспорт ---
#     timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
#     export_model(model, features, eval_results, model_dir, version=timestamp)

#     print(f"✅ {symbol} model trained with {window}-step window and horizon={horizon}")
#     print(f"📈 Metrics: {eval_results}")

# if __name__ == "__main__":
#     main()