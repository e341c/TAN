# main.py
import numpy as np
from dataset import load_dataset
from export_onnx import export_model
import yaml, datetime, os
from quantile_model import train_quantile_lstm
from features import add_features
import pickle
from eval import evaluate_model

# модель неплохо попадает в цены, только для up тренда она чуть больше берет прибыль, чем down тренд, если правильно угадывать тренд, то в целом можно делать какой-то фронт для предсказаний
def main():
    # Config
    cfg_path = os.getenv("CONFIG_PATH", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    horizon = int(cfg["horizon"])
    model_dir = cfg["output_dir"]
    window = cfg.get("window")
    if window is None:
        raise ValueError(
            "Window size must be specified in the config file under 'window' key."
        )

    # Dataset
    df = load_dataset(cfg["data_path"])

    # Features and Target
    X, y, used_features, scaler = add_features(df, seq_len=window["long"])

    print("X[0] Example:", X[0], "y[0] Example:", y[0])
    print("X[1] Example:", X[1], "y[0] Example:", y[1])
    print("X[2] Example:", X[2])

    model, train_metrics = train_quantile_lstm(X=X, y=y, cfg=cfg)
    eval_results = evaluate_model(model, X, y)

    # Export
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    os.makedirs(model_dir, exist_ok=True)

    output_dir = f"{model_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler-{timestamp}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    export_model(
        model=model,
        features=used_features,
        eval_results=eval_results,
        output_dir=output_dir,
        version=timestamp,
        horizon=window["long"],
    )

    # Metrics
    print(f"✅ {symbol} model trained with window={window} and horizon={horizon}")
    print(f"📈 Train CV metrics: {train_metrics}")
    print(f"📈 Eval metrics: {eval_results}")


if __name__ == "__main__":
    main()
