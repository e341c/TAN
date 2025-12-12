# FIX scaler save to a model and export into onnx
# train.py
import numpy as np
from dataset import load_dataset
# from features import add_features
from export_onnx import export_model
from eval import evaluate_model
import yaml, datetime, os
from model import train_model
from features import add_features

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
        raise ValueError("Window size must be specified in the config file under 'window' key.")
    os.makedirs(model_dir, exist_ok=True)

    # Dataset
    df = load_dataset(cfg["data_path"])

    # Features and Target
    X, y, used_features, scaler = add_features(df, seq_len=window["long"], model_name=symbol, scaler=None)

    model, train_metrics = train_model(X=X, y=y, cfg=cfg, feature_min=scaler.data_min_, feature_max=scaler.data_max_)

    # Evaluation
    from eval import evaluate_model
    eval_results = evaluate_model(model, X, y)

    # Export
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
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
