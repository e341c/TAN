# train.py
import yaml
import datetime
import os
import pickle

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.eval import evaluate_model
from technical_analysis_trainer.export_onnx import export_model

from technical_analysis_trainer.trend_model.features import add_features
from technical_analysis_trainer.trend_model.trend_model import train_trend_lstm

# TODO: retrain model and make covrage test
# TODO: make model predictions more accurate, for now it more like 50/50
def main():
    # Config
    BASE_DIR = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # cfg_path = os.getenv("CONFIG_PATH", "config.yaml")

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

    model, train_metrics = train_trend_lstm(X=X, y=y, cfg=cfg)
    eval_results = evaluate_model(model, X, y)

    # Export
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    os.makedirs(model_dir, exist_ok=True)

    output_dir = f"{model_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
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
