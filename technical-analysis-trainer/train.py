# train.py
import numpy as np
from dataset import load_dataset
# from features import add_features
from export_onnx import export_model
from eval import evaluate_model
import yaml, datetime, os
from model import train_model


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

    # # Indicators
    # df = add_features(df, window=window, model_name=cfg.get("model_name"))
    # print(f"✅ Added {window}-step historical context (lag_window <= 24)")

    # # Target
    # raw_target = np.log(df["close"].shift(-horizon) / df["close"])
    # df["target"] = TARGET_SCALE * raw_target
    # df = df.dropna().reset_index(drop=True)

    # # Model training
    # all_features = [c for c in df.columns if c not in ["ts", "target"]]
    # X = df[all_features]
    # y = df["target"]

    model = train_model(X=df[[ "close"]], y=df["close"].shift(-horizon), cfg=cfg)

    # Evaluation
    from eval import evaluate_model
    eval_results = evaluate_model(model, df[used_features], y)

    # Export
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
