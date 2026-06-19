import datetime
import os
import pickle

import yaml

from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.export_onnx import export_model
from technical_analysis_trainer.models.quantile.eval import evaluate_model
from technical_analysis_trainer.models.quantile.features import add_features
from technical_analysis_trainer.models.quantile.model import train_quantile_lstm


def main() -> None:
    base_dir = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(base_dir, "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    window = cfg.get("window")
    if window is None:
        raise ValueError("Config must define 'window'")

    df = load_dataset(cfg["data_path"])
    X, y, used_features, scaler = add_features(df, seq_len=window["long"])

    model, train_metrics = train_quantile_lstm(X=X, y=y, cfg=cfg)
    eval_results = evaluate_model(model, X, y)

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    model_dir = cfg["output_dir"]
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

    print(f"✅ {cfg['symbol']} quantile model trained")
    print(f"📈 Train metrics: {train_metrics}")
    print(f"📈 Eval metrics: {eval_results}")


if __name__ == "__main__":
    main()
