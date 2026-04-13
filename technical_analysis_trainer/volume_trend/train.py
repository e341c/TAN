import csv
import random
import yaml
import datetime
import os
import pickle
import numpy as np
import torch

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.export_onnx import export_model

# from technical_analysis_trainer.volume_trend.eval import evaluate_model
from technical_analysis_trainer.volume_trend.features import add_features
from technical_analysis_trainer.volume_trend.features import build_features_df


def report_last_n_volume_spikes(
    df,
    seq_len: int,
    n: int = 5,
    volume_spike_mult: float = 2.0,
    context_bars: int = 3,
) -> None:
    """
    Последние n спайков объёма (most_volume == 1) по времени.
    Для каждого — индекс в df_feat, ts, ratio и окно баров вокруг.
    """
    df_feat, _ = build_features_df(
        df, seq_len=seq_len, volume_spike_mult=volume_spike_mult
    )
    col = "volume_last_vs_prior_mean"
    spike_rows = df_feat.index[df_feat["most_volume"] >= 0.5].tolist()
    pick = spike_rows[-n:] if len(spike_rows) > n else spike_rows
    print(f"last {len(pick)} volume spike(s) (most_volume), of {len(spike_rows)} total spikes")
    show_cols = [
        c
        for c in ("ts", "open", "high", "low", "close", "volume", col, "most_volume")
        if c in df_feat.columns
    ]
    for k, idx in enumerate(pick, start=1):
        val = float(df_feat.loc[idx, col])
        print(f"--- spike {k}/{len(pick)} | df_feat index {idx} | {col}={val}")
        if "ts" in df_feat.columns:
            print(f"  ts: {df_feat.loc[idx, 'ts']}")
        lo = max(0, int(idx) - context_bars)
        hi = min(len(df_feat) - 1, int(idx) + context_bars)
        print(df_feat.loc[lo:hi, show_cols].to_string())
        print()


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple:
    """
    Хронологический split без перемешивания.
    train | val | test  (в порядке времени)
    """
    n = len(X)
    val_start = int(n * (1 - val_ratio - test_ratio))
    test_start = int(n * (1 - test_ratio))

    return (
        X[:val_start],
        y[:val_start],
        X[val_start:test_start],
        y[val_start:test_start],
        X[test_start:],
        y[test_start:],
    )


def set_seed(seed: int):
    """Фиксирует seed для воспроизводимости: веса модели, порядок батчей."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    symbol = cfg["symbol"]
    threshold = cfg["threshold"]
    horizon = cfg.get("horizon", 3)
    model_dir = cfg["output_dir"]
    window = cfg.get("window")
    if window is None:
        raise ValueError("Window size must be specified in config under 'window' key.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    df = load_dataset(cfg["data_path"])

    # Сначала делаем хронологический split по сырому датафрейму,
    # чтобы scaler обучался только на train-части (нет leakage)
    n = len(df)
    val_ratio = cfg.get("val_ratio", 0.1)
    test_ratio = cfg.get("test_ratio", 0.1)
    val_idx = int(n * (1 - val_ratio - test_ratio))
    test_idx = int(n * (1 - test_ratio))

    train_df = df.iloc[:val_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx:test_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx:].reset_index(drop=True)

    # Features: fit scaler только на train
    X_train, y_train, used_features, scaler = add_features(
        train_df, seq_len=window["long"], threshold=threshold, horizon=horizon
    )

    print("X_train:", X_train[0][0])
    print("y_train:", y_train[0])

    # ── Export ────────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(model_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "features": used_features}, f)

    # export_model(
    #     model=model,
    #     features=used_features,
    #     eval_results=eval_results,
    #     output_dir=output_dir,
    #     version=timestamp,
    #     horizon=window["long"],
    #     threshold=threshold,
    # )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"✅ {symbol} | window={window['long']} | threshold={threshold}")
    # print(f"📈 Train metrics : {train_metrics['classes']}")
    # print(f"📊 Test  metrics : {eval_results}")

    # ── Лог в CSV для сравнения запусков ──────────────────────────────────────
    train_start_ts = (
        str(train_df["ts"].iloc[0])
        if "ts" in train_df.columns and len(train_df)
        else ""
    )
    train_end_ts = (
        str(train_df["ts"].iloc[-1])
        if "ts" in train_df.columns and len(train_df)
        else ""
    )
    train_return_pct = (
        (train_df["close"].iloc[-1] - train_df["close"].iloc[0])
        / train_df["close"].iloc[0]
        * 100
        if "close" in train_df.columns and len(train_df) > 0
        else None
    )
    # cm = eval_results.get("confusion_matrix", [])
    # cm_str = str(cm) if cm else ""

    report_last_n_volume_spikes(train_df, window["long"])

    row = {
        "timestamp": timestamp,
        "seed": cfg.get("seed", ""),
        "window_long": window["long"],
        "horizon": horizon,
        "threshold": threshold,
        # "hidden_size": cfg.get("hidden_size", ""),
        # "epochs": cfg.get("epochs", ""),
        # "patience": cfg.get("patience", ""),
        # "weight_decay": cfg.get("weight_decay", ""),
        # "dropout_fc": cfg.get("dropout_fc", ""),
        # "n_train": n_train,
        # "n_val": len(y_val),
        # "n_test": len(y_test),
        # "train_start": train_start_ts,
        # "train_end": train_end_ts,
        # "train_return_pct": train_return_pct,
        # "accuracy": eval_results.get("accuracy", ""),
        # "f1": eval_results.get("f1", ""),
        # "precision": eval_results.get("precision", ""),
        # "recall": eval_results.get("recall", ""),
        # "roc_auc": eval_results.get("roc_auc", ""),
        # "confusion_matrix": cm_str,
        # "best_val_loss": train_metrics.get("best_val_loss", ""),
        # "best_epoch": train_metrics.get("best_epoch", ""),
        # "stopped_early": train_metrics.get("stopped_early", ""),
    }
    runs_csv = os.path.join(BASE_DIR, "volume_trend_train_runs.csv")
    file_exists = os.path.isfile(runs_csv)
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📋 Run logged to {runs_csv}")


if __name__ == "__main__":
    main()
