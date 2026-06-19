import csv
import datetime
import os
import pickle
import random

import numpy as np
import torch
import yaml

from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.models.volume.features import add_features, build_features_df


def report_last_n_volume_spikes(
    df,
    seq_len: int,
    n: int = 5,
    volume_spike_mult: float = 2.0,
    context_bars: int = 3,
) -> None:
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    base_dir = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(base_dir, "config.yaml"))
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

    df = load_dataset(cfg["data_path"])

    n = len(df)
    val_ratio = cfg.get("val_ratio", 0.1)
    test_ratio = cfg.get("test_ratio", 0.1)
    val_idx = int(n * (1 - val_ratio - test_ratio))
    test_idx = int(n * (1 - test_ratio))

    train_df = df.iloc[:val_idx].reset_index(drop=True)

    X_train, y_train, used_features, scaler = add_features(
        train_df, seq_len=window["long"], threshold=threshold, horizon=horizon
    )

    print("X_train:", X_train[0][0])
    print("y_train:", y_train[0])

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(model_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "features": used_features}, f)

    print(f"✅ {symbol} | window={window['long']} | threshold={threshold}")
    report_last_n_volume_spikes(train_df, window["long"])

    row = {
        "timestamp": timestamp,
        "seed": cfg.get("seed", ""),
        "window_long": window["long"],
        "horizon": horizon,
        "threshold": threshold,
    }
    runs_csv = os.path.join(base_dir, "volume_train_runs.csv")
    file_exists = os.path.isfile(runs_csv)
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📋 Run logged to {runs_csv}")


if __name__ == "__main__":
    main()
