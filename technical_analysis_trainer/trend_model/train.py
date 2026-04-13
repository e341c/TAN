import csv
import random
import secrets
import yaml
import datetime
import os
import pickle
import numpy as np
import torch

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.export_onnx import export_model

from technical_analysis_trainer.trend_model.eval import evaluate_model
from technical_analysis_trainer.trend_model.features import add_features
from technical_analysis_trainer.trend_model.trend_model import train_trend_lstm


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
        X[:val_start], y[:val_start],
        X[val_start:test_start], y[val_start:test_start],
        X[test_start:], y[test_start:],
    )


def set_seed(seed: int):
    """Фиксирует seed для воспроизводимости: веса модели, порядок батчей."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_seed_into_cfg(cfg: dict) -> None:
    """
    При seed: auto в YAML — генерирует целый seed, пишет в cfg и вызывает set_seed.
    Иначе приводит seed к int (по умолчанию 42, если ключа нет).
    """
    raw = cfg.get("seed", 42)
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        seed = secrets.randbelow(2**31)
        print(f"🎲 seed: auto → {seed}")
    else:
        try:
            seed = int(raw)
        except (TypeError, ValueError):
            seed = 42
            print(f"⚠ invalid seed {raw!r}, using {seed}")
    cfg["seed"] = seed
    set_seed(seed)


def train_one_run(cfg: dict, base_dir: str) -> None:
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
    X_val, y_val, _, _ = add_features(
        val_df, seq_len=window["long"], threshold=threshold, horizon=horizon, scaler=scaler
    )
    X_test, y_test, _, _ = add_features(
        test_df, seq_len=window["long"], threshold=threshold, horizon=horizon, scaler=scaler
    )

    # Баланс классов (0=DOWN, 1=UP)
    btrain = np.bincount(y_train, minlength=2)
    bval = np.bincount(y_val, minlength=2)
    btest = np.bincount(y_test, minlength=2)
    n_train = len(y_train)
    pct_down_train = btrain[0] / n_train * 100 if n_train else 0
    pct_up_train = btrain[1] / n_train * 100 if n_train else 0
    print("Train:", btrain, f"  (DOWN {pct_down_train:.1f}% | UP {pct_up_train:.1f}%)")
    print("Val:  ", bval)
    print("Test: ", btest)
    class_weight_cfg = cfg.get("class_weight")
    if class_weight_cfg == "auto":
        n0, n1 = max(btrain[0], 1), max(btrain[1], 1)
        cfg = {**cfg, "class_weight": [n_train / (2 * n0), n_train / (2 * n1)]}
        print("Class weights (auto):", [f"{w:.3f}" for w in cfg["class_weight"]], "  [DOWN, UP]")
    elif abs(pct_down_train - 50) > 5:
        print(
            f"  ⚠ Train class imbalance >5% (DOWN {pct_down_train:.1f}%). "
            "Consider class_weight: auto or class_weight: [w_down, w_up] in config."
        )

    # Тренд в обучающей выборке (вся train-эпоха)
    if "close" in train_df.columns and len(train_df) > 0:
        train_return = (train_df["close"].iloc[-1] - train_df["close"].iloc[0]) / train_df["close"].iloc[0]
        ts_start = train_df["ts"].iloc[0] if "ts" in train_df.columns else "?"
        ts_end = train_df["ts"].iloc[-1] if "ts" in train_df.columns else "?"
        print(f"Train period: {ts_start} → {ts_end}  (total return: {train_return*100:+.2f}%)")

    print("X_train min:", X_train.min())
    print("X_train max:", X_train.max())
    print("X_train mean:", X_train.mean())
    print("X_train std:", X_train.std())
    print("Any NaN:", np.isnan(X_train).any())
    print("Any Inf:", np.isinf(X_train).any())

    # Flatten последовательности для проверки
    X_flat_train = X_train[:, -1, :]  # только последний бар окна
    X_flat_test  = X_test[:, -1, :]

    gb = GradientBoostingClassifier(n_estimators=cfg.get("n_estimators", 100))
    gb.fit(X_flat_train, y_train)
    # pred = gb.predict_proba(X_flat_test)[:, 1]
    # print("GBM ROC-AUC:", roc_auc_score(y_test, pred))
    # print("GBM F1:", f1_score(y_test, pred))
    # print("GBM Precision:", precision_score(y_test, pred))
    # print("GBM Recall:", recall_score(y_test, pred))
    # print("GBM Accuracy:", accuracy_score(y_test, pred))
    # print("GBM Confusion Matrix:", confusion_matrix(y_test, pred))
    # print("GBM Best Val Loss:", gb.best_val_loss_)
    # print("GBM Best Epoch:", gb.best_epoch_)

    proba_up = gb.predict_proba(X_flat_test)[:, 1]
    y_pred = (proba_up >= 0.5).astype(int)

    print("GBM ROC-AUC:", roc_auc_score(y_test, proba_up))
    print("GBM F1:", f1_score(y_test, y_pred))
    print("GBM Precision:", precision_score(y_test, y_pred))
    print("GBM Recall:", recall_score(y_test, y_pred))
    print("GBM Accuracy:", accuracy_score(y_test, y_pred))
    print("GBM Confusion Matrix:", confusion_matrix(y_test, y_pred).tolist())


    # ── Class distribution ────────────────────────────────────────────────────
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        total = len(y_split)
        dist = {int(k): f"{v/total*100:.1f}%" for k, v in zip(unique, counts)}
        print(f"[{split_name}] Class distribution: {dist}  (n={total})")

    # ── Train ─────────────────────────────────────────────────────────────────
    model, train_metrics = train_trend_lstm(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        cfg=cfg,
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n── Test Set Evaluation ──")
    eval_results = evaluate_model(model, X_test, y_test)

    # ── Export ────────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(model_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "features": used_features}, f)

    export_model(
        model=model,
        features=used_features,
        eval_results=eval_results,
        output_dir=output_dir,
        version=timestamp,
        horizon=window["long"],
        threshold=threshold,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"✅ {symbol} | window={window['long']} | threshold={threshold}")
    print(f"📈 Train metrics : {train_metrics['classes']}")
    print(f"📊 Test  metrics : {eval_results}")

    # ── Лог в CSV для сравнения запусков ──────────────────────────────────────
    train_start_ts = str(train_df["ts"].iloc[0]) if "ts" in train_df.columns and len(train_df) else ""
    train_end_ts = str(train_df["ts"].iloc[-1]) if "ts" in train_df.columns and len(train_df) else ""
    train_return_pct = (
        (train_df["close"].iloc[-1] - train_df["close"].iloc[0]) / train_df["close"].iloc[0] * 100
        if "close" in train_df.columns and len(train_df) > 0
        else None
    )
    cm = eval_results.get("confusion_matrix", [])
    cm_str = str(cm) if cm else ""

    row = {
        "timestamp": timestamp,
        "seed": cfg["seed"],
        "window_long": window["long"],
        "horizon": horizon,
        "threshold": threshold,
        "hidden_size": cfg.get("hidden_size", ""),
        "epochs": cfg.get("epochs", ""),
        "patience": cfg.get("patience", ""),
        "weight_decay": cfg.get("weight_decay", ""),
        "dropout_fc": cfg.get("dropout_fc", ""),
        "n_train": n_train,
        "n_val": len(y_val),
        "n_test": len(y_test),
        "train_start": train_start_ts,
        "train_end": train_end_ts,
        "train_return_pct": train_return_pct,
        "accuracy": eval_results.get("accuracy", ""),
        "f1": eval_results.get("f1", ""),
        "precision": eval_results.get("precision", ""),
        "recall": eval_results.get("recall", ""),
        "roc_auc": eval_results.get("roc_auc", ""),
        "confusion_matrix": cm_str,
        "best_val_loss": train_metrics.get("best_val_loss", ""),
        "best_epoch": train_metrics.get("best_epoch", ""),
        "stopped_early": train_metrics.get("stopped_early", ""),
        "learning_rate": cfg.get("learning_rate", ""),
        "num_layers": cfg.get("num_layers", ""),
        "dropout_lstm": cfg.get("dropout_lstm", ""),
        "batch_size": cfg.get("batch_size", ""),
        "class_weight": cfg.get("class_weight", ""),
        "n_estimators": cfg.get("n_estimators", ""),
        "GBM ROC-AUC": roc_auc_score(y_test, proba_up),
        "GBM F1": f1_score(y_test, y_pred),
        "GBM Precision": precision_score(y_test, y_pred),
        "GBM Recall": recall_score(y_test, y_pred),
        "GBM Accuracy": accuracy_score(y_test, y_pred),
        "GBM Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    runs_csv = os.path.join(base_dir, "train_runs.csv")
    file_exists = os.path.isfile(runs_csv)
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📋 Run logged to {runs_csv}")


def main():
    base_dir = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(base_dir, "config.yaml"))

    while True:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        continuous = bool(cfg.get("continuous_training", False))
        resolve_seed_into_cfg(cfg)
        train_one_run(cfg, base_dir)

        if not continuous:
            break


if __name__ == "__main__":
    main()