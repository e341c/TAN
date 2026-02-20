import yaml
import datetime
import os
import pickle
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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


def main():
    # ── Config ────────────────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "config.yaml"))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    threshold = cfg["threshold"]
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
        train_df, seq_len=window["long"], threshold=threshold
    )
    X_val, y_val, _, _ = add_features(
        val_df, seq_len=window["long"], threshold=threshold, scaler=scaler
    )
    X_test, y_test, _, _ = add_features(
        test_df, seq_len=window["long"], threshold=threshold, scaler=scaler
    )

    print("Train:", np.bincount(y_train))  # [DOWN_count, UP_count]
    print("Val:  ", np.bincount(y_val))
    print("Test: ", np.bincount(y_test))

    print("X_train min:", X_train.min())
    print("X_train max:", X_train.max())
    print("X_train mean:", X_train.mean())
    print("X_train std:", X_train.std())
    print("Any NaN:", np.isnan(X_train).any())
    print("Any Inf:", np.isinf(X_train).any())

    # Flatten последовательности для проверки
    X_flat_train = X_train[:, -1, :]  # только последний бар окна
    X_flat_test  = X_test[:, -1, :]

    gb = GradientBoostingClassifier(n_estimators=100)
    gb.fit(X_flat_train, y_train)
    pred = gb.predict_proba(X_flat_test)[:, 1]
    print("GBM ROC-AUC:", roc_auc_score(y_test, pred))


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
        pickle.dump(scaler, f)

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


if __name__ == "__main__":
    main()