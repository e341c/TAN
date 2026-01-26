import numpy as np
from dataset import load_dataset
from export_onnx import export_model
import yaml, datetime, os
from quantile_model.quantile_model import train_quantile_lstm
from quantile_model.features import add_features
import pickle
from eval import evaluate_model
import pandas as pd
import onnxruntime as ort
from features import compute_rsi


def train(timestamp):
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

    # OLD model train
    # model, train_metrics = train_model(X=X, y=y, cfg=cfg)

    # Export
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

    print(f"✅ {symbol} model trained with window={window} and horizon={horizon}")
    print(f"📈 Train CV metrics: {train_metrics}")
    print(f"📈 Eval metrics: {eval_results}")


def test(timestamp: str):
    print("Init test")

    DATA_PATH = "./data/btcusdt_1h_predict.csv"
    SCALER_PATH = f"models/{timestamp}/scaler_{timestamp}.pkl"
    ONNX_MODEL_PATH = f"models/{timestamp}/model_{timestamp}.onnx"

    SEQ_LEN = 30
    QUANTILES = [0.1, 0.5, 0.9]
    FEATURES = [
        "open",
        "high",
        "low",
        "close",
        "log_return",
        "volume_log",
        "rsi_scaled",
    ]

    # =============================
    # 1. Load data
    # =============================
    df = pd.read_csv(DATA_PATH).copy()

    # Feature engineering (ТОЧНО как в train)
    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_log"] = np.log(df["volume"] + 1)
    df["rsi_scaled"] = df["rsi14"] / 100

    # Убираем NaN (из-за RSI и log_return)
    df = df.dropna().reset_index(drop=True)

    # =============================
    # 2. Load scaler (fit был ТОЛЬКО на train)
    # =============================
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")

    # =============================
    # 3. Prepare input sequence
    # =============================
    X_raw = df[FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    # Берём последние SEQ_LEN свечей
    x_input = X_scaled[-SEQ_LEN:][None, :, :]  # (1, seq_len, n_features)

    # =============================
    # 4. Load ONNX model
    # =============================
    ort_session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name

    # =============================
    # 5. Predict next log-return quantiles
    # =============================
    pred_next = ort_session.run(
        None,
        {input_name: x_input},
    )[
        0
    ]  # shape: (1, n_quantiles)

    # pred_next → это уже Qτ(log_return_{t+1})
    q_log_returns = pred_next[0]

    # =============================
    # 6. Convert log-return quantiles to price quantiles
    # =============================
    last_close = df["close"].iloc[-1]

    price_quantiles = {
        q: last_close * np.exp(q_log_returns[i]) for i, q in enumerate(QUANTILES)
    }

    last_ts = df["ts"].iloc[-1] if "ts" in df.columns else "N/A"

    print(f"\nLast close @ {last_ts}: {last_close:.2f}")
    for q in QUANTILES:
        print(f"Predicted next close (Q{int(q*100)}): {price_quantiles[q]:.2f}")

    return price_quantiles


if __name__ == "__main__":
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    train(timestamp)
    test(timestamp)
