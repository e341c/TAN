import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import time
import pickle
import onnxruntime as ort
from technical_analysis_trainer.features import compute_rsi
import torch
import numpy as np

timestamp = "2026-01-17T23-33-20"
trendTimestamp = "2026-01-23T13-14-54"

# DATA_PATH = "./data/btcusdt_1h.csv"
SCALER_PATH = f"technical_analysis_trainer/models/{timestamp}/scaler_{timestamp}.pkl"
ONNX_MODEL_PATH = f"technical_analysis_trainer/models/{timestamp}/model_{timestamp}.onnx"

SCALER_TREND_PATH = f"technical_analysis_trainer/trend_model/models/{trendTimestamp}/scaler-{trendTimestamp}.pkl"
ONNX_TREND_MODEL_PATH = f"technical_analysis_trainer/trend_model/models/{trendTimestamp}/model_{trendTimestamp}.onnx"

SEQ_LEN = 30
QUANTILES = [0.1, 0.5, 0.9]
FEATURES = ["open", "high", "low", "close", "log_return", "volume_log", "rsi_scaled"]


async def fetch_last_candles(symbol: str, interval: str, limit: int = 30):
    client = Client()  # public API

    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )

    if not klines:
        raise RuntimeError("No data returned from Binance")

    df = pd.DataFrame(
        klines,
        columns=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        }
    )

    return df


def scaleData(df: pd.DataFrame, scaler_path):
    print("Starting get data")

    df = df.iloc[:-1].reset_index(drop=True)

    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volume_log"] = np.log(df["volume"] + 1)
    df["rsi_scaled"] = df["rsi14"] / 100
    df = df.dropna().reset_index(drop=True)

    # =============================
    # 2. Load scaler
    # =============================
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")

    # =============================
    # 3. Prepare windows
    # =============================
    X_raw = df[FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    X_scaled = X_scaled[-SEQ_LEN:]
    df = df.iloc[-SEQ_LEN:].reset_index(drop=True)

    return X_scaled, df


def quantilePredict(X_window: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
    print("Starting quantile predict")

    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name

    # batch = 1
    X_window = X_window[np.newaxis, ...]

    # (n_quantiles,)
    pred_log_returns = ort_session.run(None, {input_name: X_window})[0][0]

    last_close = df["close"].iloc[-1]

    price_q1 = last_close * np.exp(pred_log_returns[0])
    price_q2 = last_close * np.exp(pred_log_returns[1])
    price_q3 = last_close * np.exp(pred_log_returns[2])

    next_ts = pd.to_datetime(df["ts"].iloc[-1]) + pd.Timedelta(hours=1)

    return {
        "ts": next_ts,
        "q10": price_q1,
        "q50": price_q2,
        "q90": price_q3,
        "log_returns": pred_log_returns,
    }

# TODO: Scale data for this model separeted
def trendPredict(X_window: np.ndarray, df: pd.DataFrame):
    print("Start trend prdict")

    ort_session = ort.InferenceSession(ONNX_TREND_MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name

    X_window = X_window[np.newaxis, ...]

    preds = ort_session.run(None, {input_name: X_window})[0][0]

    print(preds)

    logits = np.array(preds)

    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    print(probs)

    classes = ["DOWN", "FLAT", "UP"]
    pred_class = classes[np.argmax(probs)]
    print(pred_class)

    result = {
        "DOWN": float(probs[0]),
        "FLAT": float(probs[1]),
        "UP": float(probs[2]),
        "predicted_trend": classes[np.argmax(probs)],
    }
    print(result)

    return result


async def getPreds():
    print("Starting fetching data and prediction")

    data = await fetch_last_candles("BTCUSDT", "1h", 45)

    if data is None:
        print("No data fetched")
        return "No data fetched"

    X_window, df = scaleData(data, SCALER_PATH)
    X_window_trend, df_trend = scaleData(data, SCALER_TREND_PATH)

    print(df)

    preds = quantilePredict(X_window, df)

    trendRes = trendPredict(X_window_trend, df_trend)

    return preds, trendRes
