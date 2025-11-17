import os
import json
import math
import yaml
import numpy as np
import pandas as pd
import onnxruntime as rt

from dataset import load_dataset
from features_dynamic import add_features_dynamic as add_features


# ==============================
# ⚙️ ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# ==============================

# Путь к основной конфигурации (как в train.py)
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# Путь к конкретной модели и её метаданным
# MODEL_PATH = "./models/long/2025-11-17T21-51-26/model.onnx"
# META_PATH = "./models/long/2025-11-17T21-51-26/meta.json"
# MODEL_PATH = "./models/mid/2025-11-17T21-51-19/model.onnx"
# META_PATH = "./models/mid/2025-11-17T21-51-19/meta.json"
MODEL_PATH = "./models/short/2025-11-17T22-19-08/model.onnx"
META_PATH = "./models/short/2025-11-17T22-19-08/meta.json"

# Стартовый капитал
INITIAL_EQUITY = 1000.0

# Комиссия Binance spot (вход + выход)
FEE_RATE_PER_SIDE = 0.001  # 0.1% = 0.001

# Глобальный override EDGE (в долях, 0.005 = 0.5%).
# Если оставить None → будет автоподбор по горизонту.
EDGE_LOGRET_OVERRIDE = None

# Риск на сделку — доля от капитала
RISK_PER_TRADE = 0.15  # 15% капитала на сделку

# Разрешать ли шорты
ALLOW_SHORT = False


# EDGE и confidence по горизонту (в терминах таргета = %, не log)
# Мы считаем, что модель предсказывает % (например, 1.5 = +1.5%).
EDGE_BY_REGIME = {
    "short": 0.4 / 100.0,   # 0.4% предсказанного движения
    "mid":   0.5 / 100.0,   # 0.5%
    "long":  0.7 / 100.0,   # 0.7%
}

# Минимальная "уверенность" |pred| / MAE
MIN_CONFIDENCE_BY_REGIME = {
    "short": 2.0,
    "mid":   1.0,
    "long":  1.2,
}


# ==============================
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_meta(model_path: str, meta_path: str):
    """
    Загружаем ONNX-модель и meta.json.
    Ожидаем:
      - meta["features"] — список признаков
      - meta["metrics"]["MAE"] — ошибка модели в тех же единицах, что и таргет
      - meta["horizon"] — горизонт модели (в барах)
    """
    meta = json.load(open(meta_path))
    features = meta["features"]

    sess = rt.InferenceSession(model_path)
    return sess, features, meta


def predict_target(sess: rt.InferenceSession, features: list, df_feat: pd.DataFrame, idx: int) -> float:
    """
    Делаем предсказание ТАРГЕТА в тех же единицах, как при обучении.
    Сейчас считаем, что таргет ≈ процентное изменение (например, 1.5 = +1.5%).
    """
    X = df_feat.iloc[[idx]][features].to_numpy().astype(np.float32)
    pred = float(sess.run(None, {"float_input": X})[0][0][0])
    return pred


def compute_max_drawdown(equity_curve: list) -> float:
    """
    maxDD в долях (0.2 = 20%)
    """
    max_eq = -float("inf")
    max_dd = 0.0
    for eq in equity_curve:
        if eq > max_eq:
            max_eq = eq
        dd = (max_eq - eq) / max_eq if max_eq > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def get_regime_and_thresholds(horizon: int, meta: dict):
    """
    По горизонту определяем режим: short / mid / long
    и подбираем EDGE + MIN_CONFIDENCE.
    """
    h = meta.get("horizon", horizon)

    if h <= 1:
        regime = "short"
    elif h <= 12:
        regime = "mid"
    else:
        regime = "long"

    # EDGE: если задан override — используем его (в долях, как и pred_pct)
    if EDGE_LOGRET_OVERRIDE is not None:
        edge_pct = EDGE_LOGRET_OVERRIDE
    else:
        edge_pct = EDGE_BY_REGIME[regime]

    min_conf = MIN_CONFIDENCE_BY_REGIME[regime]

    return regime, h, edge_pct, min_conf


# ==============================
# 🧪 ОСНОВНАЯ ЛОГИКА БЭКТЕСТА
# ==============================

def run_backtest():
    # --- 1️⃣ Конфиг ---
    cfg = load_config(CONFIG_PATH)
    symbol = cfg["symbol"]
    data_path = cfg["data_path"]
    cfg_horizon = cfg.get("horizon", None)

    print(f"⚙️ Backtest for {symbol}, data={data_path}")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📦 Meta:  {META_PATH}")

    # --- 2️⃣ Данные ---
    df_raw = load_dataset(data_path)
    df_feat = add_features(df_raw.copy(), META_PATH)

    df_feat = df_feat.dropna().reset_index(drop=True)

    if "ts" in df_feat.columns:
        df_feat["ts"] = pd.to_datetime(df_feat["ts"])

    if "close" not in df_feat.columns:
        raise ValueError("В данных нет колонки 'close' — бэктест так не сможет работать.")

    # --- 3️⃣ Модель ---
    sess, features, meta = load_model_and_meta(MODEL_PATH, META_PATH)
    model_horizon_raw = meta.get("horizon", cfg_horizon)
    if model_horizon_raw is None:
        raise ValueError("Не найден horizon ни в config.yaml, ни в meta.json")

    regime, horizon, edge_pct, min_conf = get_regime_and_thresholds(model_horizon_raw, meta)
    mae = meta.get("metrics", {}).get("MAE", None)

    print(f"🧠 Using regime={regime}, horizon={horizon} bars.")
    print(f"   EDGE (|pred_pct| >=): {edge_pct * 100:.3f} %")
    if mae is not None:
        print(f"   Model MAE (from meta): {mae:.4f}")
        print(f"   MIN_CONFIDENCE:       {min_conf:.2f} (|pred| / MAE)")
    else:
        print("   ⚠️ WARNING: MAE not found in meta['metrics']['MAE'], confidence фильтр не используется.")
        min_conf = 0.0

    print(f"   RISK_PER_TRADE:       {RISK_PER_TRADE * 100:.1f} % of equity")
    print(f"   ALLOW_SHORT:          {ALLOW_SHORT}")

    # --- 4️⃣ Бэктест ---
    equity = INITIAL_EQUITY
    equity_curve = [equity]
    timestamps = [df_feat["ts"].iloc[0] if "ts" in df_feat.columns else 0]

    trades = []

    i = 0
    n = len(df_feat)

    fee_mult = (1.0 - FEE_RATE_PER_SIDE) ** 2

    while i + horizon < n:
        ts_entry = df_feat["ts"].iloc[i] if "ts" in df_feat.columns else i
        price_entry = float(df_feat["close"].iloc[i])

        # --- предсказание ---
        pred_target = predict_target(sess, features, df_feat, i)
        # считаем, что pred_target ~ проценты (например, 1.5 = +1.5%)
        pred_pct = pred_target / 100.0  # в долях
        # log-return оценка из предсказанного процента
        pred_logret_est = math.log1p(pred_pct)

        # confidence = |pred| / MAE (оба в ед. таргета: %)
        if mae is not None and mae > 0:
            confidence = abs(pred_target) / mae
        else:
            confidence = 0.0

        # --- фильтрация сигналов ---
        direction = 0

        # мало движения по предсказанию
        if abs(pred_pct) < edge_pct:
            direction = 0
        # модель "не уверена"
        elif confidence < min_conf:
            direction = 0
        # сигнал нормальный
        else:
            if pred_pct > 0:
                direction = 1
            elif pred_pct < 0 and ALLOW_SHORT:
                direction = -1
            else:
                direction = 0

        if direction == 0:
            # нет сделки — двигаемся на 1 бар
            i += 1
            # фиксируем equity как есть
            equity_curve.append(equity)
            timestamps.append(df_feat["ts"].iloc[i] if "ts" in df_feat.columns else i)
            continue

        # --- реализация сделки через horizon баров ---
        ts_exit = df_feat["ts"].iloc[i + horizon] if "ts" in df_feat.columns else i + horizon
        price_exit = float(df_feat["close"].iloc[i + horizon])

        if direction == 1:
            real_logret = math.log(price_exit / price_entry)
        else:
            real_logret = math.log(price_entry / price_exit)

        equity_before = equity

        # размер позиции = RISK_PER_TRADE от капитала
        position_equity = equity * RISK_PER_TRADE

        # изменение позиции в мультипликаторе
        position_mult = math.exp(real_logret) * fee_mult

        pnl_pos = position_equity * (position_mult - 1.0)
        equity = equity + pnl_pos

        trade_return = (equity - equity_before) / equity_before

        trades.append(
            {
                "entry_idx": i,
                "exit_idx": i + horizon,
                "entry_time": ts_entry,
                "exit_time": ts_exit,
                "entry_price": price_entry,
                "exit_price": price_exit,
                "direction": "LONG" if direction == 1 else "SHORT",
                "pred_raw": pred_target,
                "pred_pct": pred_pct,
                "pred_logret_est": pred_logret_est,
                "confidence": confidence,
                "real_logret": real_logret,
                "real_pct": math.exp(real_logret) - 1.0,
                "pnl": pnl_pos,
                "return": trade_return,
            }
        )

        equity_curve.append(equity)
        timestamps.append(ts_exit)

        # прыгаем на следующий блок без перекрытия
        i = i + horizon

    # --- 5️⃣ Метрики ---
    if not trades:
        print("❌ Нет сделок по заданным условиям EDGE/CONFIDENCE.")
        return

    trades_df = pd.DataFrame(trades)

    total_return = equity / INITIAL_EQUITY - 1.0
    n_trades = len(trades_df)
    win_trades = trades_df[trades_df["return"] > 0]
    lose_trades = trades_df[trades_df["return"] <= 0]

    winrate = len(win_trades) / n_trades

    gross_profit = win_trades["pnl"].sum()
    gross_loss = -lose_trades["pnl"].sum() if len(lose_trades) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_ret = trades_df["return"].mean()
    med_ret = trades_df["return"].median()

    max_dd = compute_max_drawdown(equity_curve)

    print("\n==============================")
    print(f"📊 BACKTEST RESULTS for {symbol}")
    print("==============================")
    print(f"Начальный депозит:   {INITIAL_EQUITY:.2f} $")
    print(f"Конечный депозит:    {equity:.2f} $")
    print(f"Итоговая доходность: {total_return * 100:.2f} %")
    print(f"Всего сделок:        {n_trades}")
    print(f"Winrate:             {winrate * 100:.2f} %")
    print(f"Profit Factor:       {profit_factor:.3f}")
    print(f"Сред. доходн. сделки:{avg_ret * 100:.3f} %")
    print(f"Медиана сделки:      {med_ret * 100:.3f} %")
    print(f"Max Drawdown:        {max_dd * 100:.2f} %")

    trades_df.to_csv("backtest_trades.csv", index=False)
    eq_df = pd.DataFrame({"ts": timestamps, "equity": equity_curve})
    eq_df.to_csv("backtest_equity.csv", index=False)

    print("\n💾 Сохранено:")
    print("  - backtest_trades.csv")
    print("  - backtest_equity.csv")


if __name__ == "__main__":
    run_backtest()

# import os
# import json
# import math
# import yaml
# import numpy as np
# import pandas as pd
# import onnxruntime as rt

# from dataset import load_dataset
# # from features import add_features
# from features_dynamic import add_features_dynamic as add_features


# # ==============================
# # ⚙️ НАСТРОЙКИ ПОД ТЕБЯ
# # ==============================

# # Путь к основной конфигурации (как в train.py)
# CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")

# # Путь к конкретной модели и её метаданным
# # 👉 сюда подставь актуальные пути, как в visualize_signals.py
# MODEL_PATH = "./models/long/2025-11-17T21-51-26/model.onnx"
# META_PATH = "./models/long/2025-11-17T21-51-26/meta.json"
# # MODEL_PATH = "./models/short/2025-11-04T22-30-06/model.onnx"
# # META_PATH = "./models/short/2025-11-04T22-30-06/meta.json"
# # MODEL_PATH = "./models/mid/2025-11-04T22-30-08/model.onnx"
# # META_PATH = "./models/mid/2025-11-04T22-30-08/meta.json"

# # Стартовый капитал для бэктеста
# INITIAL_EQUITY = 1000.0

# # Комиссия Binance spot: 0.1% на вход + 0.1% на выход
# FEE_RATE_PER_SIDE = 0.001  # 0.1% = 0.001

# # Порог для открытия позиции по предсказанию log-return
# # pred_logret ~ ln(1 + r). Пример: 0.002 ≈ +0.2%
# EDGE_LOGRET = 0.005
# # EDGE_LOGRET = 0.00001

# # Разрешать ли шорты
# ALLOW_SHORT = False


# # ==============================
# # 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# # ==============================

# def load_config(path: str):
#     with open(path) as f:
#         return yaml.safe_load(f)


# def load_model_and_meta(model_path: str, meta_path: str):
#     """
#     Загружаем ONNX-модель и meta.json, как в visualize_signals.py.
#     Ожидаем:
#       - meta["features"] — список признаков
#     """
#     meta = json.load(open(meta_path))
#     features = meta["features"]

#     sess = rt.InferenceSession(model_path)
#     return sess, features, meta


# def predict_logret(sess: rt.InferenceSession, features: list, df_feat: pd.DataFrame, idx: int) -> float:
#     """
#     Делаем предсказание log-return для одной строки df_feat[idx]
#     совместимо с твоим visualize_signals.py:
#         sess.run(None, {"float_input": X})[0][0][0]
#     """
#     X = df_feat.iloc[[idx]][features].to_numpy().astype(np.float32)
#     pred = float(sess.run(None, {"float_input": X})[0][0][0])
#     return pred


# def compute_max_drawdown(equity_curve: list) -> float:
#     """
#     maxDD в долях (0.2 = 20%)
#     """
#     max_eq = -float("inf")
#     max_dd = 0.0
#     for eq in equity_curve:
#         if eq > max_eq:
#             max_eq = eq
#         dd = (max_eq - eq) / max_eq if max_eq > 0 else 0.0
#         if dd > max_dd:
#             max_dd = dd
#     return max_dd


# # ==============================
# # 🧪 ОСНОВНАЯ ЛОГИКА БЭКТЕСТА
# # ==============================

# def run_backtest():
#     # --- 1️⃣ Конфиг ---
#     cfg = load_config(CONFIG_PATH)
#     symbol = cfg["symbol"]
#     data_path = cfg["data_path"]
#     horizon = cfg["horizon"]  # в часах / свечах

#     print(f"⚙️ Backtest for {symbol}, horizon={horizon}, data={data_path}")
#     print(f"📦 Model: {MODEL_PATH}")
#     print(f"📦 Meta:  {META_PATH}")

#     # --- 2️⃣ Данные ---
#     # Используем тот же pipeline что и train.py
#     df_raw = load_dataset(data_path)
#     df_feat = add_features(df_raw.copy(), META_PATH)

#     # Чистим NaN после фич
#     df_feat = df_feat.dropna().reset_index(drop=True)

#     if "ts" in df_feat.columns:
#         df_feat["ts"] = pd.to_datetime(df_feat["ts"])

#     if "close" not in df_feat.columns:
#         raise ValueError("В данных нет колонки 'close' — бэктест так не сможет работать.")

#     # --- 3️⃣ Модель ---
#     sess, features, meta = load_model_and_meta(MODEL_PATH, META_PATH)
#     model_horizon = meta.get("horizon", horizon)

#     if model_horizon != horizon:
#         print(f"⚠️ WARNING: horizon in config={horizon}, in meta={model_horizon}. Используем meta={model_horizon}.")
#         horizon = model_horizon

#     print(f"🧠 Using horizon={horizon} steps (bars) from meta.")

#     # --- 4️⃣ Цикл по истории (неперекрывающиеся трейды) ---
#     equity = INITIAL_EQUITY
#     equity_curve = [equity]
#     timestamps = [df_feat["ts"].iloc[0] if "ts" in df_feat.columns else 0]

#     trades = []  # список словарей с инфой по сделкам

#     i = 0
#     n = len(df_feat)

#     while i + horizon < n:
#         ts_entry = df_feat["ts"].iloc[i] if "ts" in df_feat.columns else i
#         price_entry = float(df_feat["close"].iloc[i])

#         # предсказание модели
#         pred_logret = predict_logret(sess, features, df_feat, i)
#         pred_pct = math.exp(pred_logret) - 1.0

#         # направление по модели
#         direction = 0
#         if pred_logret > EDGE_LOGRET:
#             direction = 1  # long
#         elif pred_logret < -EDGE_LOGRET and ALLOW_SHORT:
#             direction = -1  # short

#         # Если сигнал слабый — просто пропускаем этот интервал, двигаемся дальше
#         if direction == 0:
#             i += 1
#             equity_curve.append(equity)
#             timestamps.append(df_feat["ts"].iloc[i] if "ts" in df_feat.columns else i)
#             continue

#         # --- Реализованный результат через horizon шагов ---
#         ts_exit = df_feat["ts"].iloc[i + horizon] if "ts" in df_feat.columns else i + horizon
#         price_exit = float(df_feat["close"].iloc[i + horizon])

#         if direction == 1:
#             # long: log-return = ln(P_exit / P_entry)
#             real_logret = math.log(price_exit / price_entry)
#         else:
#             # short: log-return = ln(P_entry / P_exit)
#             real_logret = math.log(price_entry / price_exit)

#         # Комиссии (2 стороны)
#         fee_mult = (1.0 - FEE_RATE_PER_SIDE) ** 2

#         # Обновляем капитал (весь капитал в сделке)
#         equity_before = equity
#         equity *= math.exp(real_logret) * fee_mult

#         pnl_abs = equity - equity_before
#         trade_return = equity / equity_before - 1.0

#         trades.append(
#             {
#                 "entry_idx": i,
#                 "exit_idx": i + horizon,
#                 "entry_time": ts_entry,
#                 "exit_time": ts_exit,
#                 "entry_price": price_entry,
#                 "exit_price": price_exit,
#                 "direction": "LONG" if direction == 1 else "SHORT",
#                 "pred_logret": pred_logret,
#                 "pred_pct": pred_pct,
#                 "real_logret": real_logret,
#                 "real_pct": math.exp(real_logret) - 1.0,
#                 "pnl": pnl_abs,
#                 "return": trade_return,
#             }
#         )

#         equity_curve.append(equity)
#         timestamps.append(ts_exit)

#         # прыгаем на следующий блок без перекрытия
#         i = i + horizon

#     # --- 5️⃣ Метрики ---
#     if not trades:
#         print(f"❌ Нет сделок по заданным условиям (EDGE_LOGRET={EDGE_LOGRET} слишком большой?).")
#         return

#     trades_df = pd.DataFrame(trades)

#     total_return = equity / INITIAL_EQUITY - 1.0
#     n_trades = len(trades_df)
#     win_trades = trades_df[trades_df["return"] > 0]
#     lose_trades = trades_df[trades_df["return"] <= 0]

#     winrate = len(win_trades) / n_trades

#     gross_profit = win_trades["pnl"].sum()
#     gross_loss = -lose_trades["pnl"].sum() if len(lose_trades) > 0 else 0.0
#     profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

#     # средняя доходность сделки
#     avg_ret = trades_df["return"].mean()
#     med_ret = trades_df["return"].median()

#     max_dd = compute_max_drawdown(equity_curve)

#     print("\n==============================")
#     print(f"📊 BACKTEST RESULTS for {symbol}")
#     print("==============================")
#     print(f"Начальный депозит:   {INITIAL_EQUITY:.2f} $")
#     print(f"Конечный депозит:    {equity:.2f} $")
#     print(f"Итоговая доходность: {total_return * 100:.2f} %")
#     print(f"Всего сделок:        {n_trades}")
#     print(f"Winrate:             {winrate * 100:.2f} %")
#     print(f"Profit Factor:       {profit_factor:.3f}")
#     print(f"Сред. доходн. сделки:{avg_ret * 100:.3f} %")
#     print(f"Медиана сделки:      {med_ret * 100:.3f} %")
#     print(f"Max Drawdown:        {max_dd * 100:.2f} %")

#     # Можно сохранить результаты в CSV для анализа
#     trades_df.to_csv("backtest_trades.csv", index=False)
#     eq_df = pd.DataFrame({"ts": timestamps, "equity": equity_curve})
#     eq_df.to_csv("backtest_equity.csv", index=False)

#     print("\n💾 Сохранено:")
#     print("  - backtest_trades.csv")
#     print("  - backtest_equity.csv")


# if __name__ == "__main__":
#     run_backtest()