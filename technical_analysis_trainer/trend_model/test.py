"""
Оценка и визуализация сохранённого ONNX+LSTM на тех же данных и фичах, что и train.py.

По умолчанию: config.yaml рядом со скриптом, последняя папка в output_dir, порог/горизонт/окно из конфига.
Пример:
  python -m technical_analysis_trainer.trend_model.test --run 2026-04-18T20-55-59
  python -m technical_analysis_trainer.trend_model.test   # последний run по дате изменения
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.graph_objects as go
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from technical_analysis_trainer.dataset import load_dataset
from technical_analysis_trainer.trend_model.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
    build_features_df,
    resolve_feature_scale_clip,
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(_repo_root(), p)


def _coerce_float64_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").astype(np.float64).to_numpy()


def _raw_ohlc_for_plot(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Свечи с графика: только ts + OHLC из CSV (без фич и без scaler)."""
    need = ("ts", "open", "high", "low", "close")
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise ValueError(f"В data_path нет колонок для свечей: {missing}")
    out = df_raw.loc[:, list(need)].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(subset=["ts"], keep="last")
    return out.sort_values("ts").reset_index(drop=True)


def _ensure_schema_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """Как в train: build_features_df ожидает те же опциональные колонки, иначе список фич не совпадёт со scaler."""
    out = df.copy()
    if "trades" not in out.columns:
        out["trades"] = 1.0
    if "funding_rate" not in out.columns:
        out["funding_rate"] = 0.0
    return out


def _ts_series_for_plotly(ts: pd.Series) -> list[str | None]:
    """
    Ось времени для Plotly: ISO-строки в UTC.

    Числовые x (unix ms) для Candlestick в части версий Plotly/браузеров дают 1970
    или схлопывание; float64 теряет точность на миллисекундах для дат ~2026.
    """
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    out: list[str | None] = []
    for x in t:
        if pd.isna(x):
            out.append(None)
        else:
            out.append(pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return out


def latest_model_run(models_root: str) -> str:
    """Имя подпапки (timestamp) с максимальным mtime scaler_*.pkl."""
    best_t: float = -1.0
    best_name: str | None = None
    if not os.path.isdir(models_root):
        raise FileNotFoundError(f"Нет каталога моделей: {models_root}")
    for name in os.listdir(models_root):
        sub = os.path.join(models_root, name)
        if not os.path.isdir(sub):
            continue
        scaler_glob = os.path.join(sub, f"scaler_{name}.pkl")
        if not os.path.isfile(scaler_glob):
            continue
        mtime = os.path.getmtime(scaler_glob)
        if mtime > best_t:
            best_t = mtime
            best_name = name
    if best_name is None:
        raise FileNotFoundError(f"Не найден ни один scaler_*.pkl в {models_root}")
    return best_name


def load_scaler_bundle(path: str) -> tuple[object, list[str], float, float]:
    with open(path, "rb") as f:
        scaler_data = pickle.load(f)
    if isinstance(scaler_data, dict):
        scaler = scaler_data["scaler"]
        features = list(scaler_data["features"])
        lo = float(scaler_data.get("scale_clip_min", -5.0))
        hi = float(scaler_data.get("scale_clip_max", 5.0))
        return scaler, features, lo, hi
    raise ValueError(
        "Ожидался dict из train.py: {scaler, features, scale_clip_min, scale_clip_max}. "
        "Переобучите модель текущим train.py."
    )


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (logits - np.max(logits)) / max(temperature, 1e-9)
    e = np.exp(z)
    return e / np.sum(e)


def backfill_missing_features_with_scaler_means(
    df_feat: pd.DataFrame,
    features: list[str],
    scaler: object,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Если в df_feat не хватает колонок из features (например, trades-based фичи),
    добавляем их значением, которое после StandardScaler даст 0: scaler.mean_[i].
    Это позволяет запускать инференс на датасетах без некоторых исходных колонок,
    не нарушая размерность входа модели.
    """
    missing = [c for c in features if c not in df_feat.columns]
    if not missing:
        return df_feat, []

    out = df_feat.copy()
    mean = getattr(scaler, "mean_", None)
    feature_index = {f: i for i, f in enumerate(features)}

    for col in missing:
        i = feature_index.get(col)
        if i is not None and mean is not None and i < len(mean):
            fill_value = float(mean[i])
        else:
            fill_value = 0.0
        out[col] = fill_value

    return out, missing


def _load_df_for_inference(
    data_path: str,
    cross_cfg: dict | None,
    mtf_cfg: dict | None,
) -> pd.DataFrame:
    d = load_dataset(data_path)
    if len(d) == 0:
        raise ValueError(f"Пустой датасет после load_dataset: {data_path}")
    d = _ensure_schema_for_features(d)
    if (cross_cfg or {}).get("enabled", False):
        d = attach_cross_market_features(d, cross_cfg)
    if (mtf_cfg or {}).get("enabled", True):
        d = attach_mtf_features(d, mtf_cfg)
    return d


def _df_and_feat_with_add_features(
    data_path: str,
    cross_cfg: dict | None,
    mtf_cfg: dict | None,
    *,
    seq_len: int,
    threshold: float,
    horizon: int,
    scaler: object,
    scale_lo: float,
    scale_hi: float,
    min_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Как train.py + add_features(..., scaler=...).

    Нельзя отключать cross/mtf для «оживления» ряда: число колонок должно совпасть со scaler
    модели. Допустим только сброс cross_market.merge_tolerance → None (колонки xmkt_* те же).
    """
    xc0 = dict(cross_cfg or {})
    mc0 = dict(mtf_cfg or {})
    attempts: list[tuple[dict, dict, str]] = [(xc0, mc0, "как в конфиге")]
    if xc0.get("enabled") and xc0.get("merge_tolerance") not in (None, "", False):
        attempts.append(
            ({**xc0, "merge_tolerance": None}, mc0, "cross merge_tolerance=None"),
        )

    last_err: BaseException | None = None
    for xc, mc, label in attempts:
        d = _load_df_for_inference(data_path, xc, mc)
        try:
            *_, df_feat = add_features(
                d,
                seq_len=seq_len,
                threshold=threshold,
                horizon=horizon,
                scaler=scaler,
                mtf_cfg=mc,
                cross_market_cfg=xc,
                return_df_feat=True,
                clip_low=scale_lo,
                clip_high=scale_hi,
            )
        except ValueError as e:
            last_err = e
            msg = str(e)
            if "0 sample" in msg or "minimum of 1 is required" in msg:
                continue
            raise
        if len(df_feat) < min_rows:
            last_err = ValueError(f"После фичей {len(df_feat)} строк, нужно >= {min_rows}")
            continue
        if label != attempts[0][2]:
            print(f"⚠ Фичи: {label} (чтобы после dropna остались строки; состав колонок как при обучении).")
        return d, df_feat
    raise ValueError(
        "Не удалось собрать фичи: после dropna 0 строк или слишком короткий ряд. "
        "Продлите predict CSV для macro/benchmark или ослабьте cross_market.merge_tolerance в "
        f"test_config.yaml. Последняя ошибка: {last_err!r}"
    ) from last_err


def _df_feat_manual_fallback(
    data_path: str,
    cross_cfg: dict | None,
    mtf_cfg: dict | None,
    min_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Если add_features со scaler не проходит (0 строк), собираем фичи через build_features_df,
    при необходимости отключая cross/mtf. Недостающие колонки под модель добираются backfill-ом
    по means scaler — размерность входа ONNX как при обучении.
    """
    xc0 = dict(cross_cfg or {})
    mc0 = dict(mtf_cfg or {})
    tries: list[tuple[dict, dict, str]] = [(xc0, mc0, "как в конфиге")]
    if xc0.get("enabled") and xc0.get("merge_tolerance") not in (None, "", False):
        tries.append(({**xc0, "merge_tolerance": None}, mc0, "cross merge_tolerance=None"))
    tries.append(({**xc0, "enabled": False}, mc0, "cross выкл."))
    tries.append(({**xc0, "enabled": False}, {**mc0, "enabled": False}, "cross+mtf выкл."))

    first_label = tries[0][2]
    for xc, mc, label in tries:
        d = _load_df_for_inference(data_path, xc, mc)
        df_feat, _ = build_features_df(d, mtf_cfg=mc, cross_market_cfg=xc)
        if len(df_feat) >= min_rows:
            if label != first_label:
                print(f"⚠ Фичи (fallback): {label}")
            return d, df_feat
    raise ValueError(
        "Не удалось получить ни одной строки после build_features_df (dropna). "
        "Проверьте test_config.yaml и длину predict CSV."
    )


def run_evaluation(
    *,
    cfg_path: str,
    run_id: str = "",
    min_confidence: float = 0.55,
    temperature: float = 1.0,
) -> go.Figure:
    """
    Выполнить инференс по сохранённой модели и построить интерактивный график.

    Возвращает Plotly Figure. Метрики печатаются в stdout (как в CLI).
    """
    cfg_path_abs = _resolve_path(cfg_path)
    with open(cfg_path_abs, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models_root = _resolve_path(cfg["output_dir"])
    resolved_run_id = (run_id or "").strip() or latest_model_run(models_root)
    run_dir = os.path.join(models_root, resolved_run_id)
    scaler_path = os.path.join(run_dir, f"scaler_{resolved_run_id}.pkl")
    onnx_path = os.path.join(run_dir, f"model_{resolved_run_id}.onnx")

    print(models_root)
    print(resolved_run_id)
    print(run_dir)
    print(scaler_path)
    print(onnx_path)

    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(scaler_path)
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(onnx_path)

    threshold = float(cfg["threshold"])
    horizon = int(cfg.get("horizon", 1))
    window_cfg = cfg.get("window") or {}
    seq_len = int(window_cfg.get("long", window_cfg.get("mid", 24)))
    symbol = str(cfg.get("symbol", "BTCUSDT"))

    mtf_cfg = cfg.get("mtf")
    cross_cfg = cfg.get("cross_market")

    scaler, features, scale_lo, scale_hi = load_scaler_bundle(scaler_path)
    print(f"Run: {resolved_run_id}")
    print(f"Scaler: {len(features)} features | clip [{scale_lo}, {scale_hi}]")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_path = _resolve_path(cfg["data_path"])
    df = load_dataset(data_path)

    cross_cfg = cfg.get("cross_market")
    if (cross_cfg or {}).get("enabled", False):
        df = attach_cross_market_features(df, cross_cfg)

    # MTF (4H/1D) на полном ряду до split — иначе ресемпл/окна обрежутся на границах сплитов
    mtf_cfg = cfg.get("mtf")
    if (mtf_cfg or {}).get("enabled", True):
        df = attach_mtf_features(df, mtf_cfg)

    # Сначала делаем хронологический split по сырому датафрейму,
    # чтобы scaler обучался только на train-части (нет leakage)

    clip_lo, clip_hi = resolve_feature_scale_clip(cfg)
    print(f"Feature scale clip (after StandardScaler): [{clip_lo}, {clip_hi}]")

    clip_cfg_lo, clip_cfg_hi = resolve_feature_scale_clip(cfg)
    if (clip_cfg_lo, clip_cfg_hi) != (scale_lo, scale_hi):
        print(
            f"⚠ feature_scale_clip в YAML [{clip_cfg_lo}, {clip_cfg_hi}] ≠ "
            f"scaler.pkl [{scale_lo}, {scale_hi}] — для инференса используется pkl."
        )

    min_rows = seq_len + max(horizon, 1) + 1
    try:
        df, df_feat = _df_and_feat_with_add_features(
            data_path,
            cross_cfg,
            mtf_cfg,
            seq_len=seq_len,
            threshold=threshold,
            horizon=horizon,
            scaler=scaler,
            scale_lo=scale_lo,
            scale_hi=scale_hi,
            min_rows=min_rows,
        )
    except ValueError as e:
        if "Не удалось собрать фичи" not in str(e):
            raise
        print("⚠ add_features не удался (0 строк); fallback: build_features_df + backfill.")
        df, df_feat = _df_feat_manual_fallback(
            data_path, cross_cfg, mtf_cfg, min_rows=min_rows
        )

    df_feat, backfilled = backfill_missing_features_with_scaler_means(
        df_feat=df_feat,
        features=features,
        scaler=scaler,
    )
    if backfilled:
        print(
            "⚠ Missing features in data; backfilled with scaler means (→ 0 after scaling): "
            f"{backfilled[:8]}{'…' if len(backfilled) > 8 else ''}"
        )

    if len(df_feat) < min_rows:
        raise ValueError(
            f"Слишком мало строк после фичей ({len(df_feat)}), нужно хотя бы ~{min_rows} "
            f"(seq_len={seq_len}, horizon={horizon}). Проверьте data_path и cross_market в test_config.yaml."
        )

    X_scaled = np.clip(
        scaler.transform(df_feat[features].values.astype(np.float32)),
        scale_lo,
        scale_hi,
    ).astype(np.float32)

    # Согласовано с add_features: окно заканчивается перед баром i → X[i-seq_len : i]
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    pred_indices: list[int] = []
    pred_labels: list[int] = []
    pred_confidences: list[float] = []

    for i in range(seq_len, len(X_scaled)):
        x_input = X_scaled[i - seq_len : i].reshape(1, seq_len, len(features))
        out = ort_session.run(None, {input_name: x_input})[0][0]
        logits = np.asarray(out, dtype=np.float64).ravel()
        probs = softmax_np(logits, temperature)
        pred_indices.append(i)
        pred_labels.append(int(np.argmax(probs)))
        pred_confidences.append(float(np.max(probs)))

    df_preds = df_feat.iloc[pred_indices].copy()
    df_preds["pred_label"] = np.array(pred_labels, dtype=np.int64)
    df_preds["pred_confidence"] = np.array(pred_confidences, dtype=np.float32)

    class_names = {0: "DOWN", 1: "UP"}
    confident_mask = df_preds["pred_confidence"].to_numpy() >= float(min_confidence)
    df_preds["pred"] = df_preds["pred_label"].map(class_names)
    df_preds.loc[~confident_mask, "pred"] = np.nan

    future_return = (df_feat["close"].shift(-horizon) - df_feat["close"]) / df_feat["close"]
    fr_on_preds = future_return.iloc[pred_indices]
    mask = (fr_on_preds.abs() > threshold).to_numpy()
    eval_mask = mask & confident_mask
    df_eval = df_preds[eval_mask].copy()
    fr_eval = fr_on_preds[eval_mask]

    y_true = np.where(fr_eval.to_numpy() > threshold, 1, 0)
    y_pred = df_eval["pred_label"].to_numpy()
    classes = [0, 1]

    n_confident = int(confident_mask.sum())
    print(
        f"Signals: {len(df_preds)} bars with model output, "
        f"{n_confident} with confidence >= {min_confidence}"
    )
    print("Pred distribution (all):", dict(Counter(df_preds["pred_label"].map(class_names))))
    print("Pred distribution (confident):", dict(Counter(df_preds["pred"].dropna())))
    if len(y_true):
        unique, counts = np.unique(y_true, return_counts=True)
        actual_dist = {class_names[int(k)]: int(v) for k, v in zip(unique, counts)}
        print("Actual distribution (eval, non-flat):", actual_dist)
        print("Confusion matrix (0=DOWN, 1=UP):")
        print(confusion_matrix(y_true, y_pred, labels=classes))
        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=[class_names[c] for c in classes],
            )
        )
    else:
        print("Нет строк для метрик (порог/уверенность отфильтровали всё).")

    plot_df = _raw_ohlc_for_plot(df)
    x_plot = _ts_series_for_plotly(plot_df["ts"])

    up_mask = (df_preds["pred_label"] == 1) & confident_mask
    down_mask = (df_preds["pred_label"] == 0) & confident_mask
    hi = pd.to_numeric(df_preds["high"], errors="coerce")
    lo = pd.to_numeric(df_preds["low"], errors="coerce")
    # UP — под свечой (ниже low), DOWN — над свечой (выше high)
    pad = 0.002

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x_plot,
            open=_coerce_float64_series(plot_df["open"]),
            high=_coerce_float64_series(plot_df["high"]),
            low=_coerce_float64_series(plot_df["low"]),
            close=_coerce_float64_series(plot_df["close"]),
            name=symbol,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            opacity=0.9,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_ts_series_for_plotly(df_preds.loc[up_mask, "ts"]),
            y=(lo.loc[up_mask] * (1.0 - pad)).to_numpy(dtype=np.float64),
            mode="markers",
            name=f"UP (→ t+{horizon})",
            marker=dict(symbol="triangle-up", size=10, color="lime"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_ts_series_for_plotly(df_preds.loc[down_mask, "ts"]),
            y=(hi.loc[down_mask] * (1.0 + pad)).to_numpy(dtype=np.float64),
            mode="markers",
            name=f"DOWN (→ t+{horizon})",
            marker=dict(symbol="triangle-down", size=10, color="red"),
        )
    )
    fig.update_layout(
        title=f"{symbol} — ONNX {resolved_run_id} (window={seq_len}, h={horizon}, thr={threshold})",
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(fixedrange=False),
    )

    return fig


def main() -> None:
    base_dir = os.path.dirname(__file__)
    default_cfg = os.path.join(base_dir, "test_config.yaml")

    ap = argparse.ArgumentParser(description="ONNX inference + метрики (как train.py)")
    ap.add_argument(
        "--config",
        default=os.getenv("CONFIG_PATH", default_cfg),
        help="YAML как при обучении (data_path, window, horizon, threshold, mtf, cross_market, …)",
    )
    ap.add_argument(
        "--run",
        default=os.getenv("MODEL_RUN", ""),
        metavar="TIMESTAMP",
        help="Подпапка в output_dir (например 2026-04-18T20-55-59). Пусто = последняя по mtime.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Маркеры и бинарные метрики только при max(softmax) >= этого значения",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Температура softmax (argmax не меняется при T=1)",
    )
    args = ap.parse_args()

    fig = run_evaluation(
        cfg_path=args.config,
        run_id=args.run,
        min_confidence=float(args.min_confidence),
        temperature=float(args.temperature),
    )
    fig.show()


if __name__ == "__main__":
    main()
