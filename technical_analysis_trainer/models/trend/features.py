import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from technical_analysis_trainer.core.dataset import load_dataset


from technical_analysis_trainer.features.rsi import compute_rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


# ── Multi-timeframe (4H / 1D), as-of merged onto 1H (ts = candle open time) ──

DEFAULT_MTF: dict = {
    "enabled": True,
    "rsi_period": 14,
    "ret_window_4h": 14,
    "rv_window_4h": 14,
    "vwap_window_4h": 14,
    "ret_window_1d": 14,
    "rv_window_1d": 14,
    "vwap_window_1d": 14,
}


def resolve_mtf_cfg(mtf_cfg: dict | None) -> dict:
    return {**DEFAULT_MTF, **(mtf_cfg or {})}


def mtf_resample_rule(tf_key: str) -> str:
    if tf_key == "4h":
        return "4h"
    if tf_key == "1d":
        return "1D"
    raise ValueError(f"Unknown MTF key: {tf_key!r}")


def collect_mtf_column_names(mtf_cfg: dict | None, *, model_inputs_only: bool = False) -> list[str]:
    """
    Stable MTF column names for scaler / ONNX export.

    `model_inputs_only=True` drops `mtf_*_vwap_*`. That column is a raw VWAP
    *price*, kept because `attach_mtf_features` needs it to derive
    `dist_vwap = close / vwap - 1`, but it must never reach the model: a price
    level rises without bound, so against a scaler frozen at training time its
    z-score only ever grows, dragging the output one way until every prediction
    is the same class. `dist_vwap` carries the same information in stationary
    form and stays in the feature set.
    """
    cfg = resolve_mtf_cfg(mtf_cfg)
    if not cfg.get("enabled", True):
        return []
    rsi_p = int(cfg["rsi_period"])
    cols: list[str] = []
    for tf_key in ("4h", "1d"):
        ret_w = int(cfg[f"ret_window_{tf_key}"])
        rv_w = int(cfg[f"rv_window_{tf_key}"])
        vw_w = int(cfg[f"vwap_window_{tf_key}"])
        prefix = f"mtf_{tf_key}"
        cols.extend(
            [
                f"{prefix}_rsi_{rsi_p}",
                f"{prefix}_ret_{ret_w}",
                f"{prefix}_rv_{rv_w}",
            ]
        )
        if not model_inputs_only:
            cols.append(f"{prefix}_vwap_{vw_w}")
        cols.append(f"{prefix}_dist_vwap_{vw_w}")
    return cols


def _has_all_mtf_columns(df: pd.DataFrame, mtf_cfg: dict | None) -> bool:
    return all(c in df.columns for c in collect_mtf_column_names(mtf_cfg))


def compute_realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Realized vol over `window` bars: std(log returns) * sqrt(window)."""
    lr = np.log(close / (close.shift(1) + 1e-12))
    return lr.rolling(window=window, min_periods=window).std() * np.sqrt(float(window))


def compute_rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    tp = (high + low + close) / 3.0
    pv = (tp * volume).rolling(window=window, min_periods=window).sum()
    vsum = volume.rolling(window=window, min_periods=window).sum()
    return pv / (vsum + 1e-12)


def resample_ohlcv(df_indexed: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate 1H bars to higher TF. Index must be candle open time (UTC).

    label='right', closed='right' → index = right edge (last included 1H open),
    i.e. time when the higher-TF bucket is complete for as-of merge onto 1H `ts`.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df_indexed.sort_index().resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def _higher_tf_indicator_df(
    h: pd.DataFrame,
    col_prefix: str,
    rsi_period: int,
    ret_window: int,
    rv_window: int,
    vwap_window: int,
) -> pd.DataFrame:
    c = h["close"]
    rsi = compute_rsi(c, period=rsi_period)
    ret_n = c / (c.shift(ret_window) + 1e-12) - 1.0
    rv = compute_realized_vol(c, rv_window)
    vwap = compute_rolling_vwap(h["high"], h["low"], h["close"], h["volume"], vwap_window)
    return pd.DataFrame(
        {
            f"{col_prefix}_rsi_{rsi_period}": rsi,
            f"{col_prefix}_ret_{ret_window}": ret_n,
            f"{col_prefix}_rv_{rv_window}": rv,
            f"{col_prefix}_vwap_{vwap_window}": vwap,
        },
        index=h.index,
    )


def attach_mtf_features(df: pd.DataFrame, mtf_cfg: dict | None = None) -> pd.DataFrame:
    """
    Adds 4H/1D features aligned to each 1H row without lookahead.

    Expects `ts` = candle **open** time (Binance). Higher-TF bars are labeled at
    bucket right edge; merge_asof(..., direction='backward') attaches only
    completed higher-TF bars.

    Call on the **full** history before train/val/test split so rolling/resample
    statistics are not truncated at split boundaries.
    """
    cfg = resolve_mtf_cfg(mtf_cfg)
    if not cfg.get("enabled", True):
        return df

    out = df.copy()
    if "ts" not in out.columns:
        raise ValueError("attach_mtf_features requires a 'ts' column")
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    need = ["open", "high", "low", "close", "volume"]
    for c in need:
        if c not in out.columns:
            raise ValueError(f"attach_mtf_features missing column {c!r}")

    rsi_p = int(cfg["rsi_period"])
    df_work = out

    for tf_key in ("4h", "1d"):
        rule = mtf_resample_rule(tf_key)
        ret_w = int(cfg[f"ret_window_{tf_key}"])
        rv_w = int(cfg[f"rv_window_{tf_key}"])
        vw_w = int(cfg[f"vwap_window_{tf_key}"])
        prefix = f"mtf_{tf_key}"

        idxed = df_work.set_index("ts")[need].sort_index()
        h = resample_ohlcv(idxed, rule)
        if h.empty:
            continue

        feats = _higher_tf_indicator_df(h, prefix, rsi_p, ret_w, rv_w, vw_w)
        feats = feats.dropna(how="all")
        feats = feats.reset_index()
        bar_col = feats.columns[0]
        feats = feats.rename(columns={bar_col: "_mtf_bar_end"})

        merged = pd.merge_asof(
            df_work.sort_values("ts"),
            feats.sort_values("_mtf_bar_end"),
            left_on="ts",
            right_on="_mtf_bar_end",
            direction="backward",
        )
        merged = merged.drop(columns=["_mtf_bar_end"], errors="ignore")
        vwap_col = f"{prefix}_vwap_{vw_w}"
        dist_col = f"{prefix}_dist_vwap_{vw_w}"
        merged[dist_col] = merged["close"] / (merged[vwap_col] + 1e-12) - 1.0
        df_work = merged.sort_values("ts").reset_index(drop=True)

    return df_work


# ── Cross-market (benchmarks / macro merged on main ts, no lookahead) ─────────

DEFAULT_CROSS_MARKET: dict = {
    "enabled": False,
    "merge_tolerance": None,
    "benchmarks": {},
    "macro": {},
    "corr": {"benchmark": "eth", "windows": [24, 72, 168]},
    "spread_main_minus": ["eth"],
}


def resolve_cross_cfg(cross_cfg: dict | None) -> dict:
    if not cross_cfg:
        return {**DEFAULT_CROSS_MARKET}
    out = {**DEFAULT_CROSS_MARKET, **cross_cfg}
    out["benchmarks"] = {**DEFAULT_CROSS_MARKET["benchmarks"], **(cross_cfg.get("benchmarks") or {})}
    out["macro"] = {**DEFAULT_CROSS_MARKET["macro"], **(cross_cfg.get("macro") or {})}
    out["corr"] = {**DEFAULT_CROSS_MARKET["corr"], **(cross_cfg.get("corr") or {})}
    if out.get("spread_main_minus") is None:
        out["spread_main_minus"] = list(DEFAULT_CROSS_MARKET["spread_main_minus"])
    return out


def _cross_effective_path(entry: dict | str | None) -> str | None:
    if entry is None:
        return None
    if isinstance(entry, str):
        p = entry.strip()
        return p or None
    if isinstance(entry, dict):
        p = (entry.get("path") or "").strip()
        return p or None
    return None


def _cross_source_keys_with_paths(sources: dict) -> list[str]:
    """Keys whose configured path is non-empty and the file exists (matches what attach can load)."""
    keys: list[str] = []
    for k in sorted(sources.keys()):
        p = _cross_effective_path(sources.get(k))
        if p and os.path.isfile(p):
            keys.append(str(k))
    return keys


def collect_cross_market_feature_names(cross_cfg: dict | None) -> list[str]:
    """Stable feature names for scaler / ONNX (only sources with non-empty path)."""
    cfg = resolve_cross_cfg(cross_cfg)
    if not cfg.get("enabled", False):
        return []
    names: list[str] = []
    bench = cfg.get("benchmarks") or {}
    macro = cfg.get("macro") or {}
    for k in _cross_source_keys_with_paths(bench):
        names.append(f"xmkt_log_ret_{k}")
    for k in _cross_source_keys_with_paths(macro):
        names.append(f"xmkt_log_ret_{k}")
    loaded_spread_keys = set(_cross_source_keys_with_paths(bench)) | set(_cross_source_keys_with_paths(macro))
    for sm in cfg.get("spread_main_minus") or []:
        ks = str(sm)
        if ks in loaded_spread_keys:
            names.append(f"xmkt_log_ret_main_minus_{ks}")
    corr = cfg.get("corr") or {}
    bench_name = corr.get("benchmark")
    loaded_corr_keys = set(_cross_source_keys_with_paths(bench)) | set(_cross_source_keys_with_paths(macro))
    if bench_name is not None:
        bn = str(bench_name)
        if bn in loaded_corr_keys:
            for w in corr.get("windows") or [24, 72, 168]:
                names.append(f"xmkt_corr_main_{bn}_{int(w)}")
    return names


def _has_all_cross_market_columns(df: pd.DataFrame, cross_cfg: dict | None) -> bool:
    cfg = resolve_cross_cfg(cross_cfg)
    if not cfg.get("enabled", False):
        return True
    return all(c in df.columns for c in collect_cross_market_feature_names(cross_cfg))


def _cross_merge_tolerance(cfg: dict) -> pd.Timedelta | None:
    raw = cfg.get("merge_tolerance")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return None
    return pd.Timedelta(raw)


def attach_cross_market_features(df: pd.DataFrame, cross_cfg: dict | None) -> pd.DataFrame:
    """
    Merges external benchmark/macro closes onto the main OHLCV row index (ts = open time).
    Policy after merge_asof(backward): ffill then bfill on merged closes so low-frequency
    macro aligns with 1H steps; same policy at inference as in training.

    Computes xmkt_log_ret_<key>, xmkt_log_ret_main_minus_<key>, xmkt_corr_main_<bench>_<w>.
    Raw merged close columns are dropped before return (not model inputs).
    """
    cfg = resolve_cross_cfg(cross_cfg)
    if not cfg.get("enabled", False):
        return df.copy()

    out = df.copy()
    if "ts" not in out.columns:
        raise ValueError("attach_cross_market_features requires a 'ts' column")
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if "close" not in out.columns:
        raise ValueError("attach_cross_market_features requires 'close' on the main frame")

    tolerance = _cross_merge_tolerance(cfg)
    merge_kwargs: dict = {"on": "ts", "direction": "backward"}
    if tolerance is not None:
        merge_kwargs["tolerance"] = tolerance

    raw_close_cols: list[str] = []
    loaded_keys: list[tuple[str, str]] = []  # (kind, key) kind in bench|macro

    def merge_one(path: str, col_raw: str, label: str) -> bool:
        nonlocal out
        if not os.path.isfile(path):
            print(f"⚠ cross_market: missing file ({label}) {path!r} — skipped")
            return False
        try:
            aux = load_dataset(path)
        except Exception as e:
            print(f"⚠ cross_market: could not load {path!r}: {e}")
            return False
        if "close" not in aux.columns:
            print(f"⚠ cross_market: no 'close' in {path!r} — skipped")
            return False
        right = aux[["ts", "close"]].copy()
        right["ts"] = pd.to_datetime(right["ts"], utc=True, errors="coerce")
        right = right.dropna(subset=["ts"]).sort_values("ts")
        right = right.rename(columns={"close": col_raw})
        out = out.sort_values("ts")
        out = pd.merge_asof(out, right, **merge_kwargs)
        raw_close_cols.append(col_raw)
        return True

    bench = cfg.get("benchmarks") or {}
    for key in sorted(bench.keys()):
        p = _cross_effective_path(bench.get(key))
        if not p:
            continue
        col_raw = f"xmkt_raw_close_bench_{key}"
        if merge_one(p, col_raw, f"benchmark:{key}"):
            loaded_keys.append(("bench", str(key)))

    macro = cfg.get("macro") or {}
    for key in sorted(macro.keys()):
        p = _cross_effective_path(macro.get(key))
        if not p:
            continue
        col_raw = f"xmkt_raw_close_macro_{key}"
        if merge_one(p, col_raw, f"macro:{key}"):
            loaded_keys.append(("macro", str(key)))

    for c in raw_close_cols:
        out[c] = out[c].ffill().bfill()

    lr_main = np.log(out["close"] / (out["close"].shift(1) + 1e-12))

    bench_keys_loaded = {k for kind, k in loaded_keys if kind == "bench"}
    macro_keys_loaded = {k for kind, k in loaded_keys if kind == "macro"}
    all_loaded = bench_keys_loaded | macro_keys_loaded

    for kind, key in loaded_keys:
        col_raw = f"xmkt_raw_close_bench_{key}" if kind == "bench" else f"xmkt_raw_close_macro_{key}"
        if col_raw not in out.columns:
            continue
        s = out[col_raw].astype(float)
        # If the stitched window has no effective data for a given macro/benchmark
        # series, `merge_asof` may leave the entire column NaN (even after ffill/bfill).
        # We don't want `build_features_df(...).dropna()` to wipe the whole frame, so
        # keep a neutral fallback (0.0) when cross-market data is missing.
        lr = np.log(s / (s.shift(1) + 1e-12))
        lr = lr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"xmkt_log_ret_{key}"] = lr

    for key in cfg.get("spread_main_minus") or []:
        ks = str(key)
        if ks not in all_loaded:
            continue
        col_ret = f"xmkt_log_ret_{ks}"
        if col_ret not in out.columns:
            continue
        sm = lr_main - out[col_ret]
        sm = sm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[f"xmkt_log_ret_main_minus_{ks}"] = sm

    corr = cfg.get("corr") or {}
    bench_corr = corr.get("benchmark")
    windows = corr.get("windows") or [24, 72, 168]
    if bench_corr is not None:
        bn = str(bench_corr)
        col_ret = f"xmkt_log_ret_{bn}"
        if col_ret in out.columns:
            lr_b = out[col_ret]
            for w in windows:
                w = int(w)
                corr = lr_main.rolling(window=w, min_periods=w).corr(lr_b)
                out[f"xmkt_corr_main_{bn}_{w}"] = corr.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)

    out = out.drop(columns=raw_close_cols, errors="ignore")
    return out


def build_features_df(
    df: pd.DataFrame,
    mtf_cfg: dict | None = None,
    cross_market_cfg: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Построение датафрейма с техническими фичами без масштабирования и окон.

    Используется и при обучении (train), и при инференсе / тестах,
    чтобы гарантировать одинаковый пайплайн признаков.

    mtf_cfg:
        Параметры мульти‑таймфрейма (4H/1D). Если колонки MTF ещё не добавлены
        (например, инференс на одном CSV), при enabled=True будет вызван
        `attach_mtf_features`. Для обучения рекомендуется вызывать
        `attach_mtf_features` на полном ряду **до** train/val/test split
        (см. `train.py`), чтобы не обрезать историю для ресемпла/окон.

    cross_market_cfg:
        Кросс‑рынок (см. `attach_cross_market_features`). На полном ряду вызывать
        из `train.py` до MTF и split; здесь — догрузка, если колонок ещё нет.
    """
    resolved_cross = resolve_cross_cfg(cross_market_cfg)
    did_attach = False
    if resolved_cross.get("enabled", False) and not _has_all_cross_market_columns(df, cross_market_cfg):
        df = attach_cross_market_features(df, cross_market_cfg)
        did_attach = True

    resolved_mtf = resolve_mtf_cfg(mtf_cfg)
    if resolved_mtf.get("enabled", True) and not _has_all_mtf_columns(df, resolved_mtf):
        df = attach_mtf_features(df, mtf_cfg)
        did_attach = True

    if not did_attach:
        df = df.copy()

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    mtf_feature_list = [
        c
        for c in collect_mtf_column_names(resolved_mtf, model_inputs_only=True)
        if c in df.columns
    ]
    cross_feature_list = [
        c for c in collect_cross_market_feature_names(cross_market_cfg) if c in df.columns
    ]

    # ── Технические индикаторы ───────────────────────────────────────────────
    df["rsi14"] = compute_rsi(df["close"], period=14)
    df["atr14"] = compute_atr(df, period=14)
    # Скорость изменения цены (Rate of Change, 1 бар)
    df["price_roc"] = df["close"].pct_change()

    # Полосы Боллинджера (Bollinger Bands)
    bb_period = 20
    bb_k = 2.0
    bb_mid = df["close"].rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std = df["close"].rolling(window=bb_period, min_periods=bb_period).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + bb_k * bb_std
    df["bb_lower"] = bb_mid - bb_k * bb_std

    # Сигналы по M, RSI и Bollinger Bands
    m = df["price_roc"]
    df["M_signal"] = np.where(m > 0, 1, np.where(m < 0, -1, 0))
    df["RSI_signal"] = np.where(df["rsi14"] < 30, 1, np.where(df["rsi14"] > 70, -1, 0))
    df["BB_signal"] = np.where(
        df["close"] < df["bb_lower"],
        1,
        np.where(df["close"] > df["bb_upper"], -1, 0),
    )

    # Итоговый упрощённый Score силы сигнала
    df["score"] = df["M_signal"] + df["RSI_signal"] + df["BB_signal"]
    df["score_norm"] = df["score"] / 3.0

    # ── Моментум на разных окнах (важно для horizon=3) ───────────────────────
    df["roc_3"] = df["close"].pct_change(3)
    df["roc_6"] = df["close"].pct_change(6)
    df["roc_12"] = df["close"].pct_change(12)

    # ── MACD (тренд и смена тренда) ──────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    # Нормализуем по цене, чтобы не доминировали по масштабу
    df["macd_hist_norm"] = df["macd_hist"] / (df["close"] + 1e-9)

    # ── Позиция внутри Bollinger (перекупленность / перепроданность) ─────────
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_mid"]) / (bb_range + 1e-9)

    # ── Сила свечи: body / range (-1..1) ─────────────────────────────────────
    body = df["close"] - df["open"]
    candle_range = df["high"] - df["low"]
    df["body_ratio"] = body / (candle_range + 1e-9)

    # ── Режим волатильности: ATR сейчас vs недавнее среднее ──────────────────
    atr_ma24 = df["atr14"].rolling(window=24, min_periods=1).mean()
    df["atr_ratio"] = df["atr14"] / (atr_ma24 + 1e-9)

    # ── Расстояние от недавнего high/low (нормализовано ATR) ─────────────────
    roll_high_24 = df["high"].rolling(window=24, min_periods=1).max()
    roll_low_24 = df["low"].rolling(window=24, min_periods=1).min()
    df["dist_from_high"] = (df["close"] - roll_high_24) / (df["atr14"] + 1e-9)
    df["dist_from_low"] = (df["close"] - roll_low_24) / (df["atr14"] + 1e-9)

    # ── Нормализованные ценовые фичи ─────────────────────────────────────────
    df["high_ratio"] = df["high"] / df["close"] - 1
    df["low_ratio"] = df["low"] / df["close"] - 1
    df["open_ratio"] = df["open"] / df["close"] - 1
    df["atr_norm"] = df["atr14"] / df["close"]
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    vol_ma = df["volume"].rolling(window=24, min_periods=1).mean()
    df["volume_norm"] = df["volume"] / (vol_ma + 1e-9)
    df["rsi_scaled"] = df["rsi14"] / 100

    # ── Liquidity / flow-shape features (rolling 24/72/168 on 1H) ───────────
    eps = 1e-12
    flow_windows = (24, 72, 168)
    # Reference window for de-trending the level features below. See
    # `_detrend`: 168 bars (7 days) is chosen so total warm-up (window + 168)
    # stays at ~336 bars, which the 1D MTF features already require — the fix
    # costs no extra history at inference.
    flow_norm_window = 168

    def _detrend(level: pd.Series) -> pd.Series:
        """
        Turn an absolute level into a z-score against its own recent history.

        Amihud illiquidity, range-per-volume and average trade size are all
        *absolute* quantities: they carry the dollar price, the base-unit volume
        and the trade count of the moment, none of which are comparable across
        regimes. Their raw level therefore trends with the market rather than
        oscillating, and a StandardScaler frozen at training time cannot follow
        it — the z-scores keep marching in one direction long after the fit.

        This is not hypothetical. In the trend run served in Aug 2026 these nine
        features had drifted 1.3-1.9 training sd by the end of the model's own
        test fold, which biased p(UP) from 0.502 on train to 0.544 on test and
        left argmax picking UP on 95% of test bars and 100% of live ones. The
        model had not stopped discriminating (ROC-AUC 0.527); its operating
        point had walked off the data.

        A rolling z-score is stationary by construction, so the same drift can
        no longer accumulate, and it keeps the part that actually carries signal
        — how illiquid/large flow is *relative to the recent regime*, which is
        the same normalisation `vol_z_{w}` already applies to volume.
        """
        ref_mean = level.rolling(window=flow_norm_window, min_periods=flow_norm_window).mean()
        ref_std = level.rolling(window=flow_norm_window, min_periods=flow_norm_window).std()
        return (level - ref_mean) / (ref_std + eps)

    # volume z-score vs its own distribution
    for w in flow_windows:
        v_mean = df["volume"].rolling(window=w, min_periods=w).mean()
        v_std = df["volume"].rolling(window=w, min_periods=w).std()
        df[f"vol_z_{w}"] = (df["volume"] - v_mean) / (v_std + eps)

    # Amihud-like illiquidity: |return| / volume (then rolled)
    abs_lr = df["log_return"].abs()
    amihud_1 = abs_lr / (df["volume"] + eps)
    for w in flow_windows:
        df[f"amihud_{w}"] = _detrend(amihud_1.rolling(window=w, min_periods=w).mean())

    # Range per volume: (high-low)/volume (then rolled)
    rpv_1 = (df["high"] - df["low"]) / (df["volume"] + eps)
    for w in flow_windows:
        df[f"range_per_vol_{w}"] = _detrend(rpv_1.rolling(window=w, min_periods=w).mean())

    # Avg trade size: volume / trades (then rolled) if trades exist
    has_trades = "trades" in df.columns
    if has_trades:
        trades = df["trades"].astype(float)
        avg_trade_1 = df["volume"] / (trades + eps)
        for w in flow_windows:
            df[f"avg_trade_size_{w}"] = _detrend(
                avg_trade_1.rolling(window=w, min_periods=w).mean()
            )
        trade_features = [f"avg_trade_size_{w}" for w in flow_windows]
    else:
        trade_features = []

    # ── Funding rate ─────────────────────────────────────────────────────────
    # Выходит каждые 8H — уже замёрджен через forward fill в fetch_data.py
    # Типичный диапазон: -0.01..+0.01, умножаем на 100 → -1..+1
    has_funding = "funding_rate" in df.columns
    if has_funding:
        df["funding_rate_scaled"] = df["funding_rate"] * 100
        # Накопленное давление за последние 3 выплаты (= 24H)
        # Положительное значение = лонги переплачивают шортам (медвежий сигнал)
        df["funding_rate_cum3"] = (
            df["funding_rate"].rolling(window=3, min_periods=1).sum() * 100
        )
        funding_features = ["funding_rate_scaled", "funding_rate_cum3"]
    else:
        funding_features = []

    # ── Временны́е фичи ───────────────────────────────────────────────────────
    if "ts" in df.columns:
        dt = pd.to_datetime(df["ts"])
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        time_features = []

    # Удаляем строки с NaN (первые бары из-за индикаторов)
    df = df.dropna().reset_index(drop=True)

    features = [
        "open_ratio",
        "high_ratio",
        "low_ratio",
        "price_roc",
        "log_return",
        "volume_norm",
        *[f"vol_z_{w}" for w in (24, 72, 168)],
        *[f"amihud_{w}" for w in (24, 72, 168)],
        *[f"range_per_vol_{w}" for w in (24, 72, 168)],
        *trade_features,
        "rsi_scaled",
        "atr_norm",
        "score_norm",
        # Моментум и тренд
        "roc_3",
        "roc_6",
        "roc_12",
        "macd_hist_norm",
        "bb_position",
        "body_ratio",
        "atr_ratio",
        "dist_from_high",
        "dist_from_low",
        *mtf_feature_list,
        *cross_feature_list,
        *funding_features,
        *time_features,
    ]

    return df, features


def resolve_feature_scale_clip(cfg: dict | None) -> tuple[float, float]:
    """
    Clip bounds after StandardScaler (in units of train σ).

    Config forms:
      - feature_scale_clip: 5   → symmetric [-5, 5]
      - feature_scale_clip: { min: -5, max: 5 }
    """
    if not cfg:
        return -5.0, 5.0
    raw = cfg.get("feature_scale_clip", 5.0)
    if isinstance(raw, dict):
        lo = float(raw.get("min", -5.0))
        hi = float(raw.get("max", 5.0))
        if lo >= hi:
            raise ValueError(f"feature_scale_clip: need min < max, got {lo=} {hi=}")
        return lo, hi
    try:
        v = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"feature_scale_clip must be a positive number or {{min,max}} dict, got {raw!r}") from e
    if v <= 0:
        raise ValueError(f"feature_scale_clip as number must be > 0, got {v}")
    return (-v, v)


def triple_barrier_labels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    sigma: np.ndarray,
    upper_mult: float = 1.5,
    lower_mult: float = 1.5,
    max_holding: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triple-barrier labelling (López de Prado): does price touch +k*sigma before
    -k*sigma within `max_holding` bars?

    Why this beats next-bar sign: a single bar's close is almost pure noise, and
    the trend run log shows 184 attempts topping out at ~0.53 ROC-AUC because of
    it. A barrier label aggregates over a whole path, so the signal-to-noise is
    far better, and it maps directly onto how the prediction would be traded
    (a target and a stop) instead of onto an unusable one-bar sign.

    Barriers are volatility-scaled, so a "move" means the same thing in calm and
    violent regimes.

    Returns (labels, spans, realized):
      labels  1 = upper hit first, 0 = lower hit first, -1 = unusable
      spans   bars until the touch (or max_holding)
      realized log return over the label's span
    """
    n = len(close)
    labels = np.full(n, -1, dtype=np.int64)
    spans = np.zeros(n, dtype=np.int64)
    realized = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s = sigma[i]
        if not np.isfinite(s) or s <= 0:
            continue
        end = min(i + max_holding, n - 1)
        if end <= i:
            break

        up = close[i] * np.exp(upper_mult * s)
        dn = close[i] * np.exp(-lower_mult * s)

        lab, touch = -1, end
        for j in range(i + 1, end + 1):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn
            if hit_up and hit_dn:
                # Both barriers touched inside the same bar — OHLC cannot say
                # which came first. Labelling it either way injects a directional
                # bias, so the sample is dropped.
                lab, touch = -1, j
                break
            if hit_up:
                lab, touch = 1, j
                break
            if hit_dn:
                lab, touch = 0, j
                break

        labels[i] = lab
        spans[i] = touch - i
        realized[i] = np.log(close[touch] / close[i])

    return labels, spans, realized


def average_uniqueness(spans: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Per-label average uniqueness (López de Prado).

    Triple-barrier labels overlap heavily — a 24-bar span means ~24 consecutive
    labels share most of their outcome. Treating them as independent inflates
    the effective sample size and makes every significance test too optimistic.
    Down-weighting by uniqueness corrects for that.
    """
    n = len(spans)
    concurrency = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        if not valid[i]:
            continue
        concurrency[i : min(i + max(spans[i], 1), n)] += 1.0

    uniq = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not valid[i]:
            continue
        end = min(i + max(spans[i], 1), n)
        c = concurrency[i:end]
        c = np.where(c > 0, c, 1.0)
        uniq[i] = float(np.mean(1.0 / c))
    return uniq


def add_features(
    df: pd.DataFrame,
    seq_len: int = 30,
    threshold: float = 0.002,
    horizon: int = 3,
    scaler: StandardScaler | None = None,
    mtf_cfg: dict | None = None,
    cross_market_cfg: dict | None = None,
    return_df_feat: bool = False,
    clip_low: float = -5.0,
    clip_high: float = 5.0,
    target_mode: str = "next_bar",
    tb_cfg: dict | None = None,
    return_weights: bool = False,
):
    """
    Feature engineering pipeline for OHLCV time series (BTCUSDT 1H).

    Входные колонки: ts, open, high, low, close, volume, funding_rate (опционально)
    Выходные данные:
        X        — массив последовательностей формы (N, seq_len, n_features)
        Y        — метки классов: 1=UP, 0=DOWN (flat-бары исключены)
        features — список имён фичей
        scaler   — обученный StandardScaler (fit только на train-данных)

    ВАЖНО: передавайте scaler=None только для train-сплита.
    Для val/test передавайте обученный scaler, чтобы не было data leakage.

    mtf_cfg — опционально; см. `build_features_df(..., mtf_cfg=...)`.
    cross_market_cfg — опционально; см. `build_features_df(..., cross_market_cfg=...)`.

    return_df_feat:
        Если True, возвращается пятый элемент — `df_feat` после `build_features_df`
        (до построения окон; для аудита scaler / распределений).

    clip_low, clip_high:
        После `StandardScaler` значения режутся в [clip_low, clip_high]
        (см. `resolve_feature_scale_clip` и `config.yaml` → `feature_scale_clip`).
    """
    if clip_low >= clip_high:
        raise ValueError(f"add_features: clip_low must be < clip_high, got {clip_low=} {clip_high=}")
    df_feat, features = build_features_df(
        df, mtf_cfg=mtf_cfg, cross_market_cfg=cross_market_cfg
    )

    # ── Target ───────────────────────────────────────────────────────────────
    future_return = (df_feat["close"].shift(-horizon) - df_feat["close"]) / df_feat["close"]

    n_rows = len(df_feat)
    if target_mode == "triple_barrier":
        tb = tb_cfg or {}
        vol_window = int(tb.get("vol_window", 72))
        max_holding = int(tb.get("max_holding", 24))
        lr = np.log(
            df_feat["close"].astype(float) / df_feat["close"].astype(float).shift(1)
        )
        sigma = lr.rolling(vol_window, min_periods=vol_window).std().to_numpy()
        labels, spans, realized = triple_barrier_labels(
            df_feat["close"].to_numpy(dtype=np.float64),
            df_feat["high"].to_numpy(dtype=np.float64),
            df_feat["low"].to_numpy(dtype=np.float64),
            sigma,
            upper_mult=float(tb.get("upper_mult", 1.5)),
            lower_mult=float(tb.get("lower_mult", 1.5)),
            max_holding=max_holding,
        )
        valid_label = labels >= 0
        uniq = average_uniqueness(spans, valid_label)
        # Weight = how much the label is worth (realised move) x how much new
        # information it carries (uniqueness among overlapping labels).
        raw_w = np.abs(realized) * uniq
        label_end = np.arange(n_rows) + np.maximum(spans, 1)
    else:
        labels = np.where(
            future_return.to_numpy() > threshold,
            1,
            np.where(future_return.to_numpy() < -threshold, 0, -1),
        )
        valid_label = np.abs(future_return.to_numpy()) > threshold
        # A 0.02% bar and a 2% bar are not equally informative; weight by size.
        raw_w = np.abs(future_return.to_numpy())
        label_end = np.arange(n_rows) + horizon

    # ── Масштабирование ───────────────────────────────────────────────────────
    X_raw = df_feat[features].values

    if scaler is None:
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(X_raw)
    else:
        X_raw = scaler.transform(X_raw)

    X_raw = np.clip(X_raw, clip_low, clip_high)

    # ── Построение последовательностей ───────────────────────────────────────
    X, Y, W = [], [], []
    for i in range(seq_len, n_rows):
        if not valid_label[i] or not np.isfinite(raw_w[i]):
            continue
        # The label must resolve inside the available data.
        if label_end[i] >= n_rows:
            continue

        window = X_raw[i - seq_len : i]
        if np.isnan(window).any():
            continue

        X.append(window)
        Y.append(int(labels[i]))
        W.append(float(raw_w[i]))

    weights = np.array(W, dtype=np.float32)
    if len(weights) and weights.sum() > 0:
        weights = weights / weights.mean()  # mean 1, so loss scale is unchanged
    else:
        weights = np.ones(len(Y), dtype=np.float32)

    out = (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.int64),
        features,
        scaler,
    )
    # weights before df_feat: existing callers do `*_, df_feat = add_features(...)`
    # and must keep getting df_feat as the last element.
    if return_weights:
        out = out + (weights,)
    if return_df_feat:
        return out + (df_feat,)
    return out


# ── Пример правильного использования (train/val/test split) ──────────────────
#
#   train_df, val_df, test_df = split(df)  # хронологически, без перемешивания
#
#   X_train, y_train, features, scaler = add_features(train_df)
#   X_val,   y_val,   _,        _      = add_features(val_df,  scaler=scaler)
#   X_test,  y_test,  _,        _      = add_features(test_df, scaler=scaler)
#
# scaler обучается ТОЛЬКО на train_df — никакого data leakage.