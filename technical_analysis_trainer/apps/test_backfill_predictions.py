"""
Leak tests for the prediction backfill.

Run: python -m technical_analysis_trainer.apps.test_backfill_predictions

These use the real ONNX artifacts and the CSV history — no network, no database.
The central claim being tested is that a backfilled prediction for candle T is
identical whether or not the source frame contains candles after T. If that ever
stops holding, backfilled forecasts are seeing their own future and every
accuracy number computed from them is inflated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from technical_analysis_trainer.apps.backfill_predictions import _trim_to, _utc
from technical_analysis_trainer.core.artifacts import ArtifactStore
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.quantile.predict import predict_from_candles
from technical_analysis_trainer.models.trend.plugin import load_trend_config
from technical_analysis_trainer.models.trend.predict import (
    build_trend_x_window_from_frames,
)

CSV = "technical_analysis_trainer/data/btcusdt_1h_train.csv"


def _load(n: int = 1400) -> pd.DataFrame:
    df = pd.read_csv(CSV).tail(n).reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def test_trim_removes_future_rows() -> None:
    df = _load(300)
    cutoff = _utc(df["ts"].iloc[200])
    out = _trim_to(df, cutoff)
    assert len(out) == 201, len(out)
    assert _utc(out["ts"].iloc[-1]) == cutoff
    assert (out["ts"] <= cutoff).all(), "trim left future rows behind"


def test_trim_is_idempotent() -> None:
    df = _load(300)
    cutoff = _utc(df["ts"].iloc[150])
    once = _trim_to(df, cutoff)
    twice = _trim_to(once, cutoff)
    assert once.equals(twice)


def test_quantile_prediction_ignores_future_candles() -> None:
    """The headline leak test for the quantile model."""
    df = _load()
    store = ArtifactStore(package_models_dir("quantile"))
    anchor_i = len(df) - 60
    anchor = _utc(df["ts"].iloc[anchor_i])

    # (a) frame that still contains 59 future candles, trimmed by the backfill
    with_future, _ = predict_from_candles(
        _trim_to(df, anchor), store, drop_last=False
    )
    # (b) frame that never had them
    truncated = df.iloc[: anchor_i + 1].reset_index(drop=True)
    without_future, _ = predict_from_candles(truncated, store, drop_last=False)

    for k in ("q10", "q50", "q90", "sigma", "last_close"):
        assert with_future[k] == without_future[k], (
            f"{k} differs: {with_future[k]} vs {without_future[k]} — future data leaked"
        )


def test_quantile_anchor_is_the_last_closed_bar() -> None:
    df = _load()
    store = ArtifactStore(package_models_dir("quantile"))
    anchor_i = len(df) - 60
    anchor = _utc(df["ts"].iloc[anchor_i])
    trimmed = _trim_to(df, anchor)

    preds, _ = predict_from_candles(trimmed, store, drop_last=False)
    expected_close = float(trimmed["close"].iloc[-1])
    assert abs(preds["last_close"] - expected_close) < 1e-6, (
        f"anchored on the wrong bar: {preds['last_close']} vs {expected_close}"
    )

    # drop_last=True must anchor one bar earlier — this is the bug the flag exists
    # to prevent, so assert the two really do differ.
    live_style, _ = predict_from_candles(trimmed, store, drop_last=True)
    assert live_style["last_close"] != preds["last_close"]


def test_trend_window_ignores_future_candles() -> None:
    df = _load()
    cfg = load_trend_config()
    store = ArtifactStore(package_models_dir("trend"))
    anchor_i = len(df) - 60
    anchor = _utc(df["ts"].iloc[anchor_i])

    x_a, meta_a = build_trend_x_window_from_frames(
        cfg, store, btc=_trim_to(df, anchor), last_row_is_closed=True
    )
    x_b, meta_b = build_trend_x_window_from_frames(
        cfg, store,
        btc=df.iloc[: anchor_i + 1].reset_index(drop=True),
        last_row_is_closed=True,
    )
    assert np.array_equal(x_a, x_b), "trend window changed when future rows were present"
    assert _utc(meta_a["last_ts"]) == _utc(meta_b["last_ts"]) == anchor


def test_backfill_and_live_anchors_agree() -> None:
    """
    Backfill at T and the live path one bar later must use the same window.

    Live sees [..., T, forming] and stops before the forming bar; backfill sees
    [..., T] and includes it. Same content, different indexing — if these drift,
    backfilled and live predictions are not comparable and the accuracy panel
    mixes two different things.
    """
    df = _load()
    cfg = load_trend_config()
    store = ArtifactStore(package_models_dir("trend"))
    anchor_i = len(df) - 60
    anchor = _utc(df["ts"].iloc[anchor_i])

    backfill_x, backfill_meta = build_trend_x_window_from_frames(
        cfg, store, btc=df.iloc[: anchor_i + 1].reset_index(drop=True),
        last_row_is_closed=True,
    )
    live_x, live_meta = build_trend_x_window_from_frames(
        cfg, store, btc=df.iloc[: anchor_i + 2].reset_index(drop=True),
        last_row_is_closed=False,
    )
    assert np.array_equal(backfill_x, live_x), "backfill and live windows diverge"
    assert _utc(backfill_meta["last_ts"]) == _utc(live_meta["last_ts"]) == anchor


def test_predictions_differ_across_anchors() -> None:
    """Guard against a trivially-passing suite: different bars must differ."""
    df = _load()
    store = ArtifactStore(package_models_dir("quantile"))
    out = []
    for back in (60, 61, 62):
        a = _utc(df["ts"].iloc[len(df) - back])
        p, _ = predict_from_candles(_trim_to(df, a), store, drop_last=False)
        out.append(p["q50"])
    assert len(set(out)) == 3, f"predictions identical across anchors: {out}"


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
