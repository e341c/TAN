"""
Checks on the historical backtest. Run directly:
    PYTHONPATH=. python -m technical_analysis_trainer.apps.test_backtest

The load-bearing one is `test_batch_scoring_matches_the_live_path`. The backtest
computes features once over the whole series and scores every anchor in a few
batched ONNX calls; the live path rebuilds features per anchor from a stitched
frame. If those two ever disagree, the numbers this tool prints describe a model
nobody is actually serving. `apps/test_backfill_predictions.py` already pins
live-vs-backfill; this closes the third corner.

Everything runs over **holdout candles** — bars that closed after the model was
exported, fetched on demand by `data_fetch/holdout.py` and cached while fresh.
Set HOLDOUT_OFFLINE=1 to forbid fetching and fail instead. Scoring the agreement
on data the model was fitted on is the weaker test: a memorised window can return
the same answer for two subtly different inputs and hide the very windowing bug
this file exists to catch.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml

from technical_analysis_trainer.apps.backtest import (
    _apply_range,
    _override_cross_market_paths,
    run_backtest,
)
from technical_analysis_trainer.apps.holdout_eval import trend_backtest_config
from technical_analysis_trainer.core.artifacts import ArtifactStore, resolve_path
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.data_fetch.holdout import ensure_holdout
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.trend.predict import (
    build_trend_x_window_from_frames,
    run_trend_onnx,
)
from technical_analysis_trainer.models.trend.target import ANCHOR_LAST_OBSERVED, spec_anchor

TREND_CFG = "technical_analysis_trainer/models/trend/config.yaml"
ANCHORS = 120
_CACHE: dict[str, object] = {}


def _holdout():
    if "holdout" not in _CACHE:
        _CACHE["holdout"] = ensure_holdout(
            offline=os.environ.get("HOLDOUT_OFFLINE") == "1", quiet=True
        )
    return _CACHE["holdout"]


def _holdout_config() -> dict:
    """Backtest config pinned to the holdout CSVs, unseen anchors only."""
    return trend_backtest_config(_holdout())


def _small_result():
    """One backtest over a short window of unseen bars, reused across tests."""
    if "result" not in _CACHE:
        h = _holdout()
        cfg = _holdout_config()
        # `start` already excludes everything the model trained on; cap the window
        # at the oldest ANCHORS bars after it so the size is deterministic and the
        # suite does not slow down as the holdout grows. Derived from the CSV's own
        # timestamps rather than from a throwaway full backtest.
        ts = pd.to_datetime(pd.read_csv(h.btc, usecols=["ts"])["ts"], utc=True)
        unseen = ts[ts > h.cutoff].reset_index(drop=True)
        if len(unseen) < ANCHORS:
            raise AssertionError(
                f"only {len(unseen)} unseen bars available, need {ANCHORS} — refresh the "
                "holdout (python -m technical_analysis_trainer.data_fetch.holdout)"
            )
        cfg["range"]["end"] = unseen.iloc[ANCHORS - 1].isoformat()
        _CACHE["result"] = run_backtest(cfg)
    return _CACHE["result"]


def test_every_anchor_is_unseen() -> None:
    """The premise of the rest of this file, asserted rather than assumed."""
    r = _small_result()
    cutoff = _holdout().cutoff
    first = pd.Timestamp(r.frame["ts"].iloc[0])
    assert first > cutoff, f"first anchor {first} is not after the export cutoff {cutoff}"
    assert r.unseen_from == cutoff, "backtest was not told where the unseen window starts"
    assert len(r.out_of_sample) == len(r.resolved), "some scored anchors predate the cutoff"


def test_batch_scoring_matches_the_live_path() -> None:
    r = _small_result()
    with open(resolve_path(TREND_CFG), encoding="utf-8") as f:
        trend_cfg = yaml.safe_load(f)
    # The live path must read the same CSVs the backtest did, or a mismatch here
    # would just be reporting that two different SOL series give two different
    # answers — which is true and uninteresting.
    _override_cross_market_paths(trend_cfg, _holdout_config()["data"])
    store = ArtifactStore(package_models_dir("trend"))

    frames = {}
    for key, path in (
        ("btc", _holdout_config()["data"]["btc"]),
        ("eth", ((trend_cfg["cross_market"]["benchmarks"].get("eth") or {}).get("path"))),
        ("sol", ((trend_cfg["cross_market"]["benchmarks"].get("sol") or {}).get("path"))),
    ):
        if not path:
            continue
        d = load_dataset(str(resolve_path(path)))
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        frames[key] = d

    # Three anchors spread across the window, not just the last one.
    for row in r.frame.iloc[[0, len(r.frame) // 2, -1]].itertuples(index=False):
        # The live path treats the final row of the frame as the last CLOSED bar,
        # i.e. the last bar inside the window — that is `window_end_ts`, one bar
        # before the barrier origin the backtest anchors on.
        cut = pd.Timestamp(row.window_end_ts)
        trimmed = {k: v[v["ts"] <= cut].reset_index(drop=True) for k, v in frames.items()}
        x, meta = build_trend_x_window_from_frames(
            trend_cfg, store, run_id=r.run_id,
            btc=trimmed["btc"], eth=trimmed.get("eth"), sol=trimmed.get("sol"),
            last_row_is_closed=True,
        )
        assert pd.Timestamp(meta["last_ts"]) == cut, (
            f"live path anchored on {meta['last_ts']}, backtest window ends {cut}"
        )
        # Pass the artifact's calibrator AND the trailing prior, because the
        # backtest applies both. Comparing a corrected number against an
        # uncorrected one would fail for a reason that has nothing to do with
        # the windowing this test is about.
        #
        # This also makes the check stronger than it looks: the live prior is
        # computed from a frame trimmed at the anchor, the batch one from the
        # full series, so agreement here is evidence that the base-rate estimate
        # is not reading resolutions from after the anchor.
        live = run_trend_onnx(
            x, str(meta["onnx_path"]), calibrator=meta.get("calibrator"),
            prior_now=meta.get("prior_now"),
        )["probabilities"]["UP"]
        assert abs(live - row.p_up) < 1e-3, (
            f"anchor {row.ts}: live p(UP)={live:.6f} vs batch {row.p_up:.6f}"
        )


def test_barrier_reconstructs_from_close_and_sigma() -> None:
    r = _small_result()
    spec = r.spec
    up, dn = float(spec["upper_mult"]), float(spec["lower_mult"])
    f = r.frame
    mult = np.where(f["pred"] == 1, up, -dn)
    expected = f["close"].to_numpy() * np.exp(mult * f["sigma"].to_numpy())
    assert np.allclose(f["barrier"].to_numpy(), expected, rtol=1e-9), "barrier != close*exp(±k*sigma)"
    # And the direction must follow the prediction, or the chart lies about it.
    assert (f.loc[f["pred"] == 1, "barrier"] > f.loc[f["pred"] == 1, "close"]).all()
    assert (f.loc[f["pred"] == 0, "barrier"] < f.loc[f["pred"] == 0, "close"]).all()


def test_unresolved_anchors_are_excluded_but_counted() -> None:
    r = _small_result()
    f = r.frame
    # -1 is triple_barrier_labels' "neither barrier touched inside max_holding".
    assert set(f["actual"].unique()) <= {-1, 0, 1}
    assert (r.resolved["actual"] >= 0).all(), "resolved must contain no -1 labels"
    assert len(r.resolved) + int((f["actual"] < 0).sum()) == len(f), "coverage must add up"
    # Unresolved bars are still predicted — they just cannot be graded.
    assert f.loc[f["actual"] < 0, "pred"].isin([0, 1]).all()


def test_span_is_within_the_holding_window() -> None:
    r = _small_result()
    max_holding = int(r.spec["max_holding"])
    spans = r.resolved["span"].to_numpy()
    assert (spans >= 1).all() and (spans <= max_holding).all(), (
        f"resolved spans must lie in 1..{max_holding}, got {spans.min()}..{spans.max()}"
    )


def test_range_start_end_select_exactly_those_anchors() -> None:
    ts = pd.Series(pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"))
    idx = np.arange(50, 200)

    picked = _apply_range(idx, ts, {"start": "2026-01-05", "end": "2026-01-06"})
    times = pd.to_datetime(ts.to_numpy()[picked], utc=True)
    assert times.min() >= pd.Timestamp("2026-01-05", tz="UTC")
    assert times.max() <= pd.Timestamp("2026-01-06", tz="UTC")

    # last_bars only applies when no explicit window is given.
    assert len(_apply_range(idx, ts, {"last_bars": 10})) == 10
    assert len(_apply_range(idx, ts, {"last_bars": 10, "start": "2026-01-03"})) > 10
    # stride thins the result without moving its end.
    strided = _apply_range(idx, ts, {"last_bars": 20, "stride": 5})
    assert len(strided) == 4 and strided[0] == idx[-20]


def test_range_covers_the_configured_window() -> None:
    r = _small_result()
    assert len(r.frame) == ANCHORS, (
        f"the start/end window should give {ANCHORS} anchors, got {len(r.frame)}"
    )
    assert r.frame["ts"].is_monotonic_increasing
    # Where the anchor sits relative to the window depends on the artifact's own
    # convention, so read it rather than assume one:
    #   last_observed  the barrier origin IS the last bar in the window (gap 0)
    #   next_close     it is the bar after it (gap 1)
    expected = pd.Timedelta(0) if spec_anchor(r.spec) == ANCHOR_LAST_OBSERVED else r.bar
    delta = pd.to_datetime(r.frame["ts"]) - pd.to_datetime(r.frame["window_end_ts"])
    assert (delta == expected).all(), (
        f"anchor/window gap should be {expected} for anchor={spec_anchor(r.spec)}, "
        f"got {delta.unique()}"
    )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001 - a missing holdout must report, not traceback
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
