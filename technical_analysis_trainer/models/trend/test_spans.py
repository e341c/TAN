"""
Alignment tests for `build_span`.

Run: python -m technical_analysis_trainer.models.trend.test_spans

`build_span` is asked for the bars in `[start, end)` and is the only thing
standing between the walk-forward split boundaries and every number this repo
reports. It used to get that wrong in a way nothing could see.

The trap is that `add_features` does not return one row per bar. Two filters sit
between the frame and the windows: `build_features_df` ends with a global
`dropna().reset_index(drop=True)`, and the windowing loop skips anchors whose
label is unresolved, whose label resolves past the end of the frame, or whose
window still holds a warmup NaN. Together they drop about half the candidate
bars, non-contiguously. `build_span` nevertheless trimmed its warm-up lead-in by
COUNT — `X[len(X) - (end - start):]` — which is only correct if rows and bars are
the same thing. They are not, so the trim computed zero, and every validation and
test span silently began ~200 bars before its own start, inside the split it was
supposed to be held out from.

Nothing failed. The shapes were right, the metrics were plausible, and the only
symptom was that held-out scores were measured partly on non-held-out bars.

These tests pin the two properties that make the trim checkable:

  * every returned anchor lies inside the requested span, and
  * the returned index really is the anchor bar — the label at that bar in the
    full frame equals the label returned alongside it.

The second is the one that matters. Without it the index is just an array that
looks plausible, which is exactly what the old count-based trim produced.
"""
from __future__ import annotations

import numpy as np
import yaml

from technical_analysis_trainer.apps.rolling_eval import TREND_CONFIG, label_series, load_frame
from technical_analysis_trainer.models.trend.features import (
    resolve_feature_scale_clip,
    triple_barrier_labels,
)
from technical_analysis_trainer.models.trend.train_v2 import build_span, span_warmup

_CACHE: dict = {}


def _setup():
    """Load the training frame and the config once; these tests are read-only."""
    if not _CACHE:
        with open(TREND_CONFIG, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        clip_lo, clip_hi = resolve_feature_scale_clip(raw)
        tb = raw.get("triple_barrier") or {}
        cfg = {
            "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
            "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
            "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
            "target_mode": raw.get("target_mode", "triple_barrier"), "tb_cfg": tb,
            "seed": 42, "anchor": str(raw.get("barrier_anchor", "last_observed")),
            "weight_clip_q": raw.get("weight_clip_quantile"),
            "order_flow": bool(raw.get("order_flow", True)),
            "exclude": list(raw.get("feature_exclude") or []),
            "label_autocorr": bool(raw.get("label_autocorr", False)),
        }
        df = load_frame(raw)
        _CACHE.update(df=df, cfg=cfg, tb=tb, labels=label_series(df, tb)[0])
    return _CACHE["df"], _CACHE["cfg"], _CACHE["tb"], _CACHE["labels"]


def _span(start: int, end: int):
    df, cfg, _, _ = _setup()
    return build_span(df, (start, end), span_warmup(cfg), None, cfg, return_index=True)


def test_anchors_stay_inside_the_requested_span() -> None:
    """No row may come from before `start` — that is the leak the trim exists to stop."""
    df, _, _, _ = _setup()
    n = len(df)
    for start, end in ((n - 730, n), (n - 1460, n - 730), (n - 4380, n - 3650)):
        *_, bars = _span(start, end)
        assert len(bars), f"span [{start},{end}) produced no rows"
        assert bars.min() >= start, (
            f"span [{start},{end}) returned an anchor at bar {bars.min()}, "
            f"{start - bars.min()} bars before its own start — warm-up lead-in "
            "leaked into the span"
        )
        assert bars.max() < end, f"anchor {bars.max()} is at or past end {end}"


def test_returned_index_is_the_anchor_bar() -> None:
    """
    The label at the reported bar must be the label returned with it.

    This is what makes the index trustworthy rather than merely plausible. Any
    off-by-k, any silent re-indexing inside `build_features_df`, any reversion to
    a count-based trim breaks this and nothing else would notice.
    """
    df, _, _, labels = _setup()
    n = len(df)
    for start, end in ((n - 730, n), (n - 2920, n - 2190)):
        _, y, _, _, _, bars = _span(start, end)
        assert np.array_equal(labels[bars], y), (
            f"span [{start},{end}): labels at the reported bars disagree with the "
            f"returned labels for {int((labels[bars] != y).sum())} of {len(y)} rows — "
            "the index does not identify the anchor"
        )


def test_adjacent_spans_do_not_share_anchors() -> None:
    """Consecutive splits must be disjoint; overlap is train/test contamination."""
    df, _, _, _ = _setup()
    n = len(df)
    *_, a = _span(n - 1460, n - 730)
    *_, b = _span(n - 730, n)
    shared = np.intersect1d(a, b)
    assert not len(shared), f"{len(shared)} anchors appear in both adjacent spans"


def test_warmup_covers_the_start_of_the_span() -> None:
    """
    The lead-in must be long enough that the span starts producing rows promptly.

    With too little warm-up the features are still NaN well past `start`, so the
    span quietly loses its first weeks. At `seq_len + 200` that was 186 bars — a
    quarter of a monthly window — and the only visible effect was a smaller row
    count that looked like ordinary label attrition.
    """
    df, cfg, _, _ = _setup()
    n = len(df)
    start, end = n - 730, n
    *_, bars = _span(start, end)
    lag = int(bars.min() - start)
    assert lag <= 48, (
        f"first usable anchor is {lag} bars into the span; warm-up "
        f"({span_warmup(cfg)}) is too short for the feature set"
    )


def test_resolution_label_matches_the_frame() -> None:
    """
    The "was a barrier touched" target must be reconstructible from the frame.

    This is the same check that proves the direction target is aligned, and it
    caught a real bug in the resolution target. `triple_barrier_labels` clips its
    scan to `min(i + max_holding, n - 1)`, so an anchor near the right edge
    reports a SHORT span without a barrier ever being hit — and the obvious test
    for "touched", `spans < max_holding`, reads that as a touch. The
    `label_end >= n_rows` guard does not catch it, because the clipped span makes
    label_end land just inside the frame.

    The fix excludes the last `max_holding` anchors, whose outcome is not
    observable. This test fails if that exclusion is ever dropped.
    """
    df, cfg, tb, _ = _setup()
    n = len(df)
    H = int(tb.get("max_holding", 24))
    rcfg = dict(cfg, label_mode="resolution")
    start, end = n - 730, n
    _, y, _, _, _, bars = build_span(df, (start, end), span_warmup(rcfg), None, rcfg,
                                     return_index=True)

    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    vw = int(tb.get("vol_window", 72))
    sigma = lr.rolling(vw, min_periods=vw).std().to_numpy()
    labels, spans, _ = triple_barrier_labels(
        close.to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), sigma,
        upper_mult=float(tb.get("upper_mult", 2.5)),
        lower_mult=float(tb.get("lower_mult", 2.5)), max_holding=H,
    )
    observable = np.arange(n) + H < n
    touched = ((labels >= 0) | ((spans < H) & observable)).astype(np.int64)
    assert np.array_equal(touched[bars], y), (
        f"{int((touched[bars] != y).sum())} of {len(y)} resolution labels disagree "
        "with the frame — the target is not what it claims to be"
    )
    # And it must not be degenerate, or the assertion above proves nothing.
    assert 0.05 < y.mean() < 0.99, f"resolution label is degenerate: base rate {y.mean():.3f}"


def test_rows_are_not_assumed_contiguous() -> None:
    """
    Guard against the suite passing for the wrong reason.

    If `add_features` ever did return one row per bar, the count-based trim would
    have been fine and these tests would be vacuous. It does not, and this pins
    that: the anchors are genuinely gappy, so the index is genuinely needed.
    """
    df, _, _, _ = _setup()
    n = len(df)
    *_, bars = _span(n - 730, n)
    gaps = np.diff(bars)
    assert (gaps > 1).any(), (
        "anchors are contiguous; either the label filters stopped dropping bars "
        "or this span is degenerate — the alignment tests above are not proving "
        "what they claim"
    )


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
