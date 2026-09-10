"""
Leak tests for the barrier-resolution target (plan 5.5 / 20).

Run: python -m technical_analysis_trainer.models.trend.test_resolution

Why this file exists at all. The resolution model scores per-month ROC-AUC 0.805
over 36 rolling origins and beats 0.5 in 36 of 36 months, in a repo where the
direction model manages 0.55 and everything else is a coin. A result that far
outside the local prior is either the first real finding here or a leak, and the
difference has to be established rather than hoped for.

One structural argument comes free and is worth stating: the resolution target
runs through the SAME feature pipeline, the same windowing loop and the same
walk-forward split as the direction target. A lookahead defect anywhere in that
shared path would inflate both. It inflates neither — direction stays at 0.52-0.55
with the identical inputs — so any leak would have to live in the few lines that
differ, which is exactly the target construction these tests attack.

What is checked:

  * the label at an anchor uses only bars in (anchor, anchor + max_holding];
  * the feature window at an anchor uses only bars up to and including it;
  * the anchors whose outcome is not yet observable are excluded, which is the
    bug that was actually there (a clipped barrier scan reads as "touched");
  * with the labels permuted, the whole pipeline scores ~0.5 — if it does not,
    the signal is coming from the plumbing rather than from the data.

The last one is the load-bearing test. Everything else can be satisfied by a
target that is merely self-consistent; only the permutation control distinguishes
"the model learned something" from "the harness is feeding it the answer".
"""
from __future__ import annotations

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.apps.rolling_eval import TREND_CONFIG, load_frame
from technical_analysis_trainer.models.trend.features import (
    resolve_feature_scale_clip,
    triple_barrier_labels,
)
from technical_analysis_trainer.models.trend.model import TrendEnsemble, train_trend_lstm
from technical_analysis_trainer.models.trend.train_v2 import (
    build_span,
    proba_up,
    set_seed,
    span_warmup,
)

_CACHE: dict = {}


def _setup():
    if not _CACHE:
        with open(TREND_CONFIG, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        clip_lo, clip_hi = resolve_feature_scale_clip(raw)
        tb = raw.get("triple_barrier") or {}
        cfg = {
            "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
            "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
            "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
            "target_mode": "triple_barrier", "tb_cfg": tb, "seed": 42,
            "anchor": str(raw.get("barrier_anchor", "last_observed")),
            "weight_clip_q": raw.get("weight_clip_quantile"),
            "order_flow": bool(raw.get("order_flow", True)),
            "exclude": list(raw.get("feature_exclude") or []),
            "label_autocorr": False,
            "label_mode": "resolution",
        }
        _CACHE.update(df=load_frame(raw), cfg=cfg, tb=tb)
    return _CACHE["df"], _CACHE["cfg"], _CACHE["tb"]


def _touched(df, tb, H):
    """Frame-level resolution label, independent of add_features."""
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
    observable = np.arange(len(df)) + H < len(df)
    return ((labels >= 0) | ((spans < H) & observable)).astype(np.int64), observable


def test_label_ignores_candles_beyond_the_holding_window() -> None:
    """
    The outcome at anchor T is decided inside (T, T+max_holding]. Nothing later
    may touch it — otherwise the target is reading a future it would not have.
    """
    df, _, tb = _setup()
    H = int(tb.get("max_holding", 24))
    n = len(df)
    cut = n - 400                      # anchors well clear of the truncation edge

    full, obs_full = _touched(df, tb, H)
    # Everything after `cut` is deleted. Anchors at or before cut - H still have
    # their whole window inside the truncated frame, so their label must not move.
    trimmed, obs_trim = _touched(df.iloc[:cut].reset_index(drop=True), tb, H)

    safe = np.arange(cut - H)
    safe = safe[obs_full[safe] & obs_trim[safe]]
    assert len(safe) > 1000, f"only {len(safe)} comparable anchors"
    bad = int((full[safe] != trimmed[safe]).sum())
    assert bad == 0, (
        f"{bad} of {len(safe)} resolution labels changed when candles beyond the "
        "holding window were removed — the target is reading past its own horizon"
    )


def test_unobservable_anchors_are_excluded() -> None:
    """
    The last `max_holding` anchors have nowhere left to look, so their outcome is
    unknown. `triple_barrier_labels` clips its scan for them and reports a short
    span, which reads as "a barrier was touched" — the bug this guards.
    """
    df, cfg, tb = _setup()
    H = int(tb.get("max_holding", 24))
    n = len(df)
    _, y, _, _, _, bars = build_span(df, (n - 730, n), span_warmup(cfg), None, cfg,
                                     return_index=True)
    assert len(bars), "span produced no rows"
    assert bars.max() < n - H, (
        f"anchor at bar {bars.max()} is within max_holding ({H}) of the frame end "
        f"({n}); its outcome cannot be known yet"
    )
    touched, _ = _touched(df, tb, H)
    assert np.array_equal(touched[bars], y), "returned labels disagree with the frame"


def test_feature_window_ignores_future_candles() -> None:
    """
    The inputs for an anchor must be identical whether or not the frame contains
    anything after it. This is the same property test_backfill_predictions pins
    for the serving path, checked here on the training path.
    """
    df, cfg, _ = _setup()
    n = len(df)
    warmup = span_warmup(cfg)
    # One scaler, fitted on an earlier span and reused for both, so the two
    # windows are directly comparable. Letting each span fit its own would make
    # them differ by a per-feature affine map and test the scaler, not the
    # windowing — the values would never match even with the anchoring correct.
    *_, scaler = build_span(df, (0, n - 4000), warmup, None, cfg)[:5]
    Xa, _, _, _, _, ba = build_span(df, (n - 1200, n - 200), warmup, scaler, cfg,
                                    return_index=True)
    Xb, _, _, _, _, bb = build_span(df, (n - 1200, n), warmup, scaler, cfg,
                                    return_index=True)
    shared = np.intersect1d(ba, bb)
    assert len(shared) > 200, f"only {len(shared)} shared anchors"
    ia = {int(b): i for i, b in enumerate(ba)}
    ib = {int(b): i for i, b in enumerate(bb)}
    for b in shared:
        wa, wb = Xa[ia[int(b)]], Xb[ib[int(b)]]
        assert np.array_equal(wa, wb), (
            f"anchor {b}: feature window changed when 200 future candles were "
            "added — the window is not anchored where it claims"
        )


def test_permuted_labels_destroy_the_signal() -> None:
    """
    The control that decides whether 0.80 is real.

    Train the identical pipeline on the identical windows with the training
    labels randomly permuted. The features, the split, the weights, the scaler
    and the evaluation are untouched; only the association between window and
    outcome is broken. If the harness were feeding the answer through some
    channel other than the labels, the score would survive. It must not.
    """
    df, cfg, _ = _setup()
    n = len(df)
    warmup = span_warmup(cfg)
    tr_end = n - 2000
    Xtr, ytr, wtr, _, scaler = build_span(df, (0, tr_end), warmup, None, cfg)
    Xte, yte, _, _, _ = build_span(df, (tr_end + 200, n), warmup, scaler, cfg)
    assert len(Xtr) > 2000 and len(Xte) > 300, f"spans too small: {len(Xtr)}, {len(Xte)}"

    fold_cfg = dict(cfg["raw"])
    fold_cfg["stop_after_epoch"] = 2
    if fold_cfg.get("class_weight") == "auto":
        b = np.bincount(ytr, minlength=2)
        fold_cfg["class_weight"] = [len(ytr) / (2 * max(b[0], 1)), len(ytr) / (2 * max(b[1], 1))]

    def fit_and_score(y_train):
        members = []
        for s in range(2):
            set_seed(1234 + s)
            m, _ = train_trend_lstm(X=Xtr, y=y_train, cfg=fold_cfg, w=wtr)
            members.append(m)
        return float(roc_auc_score(yte, proba_up(TrendEnsemble(members), Xte)))

    real = fit_and_score(ytr)
    rng = np.random.default_rng(0)
    shuffled = fit_and_score(ytr[rng.permutation(len(ytr))])

    assert real > 0.65, (
        f"real labels scored only {real:.4f} on a held-out span; the headline "
        "result does not reproduce and the control below proves nothing"
    )
    # Two-sided, and that matters. Bounding the permuted score only from ABOVE
    # lets the worst case through: feed the outcome in as a feature and permuted
    # training makes the model learn the inverse, scoring 0.12 — comfortably
    # "below 0.60" while proving the pipeline hands over the answer. What a clean
    # pipeline must produce on permuted labels is CHANCE, in either direction.
    assert abs(shuffled - 0.5) < 0.10, (
        f"permuted labels scored {shuffled:.4f}, not ~0.5 — the outcome is "
        "reaching the model through something other than the labels"
    )
    assert real - shuffled > 0.15, (
        f"real {real:.4f} vs permuted {shuffled:.4f}: the gap is too small to "
        "attribute the score to the data"
    )
    print(f"        (real {real:.4f} vs permuted {shuffled:.4f})", end="")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            print(f"  ....  {t.__name__}", end="", flush=True)
            t()
            print(f"\r  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"\r  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
