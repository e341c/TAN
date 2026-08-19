"""
Checks on the walk-forward protocol. Run directly: python -m technical_analysis_trainer.core.test_validation
"""
from __future__ import annotations

import numpy as np

from technical_analysis_trainer.core.metrics import (
    conformal_offsets,
    har_rv_forecast,
    skill_score,
)
from technical_analysis_trainer.core.validation import (
    dead_features,
    expected_max_of_noise,
    feature_drift,
    select_by_val,
    walk_forward_splits,
)


def test_splits_are_ordered_and_disjoint() -> None:
    for split in walk_forward_splits(15550, n_folds=3, embargo=25):
        tr, va, te = split.train, split.val, split.test
        assert tr[0] < tr[1] <= va[0] < va[1] <= te[0] < te[1], split
        # Embargo must actually separate the segments.
        assert va[0] - tr[1] >= 25, f"train/val embargo violated: {split}"
        assert te[0] - va[1] >= 25, f"val/test embargo violated: {split}"


def test_folds_advance_in_time() -> None:
    splits = walk_forward_splits(15550, n_folds=3, embargo=25)
    starts = [s.test[0] for s in splits]
    assert starts == sorted(starts), "folds must move forward in time"
    assert len(set(starts)) == len(starts), "folds must not repeat a test period"


def test_embargo_covers_label_lookahead() -> None:
    """A label spanning `h` bars must not straddle the train/val boundary."""
    h = 24
    for split in walk_forward_splits(15550, n_folds=3, embargo=h):
        last_train_label_end = split.train[1] - 1 + h
        assert last_train_label_end <= split.val[0], (
            f"label from train reaches into val: {last_train_label_end} > {split.val[0]}"
        )


def test_select_by_val_refuses_test_metrics() -> None:
    runs = [{"val_loss": 0.5, "test_auc": 0.9}, {"val_loss": 0.4, "test_auc": 0.1}]
    assert select_by_val(runs, "val_loss")["val_loss"] == 0.4
    try:
        select_by_val(runs, "test_auc", lower_is_better=False)
    except ValueError:
        return
    raise AssertionError("selecting on a test metric must raise")


def test_expected_max_matches_the_trend_run_log() -> None:
    """The 184-run log: mean 0.5305, std 0.0052, observed best 0.5457."""
    predicted = expected_max_of_noise(0.5305, 0.0052, 184)
    assert abs(predicted - 0.5457) < 0.005, (
        f"best run {0.5457} should be explainable as noise, predicted {predicted:.4f}"
    )


def test_har_rv_is_finite_despite_leading_nan() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.003, 2000)
    r[0] = np.nan  # first diff is always NaN
    out = har_rv_forecast(r)
    tail = out[200:]
    assert np.isfinite(tail).all(), "a leading NaN must not poison the cumsum"
    assert (tail > 0).all()
    # Should land in the right ballpark for a 0.003-vol series.
    assert 0.001 < float(np.mean(tail)) < 0.01, float(np.mean(tail))


def test_conformal_offsets_restore_coverage() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 5000)
    # Deliberately too-narrow band: 80% nominal, far less actual coverage.
    lo, hi = np.full(5000, -0.5), np.full(5000, 0.5)
    before = float(np.mean((y >= lo) & (y <= hi)))
    off_lo, off_hi = conformal_offsets(y - lo, y - hi, 0.1, 0.1)
    after = float(np.mean((y >= lo + off_lo) & (y <= hi + off_hi)))
    assert before < 0.5, before
    assert abs(after - 0.8) < 0.03, f"conformal should restore ~80%, got {after:.3f}"


def test_skill_score_signs() -> None:
    assert skill_score(0.5, 1.0) == 0.5       # half the loss -> +0.5
    assert skill_score(1.0, 1.0) == 0.0       # no better than baseline
    assert skill_score(2.0, 1.0) == -1.0      # worse than baseline


def test_dead_features_finds_the_constant_column() -> None:
    from sklearn.preprocessing import StandardScaler

    x = np.column_stack([np.arange(100.0), np.full(100, 3.0), np.random.randn(100)])
    scaler = StandardScaler().fit(x)
    assert dead_features(scaler, ["moving", "constant", "noise"]) == ["constant"]
    # And the reason it matters: sklearn leaves a dead column unscaled.
    assert scaler.scale_[1] == 1.0


def test_dead_features_empty_when_all_vary() -> None:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(np.random.randn(100, 3))
    assert dead_features(scaler, ["a", "b", "c"]) == []


def test_feature_drift_flags_a_trending_input() -> None:
    # "stationary" keeps its distribution; "trending" walks 2 sd away, which is
    # what a raw price level does to a scaler frozen at training time.
    train = np.column_stack([np.random.randn(500), np.random.randn(500)])
    later = np.column_stack([np.random.randn(500), np.random.randn(500) + 2.0])
    drifted = feature_drift(train, later, ["stationary", "trending"], max_shift_sd=0.75)
    assert [name for name, _ in drifted] == ["trending"], drifted
    assert 1.5 < drifted[0][1] < 2.5, drifted


def test_feature_drift_accepts_sequence_windows() -> None:
    # Trainers hold (N, seq_len, F); the check must not require flattening first.
    train = np.random.randn(200, 24, 2)
    later = np.random.randn(200, 24, 2)
    later[:, :, 0] += 3.0
    drifted = feature_drift(train, later, ["shifted", "steady"], max_shift_sd=0.75)
    assert [name for name, _ in drifted] == ["shifted"], drifted


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
