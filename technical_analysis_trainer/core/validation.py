"""
Evaluation protocol helpers shared by the trend and quantile packages.

Two failure modes these exist to prevent:

1. **Test-set selection.** Training many seeds and deploying the best-looking
   run inflates the reported metric by roughly the max of N noise draws. On the
   trend model's 184-run log that is ~+2.5 sigma of ROC-AUC — several times the
   model's real edge. `expected_max_of_noise` quantifies the inflation, and
   `select_by_val` enforces that selection only ever reads validation.

2. **Single-split conclusions.** One chronological test set is one market
   regime. `walk_forward_splits` produces several, so a metric comes with a
   spread across regimes rather than a single lucky number.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Split:
    """Index ranges for one walk-forward fold."""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def sizes(self) -> tuple[int, int, int]:
        return (
            self.train[1] - self.train[0],
            self.val[1] - self.val[0],
            self.test[1] - self.test[0],
        )


def walk_forward_splits(
    n: int,
    n_folds: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo: int = 0,
    anchored: bool = True,
) -> list[Split]:
    """
    Chronological walk-forward folds over `n` rows.

    `embargo` drops that many rows between train/val and val/test. With
    overlapping label windows (a horizon > 1, or triple-barrier labels that look
    up to N bars ahead) the last training labels peek into the validation
    period; the embargo must be >= that lookahead or the split leaks.

    anchored=True grows the training window each fold (train always starts at 0);
    False slides a fixed-size window instead.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    test_len = int(n * test_ratio)
    val_len = int(n * val_ratio)
    if test_len < 1 or val_len < 1:
        raise ValueError(f"n={n} too small for {val_ratio=} {test_ratio=}")

    splits: list[Split] = []
    # Last fold ends at n; earlier folds step back one test block at a time.
    for k in range(n_folds):
        test_end = n - k * test_len
        test_start = test_end - test_len
        val_end = test_start - embargo
        val_start = val_end - val_len
        train_end = val_start - embargo
        train_start = 0 if anchored else max(0, train_end - (val_len + test_len) * 4)

        if train_end - train_start < val_len:
            break
        splits.append(
            Split(
                train=(train_start, train_end),
                val=(val_start, val_end),
                test=(test_start, test_end),
            )
        )

    return list(reversed(splits))


def warn_if_stale(df, label: str, max_age_days: int = 3) -> int:
    """
    Print a loud warning when training data ends well before now.

    Training on a series that stops months ago fits the model to a volatility
    regime it will not be served in — and the failure is silent, showing up only
    as a model that emits one class forever. Worth a banner at the top of every
    training run.
    """
    import pandas as pd

    if "ts" not in getattr(df, "columns", []):
        return 0
    last = pd.to_datetime(df["ts"], utc=True).max()
    age = (pd.Timestamp.now(tz="UTC") - last).days
    if age > max_age_days:
        print(
            f"  ⚠ {label}: data ends {last.date()} ({age} days ago). The model will "
            "be fitted to a regime it is not served in. Refresh with "
            "`python -m technical_analysis_trainer.data_fetch.binance_klines`."
        )
    else:
        print(f"  {label}: data through {last.date()} ({age}d old)")
    return age


def dead_features(scaler, features: list[str]) -> list[str]:
    """
    Features that were constant over the training span.

    `StandardScaler` reports `scale_ = 1.0` for a zero-variance column, so such
    a feature is not standardised at all: at inference its raw value is handed
    to the model unchanged, while during training the model only ever saw the
    single constant. A correlation feature that was dead in training and alive
    in production therefore arrives as ~0.9 where the model expects 0.0.

    This is not theoretical either — run 2026-08-17T09-48-39 shipped with
    `xmkt_log_ret_eth` and all three `xmkt_corr_main_eth_*` dead, because the
    ETH benchmark CSV merged to a single constant and
    `attach_cross_market_features` fills the resulting NaNs with 0.0 rather than
    failing. Four of 54 inputs were silently untrained.
    """
    var = getattr(scaler, "var_", None)
    if var is None:
        return []
    return [f for i, f in enumerate(features) if i < len(var) and float(var[i]) == 0.0]


def report_dead_features(scaler, features: list[str], label: str = "train") -> list[str]:
    """Print a banner for every constant feature. Returns the offenders."""
    dead = dead_features(scaler, features)
    if dead:
        print(
            f"  ⚠ {label}: {len(dead)}/{len(features)} feature(s) constant over the "
            f"fit span — the model cannot learn them and inference will feed them "
            f"unscaled: {', '.join(dead)}"
        )
        print("    Usually a data source that failed to merge. Check its CSV before trusting this run.")
    return dead


def feature_drift(
    x_train: np.ndarray,
    x_later: np.ndarray,
    features: list[str],
    max_shift_sd: float = 0.75,
) -> list[tuple[str, float]]:
    """
    Features whose mean moves more than `max_shift_sd` training sd by `x_later`.

    Accepts (N, seq_len, F) windows or flat (N, F) matrices.

    A non-stationary input is the failure mode that produced "the model only
    predicts UP": the scaler is fitted once, so a feature carrying the market's
    absolute scale keeps drifting away from its fitted mean, and the decision
    boundary — fixed at p = 0.5 — is left behind. The model still discriminates;
    every prediction just lands on one side of it.

    Returns (feature, shift_in_train_sd) sorted by absolute shift, worst first.
    """
    a = np.asarray(x_train, dtype=np.float64)
    b = np.asarray(x_later, dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim == 3:
        b = b.reshape(-1, b.shape[-1])
    if a.size == 0 or b.size == 0:
        return []

    m_a, s_a, m_b = a.mean(axis=0), a.std(axis=0), b.mean(axis=0)
    shifts = (m_b - m_a) / np.where(s_a > 1e-12, s_a, np.inf)
    out = [
        (features[i], float(shifts[i]))
        for i in range(min(len(features), len(shifts)))
        if abs(shifts[i]) > max_shift_sd
    ]
    return sorted(out, key=lambda r: -abs(r[1]))


def report_feature_drift(
    x_train: np.ndarray,
    x_later: np.ndarray,
    features: list[str],
    label: str = "test",
    max_shift_sd: float = 0.75,
) -> list[tuple[str, float]]:
    """Print drifting features. Returns them so a caller can fail the run."""
    drifted = feature_drift(x_train, x_later, features, max_shift_sd=max_shift_sd)
    if drifted:
        print(
            f"  ⚠ non-stationary inputs: {len(drifted)} feature(s) shift more than "
            f"{max_shift_sd} train sd by {label} —"
        )
        for name, shift in drifted[:10]:
            print(f"      {name:28} {shift:+.2f} sd")
        print(
            "    A frozen scaler cannot track this. Expect the served probability to "
            "drift off 0.5 and collapse to one class."
        )
    return drifted


def expected_max_of_noise(mean: float, std: float, n_draws: int) -> float:
    """
    Where the best of `n_draws` identical runs lands if the differences are pure
    noise. If your best run is not clearly above this, it is not a discovery.
    """
    if n_draws < 2:
        return mean
    z = NormalDist().inv_cdf(1 - 1 / (n_draws + 1))
    return mean + z * std


def select_by_val(runs: list[dict], metric: str = "val_loss", lower_is_better: bool = True):
    """
    Pick a run using validation only.

    Takes the same run dicts you would otherwise sort by test score, and refuses
    to look at any key starting with "test". Selection is the step where test
    contamination happens, so it is worth making structurally impossible.
    """
    if not runs:
        raise ValueError("no runs to select from")
    if metric.startswith("test"):
        raise ValueError(
            f"refusing to select on {metric!r}: selecting on test inflates the "
            "reported score. Select on a val_* metric and report test once."
        )
    return (min if lower_is_better else max)(runs, key=lambda r: r[metric])


def ensemble_mean(preds: list[np.ndarray]) -> np.ndarray:
    """
    Average predictions across seeds.

    With a flat hyperparameter response surface (which the trend run log shows),
    seed variance is the dominant source of run-to-run spread. Averaging removes
    it instead of gambling on which draw was lucky, and costs nothing beyond the
    training you already do.
    """
    if not preds:
        raise ValueError("no predictions to ensemble")
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in preds], axis=0)
    return stacked.mean(axis=0)


def ensemble_spread(preds: list[np.ndarray]) -> np.ndarray:
    """Per-sample std across seeds — a free disagreement/uncertainty signal."""
    if len(preds) < 2:
        return np.zeros_like(np.asarray(preds[0], dtype=np.float64))
    return np.stack([np.asarray(p, dtype=np.float64) for p in preds], axis=0).std(axis=0)
