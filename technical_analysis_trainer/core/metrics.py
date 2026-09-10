"""Lightweight metrics helpers (no torch dependency)."""
from __future__ import annotations

import numpy as np


def pinball_loss(y_true, y_pred, q: float) -> float:
    err = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(
        y_pred, dtype=np.float64
    ).ravel()
    return float(np.mean(np.maximum((q - 1.0) * err, q * err)))


def coverage(y_true, lo, hi) -> float:
    """Fraction of observations falling inside [lo, hi]."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    return float(np.mean((y >= lo) & (y <= hi)))


def bootstrap_ci(
    values,
    stat=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a statistic over a 1-D sample.

    Use this on any headline number before reading meaning into it — at a few
    hundred samples the CI on an accuracy or a coverage is very wide.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([stat(v[i]) for i in idx], dtype=np.float64)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def effective_n(spans, valid=None) -> float:
    """
    Independent-observation count behind a set of overlapping labels.

    Triple-barrier labels share their outcome path: a label with a 6-bar span
    overlaps the next five. Counting them as independent makes every CI too
    narrow and every significance test too easy to pass. Average uniqueness
    (Lopez de Prado) is the fraction of each label's life it does not share, so
    its sum is the effective sample size.

    Measured on the 2026-08 trend holdout: 389 resolved anchors, mean uniqueness
    0.195, effective n ~ 76. Reporting 389 there overstates precision by ~2.3x.
    """
    from technical_analysis_trainer.models.trend.features import average_uniqueness

    s = np.asarray(spans, dtype=np.int64).ravel()
    v = np.ones(len(s), dtype=bool) if valid is None else np.asarray(valid, dtype=bool).ravel()
    if len(s) == 0:
        return 0.0
    return float(average_uniqueness(s, v)[v].sum())


def block_bootstrap_ci(
    values,
    block: int,
    stat=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile CI that resamples contiguous blocks instead of single points.

    `bootstrap_ci` assumes independence. For anything indexed by time here —
    triple-barrier accuracy, interval coverage, direction hit rate — neighbouring
    samples share both their label path and their market regime, so the i.i.d.
    resample produces an interval that is roughly sqrt(1/uniqueness) too narrow.

    Resampling blocks of `block` consecutive observations keeps that local
    dependence intact. Set `block` to the label's maximum span (max_holding for
    triple barrier, the horizon otherwise); a block shorter than the overlap
    still understates the width.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    b = max(1, min(int(block), n))
    n_blocks = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    # Moving-block bootstrap: draw block start positions, concatenate, trim to n.
    starts = rng.integers(0, n - b + 1, size=(n_boot, n_blocks))
    offsets = np.arange(b)
    stats = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = (starts[k][:, None] + offsets).ravel()[:n]
        stats[k] = stat(v[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def har_rv_forecast(
    log_returns,
    windows: tuple[int, int, int] = (1, 24, 168),
    fit_slice: slice | None = None,
) -> np.ndarray:
    """
    HAR-RV: forecast next-bar realised volatility from short/medium/long
    realised-vol components (Corsi 2009).

    This is the baseline any learned volatility model has to clear. It is a
    three-regressor OLS and it is notoriously hard to beat, so if the LSTM's
    sigma does not improve on it, the honest move is to use HAR-RV for the scale
    and let the network model only the distribution's shape.

    Returns a per-row sigma forecast aligned to `log_returns` (NaN during warm-up).
    """
    r = np.asarray(log_returns, dtype=np.float64).ravel()
    n = len(r)
    # A single non-finite value (the first diff is always NaN) would propagate
    # through the cumulative sums below and NaN out every downstream row.
    finite = np.isfinite(r)
    r = np.where(finite, r, 0.0)
    sq = r**2

    # Realised variance over each horizon, using only past data at each row.
    comps = []
    for w in windows:
        rv = np.full(n, np.nan)
        csum = np.cumsum(np.insert(sq, 0, 0.0))
        rv[w - 1 :] = (csum[w:] - csum[:-w]) / w
        comps.append(np.sqrt(rv))
    Xc = np.column_stack(comps)

    # Target: next bar's |return| as the realised-vol proxy.
    target = np.full(n, np.nan)
    target[:-1] = np.abs(r[1:])

    # Fitting needs a target; predicting does not. Keeping these separate matters
    # for the final row — it has no next bar to fit against, but it is exactly
    # the row you need a live forecast for.
    can_predict = ~np.isnan(Xc).any(axis=1)
    fit_mask = can_predict & ~np.isnan(target)
    if fit_slice is not None:
        keep = np.zeros(n, dtype=bool)
        keep[fit_slice] = True
        fit_mask &= keep

    if fit_mask.sum() < 10:
        return np.full(n, np.nan)

    A = np.column_stack([np.ones(fit_mask.sum()), Xc[fit_mask]])
    coef, *_ = np.linalg.lstsq(A, target[fit_mask], rcond=None)

    out = np.full(n, np.nan)
    A_all = np.column_stack([np.ones(can_predict.sum()), Xc[can_predict]])
    # |return| has mean sqrt(2/pi)*sigma under a normal, so rescale to a sigma.
    out[can_predict] = np.maximum(A_all @ coef, 1e-12) * np.sqrt(np.pi / 2)
    return out


def conformal_offsets(
    residuals_lo,
    residuals_hi,
    alpha_lo: float,
    alpha_hi: float,
) -> tuple[float, float]:
    """
    Split-conformal correction for a predicted interval.

    Feed calibration-set residuals (actual - predicted_quantile) and get the
    additive offsets that make empirical coverage match nominal on held-out
    data. This turns "approximately 80%" coverage into a finite-sample
    guarantee, which matters if position sizing reads the band.
    """
    rl = np.asarray(residuals_lo, dtype=np.float64).ravel()
    rh = np.asarray(residuals_hi, dtype=np.float64).ravel()
    if len(rl) == 0 or len(rh) == 0:
        return (0.0, 0.0)
    # Lower bound should sit at the alpha_lo quantile of actuals; shift it by the
    # corresponding residual quantile so that exactly alpha_lo fall below.
    return (
        float(np.quantile(rl, alpha_lo)),
        float(np.quantile(rh, 1.0 - alpha_hi)),
    )


def skill_score(loss_model: float, loss_baseline: float) -> float:
    """
    1 - model/baseline. Positive = model beats baseline, 0 = no better, <0 = worse.

    This is the number that decides whether a forecast is worth anything; a raw
    MAE/MAPE in isolation says nothing about skill.
    """
    if loss_baseline == 0:
        return float("nan")
    return float(1.0 - loss_model / loss_baseline)
