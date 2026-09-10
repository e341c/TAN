"""
Probability calibration, fitted on held-out data and carried in the artifact.

A classifier can rank well and still put its probabilities in the wrong place.
The trend model measured on the 2026-08 holdout is the clean example: mean
p(UP) = 0.481 against a realised UP rate of 0.612, a gap of +0.131. Every
decision taken at a fixed 0.5 threshold inherits that gap, and the model
predicted DOWN on 63% of a window where UP happened 61% of the time.

Platt scaling rather than isotonic, deliberately. Isotonic needs a lot of data
to beat a sigmoid and will happily reproduce a non-monotone wiggle it saw in a
few hundred calibration points; here it would be fitted on a slice with an
effective sample size in the low hundreds. A two-parameter sigmoid on the logit
cannot invent structure, which is what you want from a correction.

What this does and does not fix, stated plainly so a calibration change is never
read as a discovery: it moves accuracy, Brier score and the decision threshold.
It cannot move ROC-AUC — a monotone transform leaves the ranking untouched. If
AUC is at chance, calibrating it produces a well-calibrated coin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass(frozen=True)
class Calibrator:
    """
    p_calibrated = sigmoid(a * logit(p_raw) + b), optionally re-centred on the
    current base rate.

    `prior_ref` is the UP rate on the slice the Platt fit was made on. When a
    caller supplies the base rate prevailing now, `shift()` moves the
    probabilities from the reference prior onto it. Measured across 18 monthly
    rolling origins (2025-03 .. 2026-08): pooled accuracy 52.47% -> 56.69%,
    Brier 0.2499 -> 0.2450, mean edge vs the monthly majority -5.15pp -> -0.87pp.
    Set `prior_shift=False` in the bundle to disable it without retraining.
    """

    a: float
    b: float
    n: int = 0
    threshold: float = 0.5
    prior_ref: float = 0.5
    prior_shift: bool = False

    def __call__(self, p) -> np.ndarray:
        return _sigmoid(self.a * _logit(p) + self.b)

    def shift(self, p, prior_now) -> np.ndarray:
        """Calibrated probabilities re-centred onto `prior_now`, if enabled."""
        if not self.prior_shift or prior_now is None:
            return np.asarray(self(p), dtype=np.float64)
        return apply_prior_shift(self(p), prior_now, self.prior_ref)

    def cut(self, prior_now=None) -> float:
        """
        Decision threshold to use with `shift()`.

        The fitted threshold belongs to the unshifted scale; once the
        probabilities are re-centred, 0.5 is the matching cut. Keeping the
        fitted one after shifting would apply the correction twice.
        """
        return 0.5 if (self.prior_shift and prior_now is not None) else self.threshold

    def to_dict(self) -> dict:
        return {"kind": "platt", "a": float(self.a), "b": float(self.b),
                "n": int(self.n), "threshold": float(self.threshold),
                "prior_ref": float(self.prior_ref), "prior_shift": bool(self.prior_shift)}

    @staticmethod
    def identity() -> "Calibrator":
        return Calibrator(a=1.0, b=0.0, n=0, threshold=0.5)

    @staticmethod
    def from_dict(d: dict | None) -> "Calibrator":
        if not isinstance(d, dict) or d.get("kind") != "platt":
            return Calibrator.identity()
        return Calibrator(
            a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)),
            n=int(d.get("n", 0)), threshold=float(d.get("threshold", 0.5)),
            prior_ref=float(d.get("prior_ref", 0.5)),
            prior_shift=bool(d.get("prior_shift", False)),
        )


def fit_platt(p_raw, y, sample_weight=None, max_iter: int = 200) -> Calibrator:
    """
    Fit a, b by weighted logistic regression of y on logit(p_raw).

    Newton-Raphson on two parameters: small, deterministic, and no sklearn
    dependency in the artifact path. Falls back to the identity if the data
    cannot support a fit (one class, degenerate probabilities, no convergence),
    because an identity calibrator is merely useless whereas a diverged one is
    actively wrong.
    """
    z = _logit(p_raw)
    y = np.asarray(y, dtype=np.float64).ravel()
    w = (np.ones_like(y) if sample_weight is None
         else np.asarray(sample_weight, dtype=np.float64).ravel())
    if len(y) < 30 or len(np.unique(y)) < 2 or not np.isfinite(z).all():
        return Calibrator.identity()

    a, b = 1.0, 0.0
    for _ in range(max_iter):
        p = _sigmoid(a * z + b)
        r = w * (p - y)                 # gradient residual
        s = w * p * (1.0 - p)           # Hessian weights
        g = np.array([np.sum(r * z), np.sum(r)])
        H = np.array([
            [np.sum(s * z * z), np.sum(s * z)],
            [np.sum(s * z),     np.sum(s)],
        ])
        # Ridge term keeps the Hessian invertible when p saturates.
        H[0, 0] += 1e-9
        H[1, 1] += 1e-9
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return Calibrator.identity()
        a, b = a - step[0], b - step[1]
        if not np.isfinite([a, b]).all():
            return Calibrator.identity()
        if np.max(np.abs(step)) < 1e-10:
            break

    return Calibrator(a=float(a), b=float(b), n=int(len(y)))


def best_threshold(p, y, grid=None) -> float:
    """
    Decision threshold that maximises accuracy on the calibration data.

    Reported and stored rather than assumed: 0.5 is only the accuracy-optimal
    cut when the calibrated probabilities and the serving base rate agree, and
    the trend model's base rate moved from 50.6% to 62.7% between its training
    span and the following quarter.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.int64).ravel()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    grid = np.arange(0.30, 0.7001, 0.005) if grid is None else np.asarray(grid)
    accs = [(float(((p >= t).astype(int) == y).mean()), float(t)) for t in grid]
    best_acc = max(a for a, _ in accs)
    # Among ties prefer the threshold closest to 0.5 — a tie broken toward an
    # extreme is usually one sample of noise, not a decision boundary.
    return min((t for a, t in accs if a == best_acc), key=lambda t: abs(t - 0.5))


def trailing_prior(
    labels: np.ndarray,
    label_end: np.ndarray,
    n_rows: int,
    window: int,
    min_count: int = 200,
    fallback: float = 0.5,
) -> np.ndarray:
    """
    Base rate of resolved labels available at each bar, as a series over time.

    The trend model's deficit on the frozen August set was almost entirely a
    base-rate mismatch: it was calibrated where UP ran at 51.9% and served where
    it ran at 60.7%. Correcting that needs an estimate of the *current* prior —
    and the only honest one uses labels that have already resolved.

    That is the subtlety this function exists for. A triple-barrier label at bar
    j is not known until bar `label_end[j]`, up to `max_holding` bars later. A
    naive rolling mean of `labels[j]` over the trailing window would use outcomes
    from the future of the bar it is meant to inform. Here each label is
    accumulated at the bar it RESOLVES on, so the value at bar t is built only
    from information a live system would have had at t.

    Returns an array of length `n_rows`; bars with fewer than `min_count`
    resolved labels behind them get `fallback`.
    """
    resolved = labels >= 0
    ends = np.clip(label_end, 0, n_rows - 1).astype(np.int64)

    # Accumulate each label at its resolution bar, not its anchor bar.
    ups = np.zeros(n_rows, dtype=np.float64)
    cnt = np.zeros(n_rows, dtype=np.float64)
    np.add.at(ups, ends[resolved], labels[resolved].astype(np.float64))
    np.add.at(cnt, ends[resolved], 1.0)

    cum_u = np.concatenate([[0.0], np.cumsum(ups)])
    cum_c = np.concatenate([[0.0], np.cumsum(cnt)])
    lo = np.maximum(0, np.arange(n_rows) - window + 1)
    hi = np.arange(n_rows) + 1
    u = cum_u[hi] - cum_u[lo]
    c = cum_c[hi] - cum_c[lo]

    out = np.full(n_rows, fallback, dtype=np.float64)
    ok = c >= min_count
    out[ok] = u[ok] / c[ok]
    # Never let the correction be driven by a degenerate window.
    return np.clip(out, 0.05, 0.95)


def apply_prior_shift(p, prior_now, prior_ref: float) -> np.ndarray:
    """
    Re-centre calibrated probabilities from a reference base rate onto a current one.

    The standard prior-shift correction: add the difference of log-odds to the
    logit. Monotone, so it moves accuracy and Brier but leaves AUC untouched —
    the same caveat as Platt scaling above.

    A warning worth keeping attached to the code: a trailing base rate is itself
    a momentum bet. When the recent prior is high because the market has been
    rising, shifting toward it buys the trend. That can look like skill on a
    trending evaluation window and cost real money on a turn, so this belongs in
    a rolling evaluation across regimes before it belongs in production.

    That warning turned out to be the right one. The correction was accepted on
    18 rolling months (accuracy +4.48 pp, p = 0.037; Brier +0.0054, p = 0.0004)
    and is now DISABLED by default, because that evaluation was reading the prior
    ~47 bars ahead of the anchor it was correcting — far enough for the prior to
    have absorbed the outcome of the very label being predicted. Re-measured over
    36 months with the alignment fixed (models/trend/test_spans.py):

        accuracy  +1.87 pp   95% CI [-1.03, +4.77]     p = 0.21   19/36
        Brier     -0.0013    95% CI [-0.0043, +0.0017] p = 0.40   11/36

    The Brier effect changed sign. On its own the trailing prior scores a
    per-month AUC of 0.4217 (5 of 36 months above 0.5) against these labels, so
    at this horizon it points the decision the wrong way more often than not.

    The function is kept — prior shift is a standard correction and the artifact
    bundle can still enable it — but it is off unless something demonstrates it.
    """
    prior_now = np.clip(np.asarray(prior_now, dtype=np.float64), 1e-6, 1 - 1e-6)
    ref = float(np.clip(prior_ref, 1e-6, 1 - 1e-6))
    shift = np.log(prior_now / (1 - prior_now)) - np.log(ref / (1 - ref))
    return _sigmoid(_logit(p) + shift)


def brier(p, y) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def reliability_table(p, y, bins: int = 5) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, n) per equal-count bin, for reporting."""
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(p) < bins:
        return []
    order = np.argsort(p)
    out = []
    for k in range(bins):
        sel = order[int(k * len(order) / bins): int((k + 1) * len(order) / bins)]
        if len(sel):
            out.append((float(p[sel].mean()), float(y[sel].mean()), int(len(sel))))
    return out
