from __future__ import annotations

import numpy as np
import torch

from technical_analysis_trainer.core.metrics import (
    bootstrap_ci,
    coverage,
    pinball_loss,
    skill_score,
)


@torch.no_grad()
def predict_z(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Model output in volatility-normalised space, shape (N, n_quantiles)."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.empty((0, 3))


def evaluate_model(
    model,
    X: np.ndarray,
    y_z: np.ndarray,
    sigma: np.ndarray,
    last_close: np.ndarray,
    quantiles=(0.1, 0.5, 0.9),
    train_z: np.ndarray | None = None,
    har_sigma: np.ndarray | None = None,
) -> dict:
    """
    Evaluate quantile forecasts with baselines attached.

    Every headline number is reported next to the baseline it has to beat:

      * q50 vs. persistence (predict no change). If the skill score is not
        clearly positive, the median forecast is just tracking spot and the
        MAPE is measuring the asset's volatility, not the model.
      * intervals vs. an unconditional-quantile band (train-set quantiles of z
        scaled by sigma_t). Beating this means the model is genuinely reading
        the current regime, not just reproducing the average one.

    Works in log-return space for the loss metrics and price space for MAE/MAPE.
    """
    quantiles = list(quantiles)
    pred_z = predict_z(model, X)

    # Back out of normalised space into log returns
    pred_lr = pred_z * sigma[:, None]
    true_lr = y_z * sigma

    results: dict = {"n": int(len(y_z))}

    # ── Pinball, per quantile, model vs unconditional baseline ───────────────
    if train_z is not None and len(train_z) > 0:
        base_z = np.quantile(train_z, quantiles)
        base_lr = base_z[None, :] * sigma[:, None]
    else:
        base_lr = None

    # HAR-RV baseline: same distribution shape, but the scale comes from a
    # three-regressor realised-volatility regression instead of the network.
    # This is the benchmark that decides whether the LSTM is earning its keep on
    # the volatility side — if it does not beat HAR, use HAR for the scale.
    har_lr = None
    if har_sigma is not None and train_z is not None and len(train_z) > 0:
        har_lr = np.quantile(train_z, quantiles)[None, :] * har_sigma[:, None]

    for i, q in enumerate(quantiles):
        results[f"pinball_q{q}"] = pinball_loss(true_lr, pred_lr[:, i], q)
        if base_lr is not None:
            b = pinball_loss(true_lr, base_lr[:, i], q)
            results[f"pinball_q{q}_baseline"] = b
            results[f"pinball_q{q}_skill"] = skill_score(results[f"pinball_q{q}"], b)
        if har_lr is not None:
            h = pinball_loss(true_lr, har_lr[:, i], q)
            results[f"pinball_q{q}_har"] = h
            results[f"pinball_q{q}_skill_vs_har"] = skill_score(
                results[f"pinball_q{q}"], h
            )

    # ── Interval calibration ─────────────────────────────────────────────────
    lo_i, hi_i = 0, len(quantiles) - 1
    nominal = quantiles[hi_i] - quantiles[lo_i]
    lo, hi = pred_lr[:, lo_i], pred_lr[:, hi_i]

    inside = (true_lr >= lo) & (true_lr <= hi)
    results["coverage"] = coverage(true_lr, lo, hi)
    results["coverage_nominal"] = float(nominal)
    results["coverage_ci95"] = bootstrap_ci(inside.astype(float))

    # Breaches split by side — both should sit near the tail probability.
    # A lopsided split is the signature of a mis-centred, not just mis-sized, band.
    below = true_lr < lo
    above = true_lr > hi
    results["breach_below_rate"] = float(np.mean(below))
    results["breach_below_nominal"] = float(quantiles[lo_i])
    results["breach_above_rate"] = float(np.mean(above))
    results["breach_above_nominal"] = float(1.0 - quantiles[hi_i])

    # Breach magnitude, as % of price
    breach_amt = np.where(below, lo - true_lr, np.where(above, true_lr - hi, 0.0))
    n_breach = int(below.sum() + above.sum())
    results["avg_breach_pct"] = (
        float(np.sum(np.abs(breach_amt)) / n_breach * 100) if n_breach else 0.0
    )

    # Mean interval width (%) — pairs with coverage; a band can only be judged
    # by the two together.
    results["avg_interval_width_pct"] = float(np.mean(hi - lo) * 100)

    # ── Point forecast (q50) in price space, vs persistence ──────────────────
    mid = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    pred_price = last_close * np.exp(pred_lr[:, mid])
    true_price = last_close * np.exp(true_lr)
    naive_price = last_close  # persistence: next price = last price

    abs_err = np.abs(pred_price - true_price)
    naive_abs_err = np.abs(naive_price - true_price)

    results["mae_q50"] = float(np.mean(abs_err))
    results["mae_naive"] = float(np.mean(naive_abs_err))
    results["mae_skill_vs_naive"] = skill_score(
        results["mae_q50"], results["mae_naive"]
    )
    results["mae_skill_ci95"] = bootstrap_ci(
        naive_abs_err - abs_err
    )  # >0 throughout = model genuinely beats persistence

    results["mape_q50"] = float(np.mean(abs_err / true_price) * 100)
    results["mape_naive"] = float(np.mean(naive_abs_err / true_price) * 100)

    # ── Direction, vs the majority-class baseline ────────────────────────────
    true_up = true_lr > 0
    pred_up = pred_lr[:, mid] > 0
    correct = (true_up == pred_up).astype(float)
    results["direction_accuracy"] = float(np.mean(correct))
    results["direction_accuracy_ci95"] = bootstrap_ci(correct)
    majority = max(float(np.mean(true_up)), float(1 - np.mean(true_up)))
    results["direction_majority_baseline"] = majority
    results["direction_edge_vs_majority"] = results["direction_accuracy"] - majority

    # Quantile crossing sanity check (should be 0 with the monotonic head)
    results["crossing_rate"] = float(np.mean(np.any(np.diff(pred_z, axis=1) < 0, axis=1)))

    return results


def format_report(r: dict) -> str:
    """Human-readable summary that puts every metric next to its baseline."""
    def pct(x):
        return f"{x * 100:.2f}%"

    lines = [
        f"n = {r['n']}",
        "",
        "── Intervals ──",
        f"  coverage        {pct(r['coverage'])}  (nominal {pct(r['coverage_nominal'])}, "
        f"95% CI {pct(r['coverage_ci95'][0])}–{pct(r['coverage_ci95'][1])})",
        f"  breach below    {pct(r['breach_below_rate'])}  (nominal {pct(r['breach_below_nominal'])})",
        f"  breach above    {pct(r['breach_above_rate'])}  (nominal {pct(r['breach_above_nominal'])})",
        f"  avg breach      {r['avg_breach_pct']:.3f}% of price",
        f"  avg width       {r['avg_interval_width_pct']:.3f}% of price",
        f"  crossings       {pct(r['crossing_rate'])}",
        "",
        "── Point forecast (q50) ──",
        f"  MAE             {r['mae_q50']:.2f}   vs naive {r['mae_naive']:.2f}",
        f"  MAPE            {r['mape_q50']:.3f}%  vs naive {r['mape_naive']:.3f}%",
        f"  skill vs naive  {r['mae_skill_vs_naive']:+.4f}  "
        f"(95% CI on per-sample gain: {r['mae_skill_ci95'][0]:+.2f}–{r['mae_skill_ci95'][1]:+.2f})",
        "",
        "── Direction ──",
        f"  accuracy        {pct(r['direction_accuracy'])}  "
        f"(95% CI {pct(r['direction_accuracy_ci95'][0])}–{pct(r['direction_accuracy_ci95'][1])})",
        f"  majority base   {pct(r['direction_majority_baseline'])}",
        f"  edge            {r['direction_edge_vs_majority'] * 100:+.2f} pp",
    ]

    skills = [(k, v) for k, v in r.items() if k.endswith("_skill")]
    if skills:
        lines += ["", "── Pinball skill vs unconditional band ──"]
        lines += [f"  {k.replace('_skill', ''):<16}  {v:+.4f}" for k, v in skills]

    har = [(k, v) for k, v in r.items() if k.endswith("_skill_vs_har")]
    if har:
        lines += ["", "── Pinball skill vs HAR-RV ──"]
        lines += [f"  {k.replace('_skill_vs_har', ''):<16}  {v:+.4f}" for k, v in har]

    return "\n".join(lines)
