"""
Rolling-origin evaluation: one model per month, each scored on the month after it.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.rolling_eval
    PYTHONPATH=. python -m technical_analysis_trainer.apps.rolling_eval --months 12 --seeds 3

Why this exists. The frozen August set holds 731 resolved labels whose average
uniqueness is 0.193 — about 141 independent observations. At that size the
accuracy confidence interval is roughly 25 points wide, so the old artifact
(AUC 0.4794) and the new one (0.4558) are indistinguishable from each other and
from a coin. One window cannot answer the question, and adding features or
changing the target while it is the only measurement means tuning against noise.

This harness answers it differently: instead of one long window, many short ones,
each with its own freshly-fitted model. For each origin T:

    train        [0 .. A)              A = T - calibration - embargo
    calibrate    [A+embargo .. T)      Platt + decision threshold
    evaluate     [T .. T+1 month)      never seen by either

No model ever touches its own evaluation month, and no evaluation month is ever
reused for a later origin's training in a way that could leak — later origins
train on it, but they are scored on a *different*, still-later month.

What comes out is a distribution rather than a point: twelve-plus monthly AUCs
with their own spread, which is what "is there an edge" actually requires. A
model with a real 0.53 edge and a model with none look identical in one window;
across a dozen they do not.

Cost. Each origin trains `--seeds` members for the fold-derived epoch budget, so
this is roughly one training run per origin. Run it in the background.

The harness also evaluates the prior-shift correction (plan 4.4) alongside the
plain calibrated model, because the correction has to be judged across regimes:
a trailing base rate is a momentum bet, and a single rising window would flatter
it. Both variants are scored on identical anchors.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.core.calibration import (
    apply_prior_shift,
    best_threshold,
    brier,
    fit_platt,
    trailing_prior,
)
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.metrics import block_bootstrap_ci, effective_n
from technical_analysis_trainer.core.validation import report_dead_features
from technical_analysis_trainer.models.trend.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
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

TREND_CONFIG = "technical_analysis_trainer/models/trend/config.yaml"
# Trailing window for the prior-shift base rate, in bars (~30 days at 1h).
PRIOR_WINDOW = 720


def load_frame(cfg: dict) -> pd.DataFrame:
    df = load_dataset(cfg["data_path"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if (cfg.get("cross_market") or {}).get("enabled", False):
        df = attach_cross_market_features(df, cfg["cross_market"])
    if (cfg.get("mtf") or {}).get("enabled", True):
        df = attach_mtf_features(df, cfg["mtf"])
    return df.reset_index(drop=True)


def label_series(df: pd.DataFrame, tb: dict) -> tuple[np.ndarray, np.ndarray]:
    """Triple-barrier labels and the bar each one resolves on, over the whole frame."""
    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    sigma = lr.rolling(int(tb.get("vol_window", 72)), min_periods=int(tb.get("vol_window", 72))).std().to_numpy()
    labels, spans, _ = triple_barrier_labels(
        close.to_numpy(np.float64), df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64),
        sigma, upper_mult=float(tb.get("upper_mult", 2.5)),
        lower_mult=float(tb.get("lower_mult", 2.5)), max_holding=int(tb.get("max_holding", 24)),
    )
    return labels, np.arange(len(labels)) + np.maximum(spans, 1)


def run_origin(df, cfg, origin: int, end: int, embargo: int, cal_bars: int,
               seeds: int, epochs: int, priors: np.ndarray) -> dict | None:
    """Train up to `origin`, calibrate on the bars just before it, score [origin, end)."""
    a = origin - cal_bars - embargo
    if a < 5000:
        return None
    warmup = span_warmup(cfg)

    Xtr, ytr, wtr, feats, scaler = build_span(df, (0, a), warmup, None, cfg)
    Xcal, ycal, wcal, _, _ = build_span(df, (a + embargo, origin), warmup, scaler, cfg)
    Xte, yte, wte, _, _, bars_te = build_span(df, (origin, end), warmup, scaler, cfg, return_index=True)
    if min(len(Xtr), len(Xcal), len(Xte)) < 50 or len(np.unique(yte)) < 2:
        return None

    fold_cfg = dict(cfg["raw"])
    fold_cfg["stop_after_epoch"] = epochs
    if fold_cfg.get("class_weight") == "auto":
        b = np.bincount(ytr, minlength=2)
        fold_cfg["class_weight"] = [len(ytr) / (2 * max(b[0], 1)), len(ytr) / (2 * max(b[1], 1))]

    members = []
    for s in range(seeds):
        set_seed(cfg["seed"] + 7000 * origin % 100000 + s)
        m, _ = train_trend_lstm(X=Xtr, y=ytr, cfg=fold_cfg, w=wtr)
        members.append(m)
    ens = TrendEnsemble(members)

    cal = fit_platt(proba_up(ens, Xcal), ycal, sample_weight=wcal)
    p_cal_slice = cal(proba_up(ens, Xcal))
    thr = best_threshold(p_cal_slice, ycal)
    prior_ref = float(ycal.mean())

    p = cal(proba_up(ens, Xte))
    # `bars_te` is the actual anchor bar of each row, reported by build_span.
    # This used to assume the rows were the last len(Xte) bars of [origin, end).
    # They are not: over half the candidate bars are skipped and the survivors
    # are not contiguous, so the prior was read at the wrong bar for nearly every
    # anchor — increasingly so toward the start of the month.
    idx = bars_te
    p_shift = apply_prior_shift(p, priors[idx], prior_ref)

    maj = float(max(yte.mean(), 1 - yte.mean()))
    out = {
        "origin_ts": str(df["ts"].iloc[origin])[:10],
        "n": int(len(yte)),
        "auc": float(roc_auc_score(yte, p)),
        "majority": maj,
        "up_rate": float(yte.mean()),
        "prior_ref": prior_ref,
        "prior_used": float(np.mean(priors[idx])),
        "threshold": float(thr),
        "acc": float(((p >= thr).astype(int) == yte).mean()),
        "acc_shift": float(((p_shift >= 0.5).astype(int) == yte).mean()),
        "brier": brier(p, yte), "brier_shift": brier(p_shift, yte),
        "mean_p": float(p.mean()), "mean_p_shift": float(p_shift.mean()),
    }
    out["edge"] = out["acc"] - maj
    out["edge_shift"] = out["acc_shift"] - maj
    out["_correct"] = ((p >= thr).astype(int) == yte).astype(float)
    out["_correct_shift"] = ((p_shift >= 0.5).astype(int) == yte).astype(float)
    out["_y"] = yte
    out["_p"] = p
    # Per-anchor predictions are persisted, not just the summaries. Each rolling
    # run costs ~18 model fits; without the raw arrays every follow-up question
    # ("what does accuracy look like at 20% coverage?", "does the top decile beat
    # the baseline?") means paying that cost again. With them, the whole class of
    # coverage and meta-labelling analysis is free and reproducible from disk.
    out["p"] = [float(v) for v in p]
    out["p_shift"] = [float(v) for v in p_shift]
    out["y"] = [int(v) for v in yte]
    out["prior"] = [float(v) for v in priors[idx]]
    # The anchor bar of every row, so any later analysis can align against the
    # frame instead of re-deriving an index that cannot be re-derived.
    out["bar"] = [int(v) for v in idx]
    return out


def _month_beats(row: dict, frac: float) -> bool:
    """Did this month's model beat the trailing-prior baseline at `frac` coverage?"""
    p = np.asarray(row["p_shift"]); y = np.asarray(row["y"]); pr = np.asarray(row["prior"])
    k = max(1, int(len(p) * frac))
    sel = np.argsort(np.abs(p - 0.5))[-k:]
    acc_m = ((p[sel] >= 0.5).astype(int) == y[sel]).mean()
    acc_b = ((pr[sel] > 0.5).astype(int) == y[sel]).mean()
    return bool(acc_m > acc_b)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling-origin evaluation of the trend model")
    ap.add_argument("--months", type=int, default=12, help="number of monthly origins")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3, help="fixed epoch budget per origin")
    ap.add_argument("--cal-months", type=float, default=3.0, help="calibration slice length")
    ap.add_argument("--out", default="artifacts/rolling_eval.json")
    # Overrides the config so an A/B is one flag rather than an edit between two
    # runs — the config-drift trap that already cost a debugging session once.
    ap.add_argument("--label-autocorr", choices=("auto", "on", "off"), default="auto",
                    help="override config's label_autocorr for this run")
    # Barrier geometry override, so a candidate from apps/barrier_sweep can be
    # confirmed on this harness (36 origins) instead of on the sweep's 2 val
    # folds, without editing the config the incumbent is still serving from.
    # Re-test a prune on the harness that decides things, not on the fold-level
    # importance that only nominates candidates.
    ap.add_argument("--no-exclude", action="store_true",
                    help="ignore config feature_exclude and use every available input")
    ap.add_argument("--mult", type=float, default=None,
                    help="override triple_barrier upper/lower mult")
    ap.add_argument("--holding", type=int, default=None,
                    help="override triple_barrier max_holding")
    a = ap.parse_args()

    with open(TREND_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    clip_lo, clip_hi = resolve_feature_scale_clip(raw)
    tb = dict(raw.get("triple_barrier") or {})
    if a.mult is not None:
        tb["upper_mult"] = tb["lower_mult"] = float(a.mult)
    if a.holding is not None:
        tb["max_holding"] = int(a.holding)
    # `raw` is what the fold config is built from, so the override has to land
    # there too or the model would train against a different target than the one
    # being scored.
    raw["triple_barrier"] = tb
    cfg = {
        "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
        "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
        "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
        "target_mode": raw.get("target_mode", "triple_barrier"), "tb_cfg": tb,
        "seed": 42, "anchor": str(raw.get("barrier_anchor", "last_observed")),
        "weight_clip_q": raw.get("weight_clip_quantile"),
        "order_flow": bool(raw.get("order_flow", True)),
        "exclude": [] if a.no_exclude else list(raw.get("feature_exclude") or []),
        "label_autocorr": bool(raw.get("label_autocorr", False))
        if a.label_autocorr == "auto" else (a.label_autocorr == "on"),
    }

    # Deliberately NOT trimmed to frozen_eval_start. Each origin's model trains
    # only on bars before its own origin, so the boundary is enforced per origin
    # by construction; withholding August here would simply throw away the one
    # month that has never been used for anything.
    df = load_frame(raw)
    labels, label_end = label_series(df, tb)
    priors = trailing_prior(labels, label_end, len(df), PRIOR_WINDOW)
    embargo = int(tb.get("max_holding", 24)) + cfg["seq_len"]
    cal_bars = int(a.cal_months * 730)

    month = 730
    n = len(df)
    origins = [n - month * (k + 1) for k in range(a.months)][::-1]

    print(f"data {df['ts'].iloc[0]:%Y-%m-%d} .. {df['ts'].iloc[-1]:%Y-%m-%d}  ({n} bars)")
    print(f"{a.months} origins, {a.seeds} seeds x {a.epochs} epochs each, "
          f"calibration {cal_bars} bars, embargo {embargo}\n")

    rows = []
    for i, o in enumerate(origins):
        end = min(o + month, n)
        r = run_origin(df, cfg, o, end, embargo, cal_bars, a.seeds, a.epochs, priors)
        if r is None:
            print(f"  [{i+1}/{len(origins)}] origin {df['ts'].iloc[o]:%Y-%m} skipped (too little history)")
            continue
        rows.append(r)
        print(f"  [{i+1}/{len(origins)}] {r['origin_ts']}  n={r['n']:4d}  AUC={r['auc']:.4f}  "
              f"acc={r['acc']*100:5.2f}%  maj={r['majority']*100:5.2f}%  edge={r['edge']*100:+6.2f}pp"
              f"   prior-shift edge={r['edge_shift']*100:+6.2f}pp")

    if not rows:
        print("no usable origins")
        return 1

    aucs = np.array([r["auc"] for r in rows])
    edges = np.array([r["edge"] for r in rows])
    edges_s = np.array([r["edge_shift"] for r in rows])
    corr = np.concatenate([r["_correct"] for r in rows])
    corr_s = np.concatenate([r["_correct_shift"] for r in rows])
    y_all = np.concatenate([r["_y"] for r in rows])
    p_all = np.concatenate([r["_p"] for r in rows])

    print(f"\n{'='*72}\n── Across {len(rows)} monthly origins ──")
    print(f"  AUC              mean {aucs.mean():.4f}  std {aucs.std(ddof=1):.4f}  "
          f"min {aucs.min():.4f}  max {aucs.max():.4f}")
    print(f"  months AUC > 0.5 {int((aucs > 0.5).sum())}/{len(aucs)}"
          "   (a coin gives about half)")
    # Sign test: how surprising is that count under "no edge"? Exact binomial.
    from math import comb
    k, m = int((aucs > 0.5).sum()), len(aucs)
    p_val = sum(comb(m, j) for j in range(k, m + 1)) / 2 ** m
    print(f"  one-sided sign test p = {p_val:.3f}"
          f"   {'(not significant)' if p_val > 0.05 else '(significant)'}")

    print(f"\n  pooled AUC       {roc_auc_score(y_all, p_all):.4f}  (n={len(y_all)})")
    lo, hi = block_bootstrap_ci(corr, block=int(tb.get("max_holding", 24)))
    print(f"  pooled accuracy  {corr.mean()*100:.2f}%  (95% block CI {lo*100:.2f}-{hi*100:.2f})")
    print(f"  mean edge        {edges.mean()*100:+.2f} pp  std {edges.std(ddof=1)*100:.2f}  "
          f"({int((edges > 0).sum())}/{len(edges)} months positive)")
    los, his = block_bootstrap_ci(corr_s, block=int(tb.get("max_holding", 24)))
    print(f"\n  with prior shift (plan 4.4):")
    print(f"  pooled accuracy  {corr_s.mean()*100:.2f}%  (95% block CI {los*100:.2f}-{his*100:.2f})")
    print(f"  mean edge        {edges_s.mean()*100:+.2f} pp  std {edges_s.std(ddof=1)*100:.2f}  "
          f"({int((edges_s > 0).sum())}/{len(edges_s)} months positive)")
    print(f"  Brier            {np.mean([r['brier'] for r in rows]):.4f} -> "
          f"{np.mean([r['brier_shift'] for r in rows]):.4f}")
    print("\n  AUC is identical under the prior shift by construction — it is a")
    print("  monotone map. Only the decision and the Brier score can move.")

    # Does the model add anything over a baseline you could actually trade?
    #
    # An AUC of ~0.54 says the ranking carries information. That is not the same
    # as a usable decision: after the prior shift the model's call turns out to
    # agree with "call whatever the trailing prior favours" on the large majority
    # of bars, so pooled accuracy mostly measures the baseline, not the model.
    #
    # The decisive test is the bars where they DISAGREE. On those the model has
    # overridden the base rate with its own signal, and its hit rate there is the
    # model's marginal contribution. Above 50% it is adding value; below, it is
    # subtracting. The unshifted calibrated probability and its fitted threshold
    # are used for the model's call, so the ranking drives the decision instead
    # of being overwritten by the prior.
    p_raw_all = np.concatenate([np.asarray(r["p"]) for r in rows])
    thr_all = np.concatenate([np.full(len(r["p"]), r["threshold"]) for r in rows])
    prior_all = np.concatenate([np.asarray(r["prior"]) for r in rows])
    call_model = (p_raw_all >= thr_all).astype(int)
    call_base = (prior_all > 0.5).astype(int)
    disagree = call_model != call_base

    print(f"\n── Marginal value over the trailing-prior baseline ──")
    print(f"  baseline = call whatever the trailing resolved base rate favours")
    print(f"  agreement        {(~disagree).mean()*100:.1f}% of {len(disagree)} bars")
    if disagree.sum() > 30:
        hit = float((call_model[disagree] == y_all[disagree]).mean())
        base_hit = float((call_base[disagree] == y_all[disagree]).mean())
        lo_d, hi_d = block_bootstrap_ci(
            (call_model[disagree] == y_all[disagree]).astype(float),
            block=int(tb.get("max_holding", 24)),
        )
        print(f"  where they differ  n={int(disagree.sum())}")
        print(f"    model right      {hit*100:.2f}%  (95% block CI {lo_d*100:.2f}-{hi_d*100:.2f})")
        print(f"    baseline right   {base_hit*100:.2f}%")
        print(f"    -> the model {'ADDS' if lo_d > 0.5 else 'does not measurably add'} value "
              "on the bars where it overrides the base rate")
    else:
        print("  too few disagreements to measure")

    print(f"\n── Accuracy by coverage (act only on the most confident bars) ──")
    print(f"  {'coverage':>10}{'n':>8}{'model':>10}{'baseline':>10}{'edge':>9}{'disagree':>10}")
    conf = np.abs(p_raw_all - thr_all)
    order = np.argsort(conf)
    for frac in (0.10, 0.20, 0.30, 0.50, 1.00):
        k = max(1, int(len(conf) * frac))
        sel = order[-k:]
        acc_m = float((call_model[sel] == y_all[sel]).mean())
        acc_b = float((call_base[sel] == y_all[sel]).mean())
        print(f"  {frac*100:9.0f}%{k:8d}{acc_m*100:9.2f}%{acc_b*100:9.2f}%"
              f"{(acc_m-acc_b)*100:+8.2f}pp{disagree[sel].mean()*100:9.1f}%")
    print("  Confidence is distance from the model's own fitted threshold, so this")
    print("  ranks by the model's conviction rather than by the base rate.")

    from technical_analysis_trainer.core.artifacts import resolve_path
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The barrier geometry is stamped on every row. Without it a consumer that
    # prices these predictions (apps/tradeability) has to guess the target from
    # config.yaml, which is wrong the moment --mult/--holding are used or the
    # config moves on — and the error is invisible, because a P&L priced at the
    # wrong barrier width still looks like a P&L.
    geom = {"upper_mult": float(tb.get("upper_mult", 2.5)),
            "lower_mult": float(tb.get("lower_mult", 2.5)),
            "max_holding": int(tb.get("max_holding", 24)),
            "vol_window": int(tb.get("vol_window", 72))}
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{**{k: v for k, v in r.items() if not k.startswith("_")},
                    "geometry": geom} for r in rows], f, indent=2)
    print(f"\nper-month rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
