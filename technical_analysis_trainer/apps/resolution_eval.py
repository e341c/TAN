"""
Plan 5.5: predict whether a barrier is TOUCHED, not which one.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.resolution_eval
    PYTHONPATH=. python -m technical_analysis_trainer.apps.resolution_eval --mult 5.0 --holding 48

Direction is exhausted. Five approaches — the direction model on three feature
sets, sixteen barrier geometries, a trailing-prior baseline and two
meta-labelling primaries — all land at AUC ~0.52 and none survives costs
(plan 17, 18, 19). Section 11 named the exit in advance: predict volatility or
touch time instead, because that family demonstrably works here. The quantile
model's interval coverage is 91.09% against a 90% nominal, still the only thing
in this repo that generalises out of sample.

"Will price travel k*sigma within H bars" is that question in the form the trend
pipeline can already ask. It is a volatility forecast wearing the triple
barrier's clothes: sigma is a trailing 72-bar estimate, so the target is
essentially "will realised volatility over the next H bars beat the recent
past", plus a path-shape term.

Two reasons it is worth money rather than just being easier:

  * It uses the anchors direction throws away. At 2.5 sigma / 24 bars, 11.1% of
    anchors never resolve and are dropped from training and from every accuracy
    number in this repo. Here they are half the signal.
  * Timeouts are what sink the wide geometries. Plan 18.3: a timed-out trade
    pays the cost and earns nothing, which is why 5.0 sigma / 12 bars looked
    twice as good as anything else until its 31% resolve rate was priced in. If
    resolution is predictable, that geometry becomes reachable — filtering to
    anchors likely to resolve raises the effective resolve rate, and the whole
    economic argument for a wide barrier turns on it.

So the number that matters is not AUC alone but the resolve rate among the
anchors the model is most confident about, against the unconditional rate.

Same 36 monthly origins, same embargo and calibration slice as
apps/rolling_eval, so results sit beside the direction model's unadjusted.
"""
from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.apps.rolling_eval import TREND_CONFIG, load_frame
from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.metrics import block_bootstrap_ci
from technical_analysis_trainer.models.trend.features import resolve_feature_scale_clip
from technical_analysis_trainer.models.trend.model import TrendEnsemble, train_trend_lstm
from technical_analysis_trainer.models.trend.train_v2 import (
    build_span,
    proba_up,
    set_seed,
    span_warmup,
)


def run_origin(df, cfg, origin: int, end: int, embargo: int, cal_bars: int,
               seeds: int, epochs: int) -> dict | None:
    a = origin - cal_bars - embargo
    if a < 5000:
        return None
    warmup = span_warmup(cfg)

    Xtr, ytr, wtr, feats, scaler = build_span(df, (0, a), warmup, None, cfg)
    Xte, yte, wte, _, _, bte = build_span(df, (origin, end), warmup, scaler, cfg,
                                          return_index=True)
    if min(len(Xtr), len(Xte)) < 50 or len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
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
    p = proba_up(TrendEnsemble(members), Xte)

    return {
        "origin_ts": str(df["ts"].iloc[origin])[:10],
        "n": int(len(yte)),
        "resolve_rate": float(yte.mean()),
        "auc": float(roc_auc_score(yte, p)),
        "p": [float(v) for v in p],
        "y": [int(v) for v in yte],
        "bar": [int(v) for v in bte],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Barrier-resolution model on rolling origins")
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--cal-months", type=float, default=3.0)
    ap.add_argument("--mult", type=float, default=None, help="barrier width override")
    ap.add_argument("--holding", type=int, default=None, help="max_holding override")
    ap.add_argument("--label-mode", default="resolution",
                    choices=("resolution", "touch_upper", "touch_lower"),
                    help="either barrier, or one side only (the directional decomposition)")
    ap.add_argument("--out", default="artifacts/resolution_eval.json")
    a = ap.parse_args()

    with open(TREND_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    clip_lo, clip_hi = resolve_feature_scale_clip(raw)
    tb = dict(raw.get("triple_barrier") or {})
    if a.mult is not None:
        tb["upper_mult"] = tb["lower_mult"] = float(a.mult)
    if a.holding is not None:
        tb["max_holding"] = int(a.holding)
    raw["triple_barrier"] = tb

    cfg = {
        "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
        "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
        "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
        "target_mode": "triple_barrier", "tb_cfg": tb,
        "seed": 42, "anchor": str(raw.get("barrier_anchor", "last_observed")),
        "weight_clip_q": raw.get("weight_clip_quantile"),
        "order_flow": bool(raw.get("order_flow", True)),
        "exclude": list(raw.get("feature_exclude") or []),
        "label_autocorr": False,
        # The one line that changes the question.
        "label_mode": a.label_mode,
    }

    df = load_frame(raw)
    embargo = int(tb.get("max_holding", 24)) + cfg["seq_len"]
    cal_bars = int(a.cal_months * 730)
    month, n = 730, len(df)
    origins = [n - month * (k + 1) for k in range(a.months)][::-1]

    print(f"target [{a.label_mode}]: within {tb.get('max_holding')} bars at "
          f"±{tb.get('upper_mult')}σ")
    print(f"{a.months} origins, {a.seeds} seeds x {a.epochs} epochs, embargo {embargo}\n")

    rows = []
    for i, o in enumerate(origins):
        end = min(o + month, n)
        r = run_origin(df, cfg, o, end, embargo, cal_bars, a.seeds, a.epochs)
        if r is None:
            print(f"  [{i+1}/{len(origins)}] skipped")
            continue
        rows.append(r)
        print(f"  [{i+1}/{len(origins)}] {r['origin_ts']}  n={r['n']:4d}  "
              f"resolve {r['resolve_rate']*100:5.1f}%  AUC={r['auc']:.4f}")

    if not rows:
        print("no usable origins")
        return 1

    aucs = np.array([r["auc"] for r in rows])
    y = np.concatenate([np.asarray(r["y"]) for r in rows])
    p = np.concatenate([np.asarray(r["p"]) for r in rows])
    k, m = int((aucs > 0.5).sum()), len(aucs)
    pv = sum(comb(m, j) for j in range(k, m + 1)) / 2 ** m

    print(f"\n{'='*74}\n── Barrier resolution over {m} monthly origins ──")
    print(f"  AUC              mean {aucs.mean():.4f}  std {aucs.std(ddof=1):.4f}  "
          f"min {aucs.min():.4f}  max {aucs.max():.4f}")
    print(f"  months AUC > 0.5 {k}/{m}   one-sided sign test p = {pv:.4f}")
    print(f"  pooled AUC       {roc_auc_score(y, p):.4f}  (n={len(y)})")
    print(f"  base resolve rate {y.mean()*100:.2f}%\n")

    print("  Resolve rate among the anchors the model is most sure will resolve —")
    print("  this is what a wide barrier needs (plan 18.3), not AUC:")
    print(f"  {'coverage':>10}{'n':>8}{'resolve rate':>15}{'95% block CI':>22}{'vs base':>10}")
    order = np.argsort(p)
    blk = int(tb.get("max_holding", 24))
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        sel = order[-max(1, int(len(p) * frac)):]
        rate = float(y[sel].mean())
        lo, hi = block_bootstrap_ci(y[sel].astype(float), block=blk)
        print(f"  {frac*100:9.0f}%{len(sel):8d}{rate*100:14.2f}%   [{lo*100:6.2f}%,{hi*100:6.2f}%]"
              f"{(rate-y.mean())*100:+9.2f}pp")

    # Per-month: does the top decile beat that month's own base rate?
    wins = 0
    for r in rows:
        pp, yy = np.asarray(r["p"]), np.asarray(r["y"])
        kk = max(1, len(pp) // 10)
        wins += float(yy[np.argsort(pp)[-kk:]].mean()) > float(yy.mean())
    pv2 = sum(comb(len(rows), j) for j in range(wins, len(rows) + 1)) / 2 ** len(rows)
    print(f"\n  top decile beat its own month's base rate in {wins}/{len(rows)} months "
          f"(p = {pv2:.4f})")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"geometry": {k2: tb.get(k2) for k2 in
                                ("upper_mult", "lower_mult", "max_holding", "vol_window")},
                   "rows": rows}, f, indent=2)
    print(f"\nper-month rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
