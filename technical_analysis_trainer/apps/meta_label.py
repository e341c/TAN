"""
Meta-labelling (Lopez de Prado, plan 5.2): learn WHEN a simple rule is right.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.meta_label
    PYTHONPATH=. python -m technical_analysis_trainer.apps.meta_label --months 36 --seeds 5

The direction model has a real but thin ranking edge (AUC 0.5512 over 36 monthly
origins, p = 0.0006) that does not survive costs and the one-position constraint
(plan 17). The one place anything survives is low coverage, and that is exactly
what meta-labelling optimises: instead of asking the network which way price
goes, hand it a direction from a transparent rule and ask only "is this
particular signal worth taking?".

Why that is a better-conditioned problem:

  * The label is the primary rule's own hit/miss, so the network spends its
    capacity on reliability rather than on re-deriving direction from scratch.
  * It produces abstention natively. A trading bot needs "stand aside", and a
    direction model has no way to say it — its 0.5 is a coin, not a refusal.
  * Precision at low coverage is the objective, not AUC, so the metric matches
    where the edge actually is.

The primary is a plain EMA cross. Screened against the triple-barrier label over
the whole frame, the trend-following rules all sit below the 53.21% break-even
that a 1.245% barrier and 8 bp of cost imply:

    EMA24 > EMA96   51.71%  [50.72, 52.70]   <- best, still short of break-even
    EMA12 > EMA48   50.86%  [49.90, 51.88]
    mom168          51.21%  [50.27, 52.16]
    RSI14 revert    43.72%  [41.29, 46.20]   <- actively wrong

So the primary alone loses money. The whole question is whether gating lifts it
past 53.21% at a coverage worth trading, and the answer has to come from the
36-origin harness, not from a validation fold — plan 18 is what happens when a
geometry is chosen on 81 validation trades.

Evaluation mirrors apps/rolling_eval exactly (same origins, same calibration
slice, same embargo) so the numbers sit next to the direction model's without
adjustment, and the per-anchor arrays are dumped for apps/tradeability.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.metrics import block_bootstrap_ci
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
from technical_analysis_trainer.apps.rolling_eval import TREND_CONFIG, load_frame


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def primary_side(close: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """+1 long / -1 short, from an EMA cross. Uses only closed bars."""
    s = np.sign(ema(close, fast) - ema(close, slow))
    # A flat cross is not a view; carry the previous side rather than inventing one.
    s = pd.Series(np.where(s == 0, np.nan, s)).ffill().fillna(1.0).to_numpy()
    return s


def label_frame(df: pd.DataFrame, tb: dict):
    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    vw = int(tb.get("vol_window", 72))
    sigma = lr.rolling(vw, min_periods=vw).std().to_numpy()
    labels, spans, _ = triple_barrier_labels(
        close.to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), sigma,
        upper_mult=float(tb.get("upper_mult", 2.5)),
        lower_mult=float(tb.get("lower_mult", 2.5)),
        max_holding=int(tb.get("max_holding", 24)),
    )
    return labels, spans, sigma


def attach_side_channel(X: np.ndarray, side: np.ndarray) -> np.ndarray:
    """
    Append the primary's side as an extra input channel.

    The rule is a deterministic function of price and the window already carries
    price features, so the network could in principle infer it — but only by
    spending capacity on rediscovering the rule. Handing it over explicitly is
    the point of meta-labelling: the model's whole job is the conditional
    reliability of a side it is told.
    """
    ch = np.repeat(side[:, None, None].astype(np.float32), X.shape[1], axis=1)
    return np.concatenate([X, ch], axis=2)


def run_origin(df, cfg, origin: int, end: int, embargo: int, cal_bars: int,
               seeds: int, epochs: int, side_all: np.ndarray) -> dict | None:
    """Train the meta-model up to `origin`, threshold on the slice before it, score after."""
    a = origin - cal_bars - embargo
    if a < 5000:
        return None
    warmup = span_warmup(cfg)

    Xtr, ytr, wtr, feats, scaler, btr = build_span(df, (0, a), warmup, None, cfg, return_index=True)
    Xcal, ycal, wcal, _, _, bcal = build_span(df, (a + embargo, origin), warmup, scaler, cfg,
                                              return_index=True)
    Xte, yte, wte, _, _, bte = build_span(df, (origin, end), warmup, scaler, cfg,
                                          return_index=True)
    if min(len(Xtr), len(Xcal), len(Xte)) < 50:
        return None

    # Meta-label: did the primary's side match what actually happened?
    def meta(y, bars):
        s = side_all[bars]
        return (np.where(s > 0, 1, 0) == y).astype(np.int64), s

    mtr, str_ = meta(ytr, btr)
    mcal, scal = meta(ycal, bcal)
    mte, ste = meta(yte, bte)
    if len(np.unique(mte)) < 2 or len(np.unique(mtr)) < 2:
        return None

    Xtr2, Xcal2, Xte2 = (attach_side_channel(Xtr, str_), attach_side_channel(Xcal, scal),
                         attach_side_channel(Xte, ste))

    fold_cfg = dict(cfg["raw"])
    fold_cfg["stop_after_epoch"] = epochs
    if fold_cfg.get("class_weight") == "auto":
        b = np.bincount(mtr, minlength=2)
        fold_cfg["class_weight"] = [len(mtr) / (2 * max(b[0], 1)), len(mtr) / (2 * max(b[1], 1))]

    members = []
    for s in range(seeds):
        set_seed(cfg["seed"] + 7000 * origin % 100000 + s)
        m, _ = train_trend_lstm(X=Xtr2, y=mtr, cfg=fold_cfg, w=wtr)
        members.append(m)
    ens = TrendEnsemble(members)

    p_te = proba_up(ens, Xte2)          # P(the primary is right here)
    p_cal = proba_up(ens, Xcal2)

    base = float(mte.mean())            # the primary's own precision on this month
    return {
        "origin_ts": str(df["ts"].iloc[origin])[:10],
        "n": int(len(mte)),
        "primary_precision": base,
        "cal_precision": float(mcal.mean()),
        # Where a coverage threshold picked on the calibration slice would sit.
        "q90_cal": float(np.quantile(p_cal, 0.90)),
        "q80_cal": float(np.quantile(p_cal, 0.80)),
        "p": [float(v) for v in p_te],
        "m": [int(v) for v in mte],       # 1 = primary was right
        "y": [int(v) for v in yte],       # direction outcome, for reference
        "side": [float(v) for v in ste],
        "bar": [int(v) for v in bte],
    }


def precision_at(rows: list[dict], frac: float) -> tuple[float, int, float]:
    """Pooled precision when taking the top `frac` of signals by meta-probability."""
    p = np.concatenate([np.asarray(r["p"]) for r in rows])
    m = np.concatenate([np.asarray(r["m"]) for r in rows])
    k = max(1, int(len(p) * frac))
    sel = np.argsort(p)[-k:]
    return float(m[sel].mean()), len(sel), float(m.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="Meta-labelling: gate a simple rule with the network")
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--cal-months", type=float, default=3.0)
    ap.add_argument("--fast", type=int, default=24, help="primary EMA fast span")
    ap.add_argument("--slow", type=int, default=96, help="primary EMA slow span")
    ap.add_argument("--cost-bp", type=float, default=8.0)
    ap.add_argument("--out", default="artifacts/meta_label.json")
    a = ap.parse_args()

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
    labels_all, spans_all, sigma_all = label_frame(df, tb)
    side_all = primary_side(df["close"].astype(float).to_numpy(), a.fast, a.slow)
    embargo = int(tb.get("max_holding", 24)) + cfg["seq_len"]
    cal_bars = int(a.cal_months * 730)
    month, n = 730, len(df)
    origins = [n - month * (k + 1) for k in range(a.months)][::-1]

    width = float(np.nanmean(sigma_all)) * float(tb.get("upper_mult", 2.5))
    cost = a.cost_bp / 1e4
    be = 0.5 + cost / (2 * width)
    print(f"primary EMA{a.fast} vs EMA{a.slow}   barrier half-width {width*100:.3f}%")
    print(f"break-even precision at {a.cost_bp:.0f} bp = {be*100:.2f}%")
    print(f"{a.months} origins, {a.seeds} seeds x {a.epochs} epochs, embargo {embargo}\n")

    rows = []
    for i, o in enumerate(origins):
        end = min(o + month, n)
        r = run_origin(df, cfg, o, end, embargo, cal_bars, a.seeds, a.epochs, side_all)
        if r is None:
            print(f"  [{i+1}/{len(origins)}] skipped")
            continue
        rows.append(r)
        top = np.asarray(r["p"]); mm = np.asarray(r["m"])
        k = max(1, len(top) // 10)
        prec10 = float(mm[np.argsort(top)[-k:]].mean())
        print(f"  [{i+1}/{len(origins)}] {r['origin_ts']}  n={r['n']:4d}  "
              f"primary {r['primary_precision']*100:5.2f}%  gated@10% {prec10*100:5.2f}%")

    if not rows:
        print("no usable origins")
        return 1

    print(f"\n{'='*74}\n── Meta-labelling over {len(rows)} monthly origins ──")
    m_all = np.concatenate([np.asarray(r["m"]) for r in rows])
    lo, hi = block_bootstrap_ci(m_all.astype(float), block=int(tb.get("max_holding", 24)))
    print(f"  primary precision (no gate)  {m_all.mean()*100:.2f}%  "
          f"(95% block CI {lo*100:.2f}-{hi*100:.2f})   n={len(m_all)}")
    print(f"  break-even needed            {be*100:.2f}%\n")
    print(f"  {'coverage':>9}{'n':>8}{'gated precision':>18}{'95% block CI':>22}{'vs primary':>12}")
    for frac in (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        p = np.concatenate([np.asarray(r["p"]) for r in rows])
        k = max(1, int(len(p) * frac))
        sel = np.argsort(p)[-k:]
        prec = float(m_all[sel].mean())
        l, h = block_bootstrap_ci(m_all[sel].astype(float), block=int(tb.get("max_holding", 24)))
        print(f"  {frac*100:8.0f}%{k:8d}{prec*100:17.2f}%   [{l*100:6.2f}%,{h*100:6.2f}%]"
              f"{(prec-m_all.mean())*100:+11.2f}pp")

    # Per-month sign test: does gating beat the ungated primary in most months?
    wins = 0
    for r in rows:
        p = np.asarray(r["p"]); m = np.asarray(r["m"])
        k = max(1, len(p) // 10)
        wins += float(m[np.argsort(p)[-k:]].mean()) > float(m.mean())
    from math import comb
    pv = sum(comb(len(rows), j) for j in range(wins, len(rows) + 1)) / 2 ** len(rows)
    print(f"\n  gating beat the raw primary in {wins}/{len(rows)} months "
          f"(one-sided sign test p = {pv:.4f})")
    print("  Precision, not AUC, is the objective here — the direction is the rule's.")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"primary": f"EMA{a.fast}>EMA{a.slow}", "break_even": be,
                   "width": width, "rows": rows}, f, indent=2)
    print(f"\nper-month rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
