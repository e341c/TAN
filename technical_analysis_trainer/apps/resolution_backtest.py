"""
Score the barrier-resolution model (plan 5.5 / 20) on the frozen evaluation tail.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.resolution_backtest
    PYTHONPATH=. python -m technical_analysis_trainer.apps.resolution_backtest \
        --start 2026-08-01 --mult 5.0 --holding 12

apps/rolling_eval and apps/resolution_eval both refit a model per origin, which
is the right way to estimate a PROCEDURE. This does the other thing: fits ONE
model on everything before the frozen boundary and scores it forward, which is
what a deployed artifact actually experiences. Section 20 measured the procedure
at AUC 0.805; this asks whether a single frozen artifact still does it.

It reads the holdout CSVs rather than the training file, because those run a
couple of days fresher (data_fetch/holdout.py keeps them current) and the whole
point is to reach today. Benchmark and macro series move with them: leaving
those on the training CSVs would feed a live BTC series alongside a SOL series
that stopped months ago, the forward-fill trap apps/backtest.py warns about.

Training stops an embargo before the boundary, so no training label can resolve
into the scored window.

There is no ONNX artifact for this target yet — it is a research result, not a
deployed model — so the fit happens here rather than being loaded.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.apps.backtest import _override_cross_market_paths
from technical_analysis_trainer.apps.rolling_eval import TREND_CONFIG, load_frame
from technical_analysis_trainer.core.metrics import block_bootstrap_ci
from technical_analysis_trainer.core.protocol import frozen_eval_start
from technical_analysis_trainer.data_fetch.holdout import ensure_holdout
from technical_analysis_trainer.models.trend.features import resolve_feature_scale_clip
from technical_analysis_trainer.models.trend.model import TrendEnsemble, train_trend_lstm
from technical_analysis_trainer.models.trend.train_v2 import (
    build_span,
    proba_up,
    set_seed,
    span_warmup,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen-tail evaluation of the resolution model")
    ap.add_argument("--start", default=None,
                    help="ISO date to start scoring (default: frozen_eval_start)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mult", type=float, default=None)
    ap.add_argument("--holding", type=int, default=None)
    ap.add_argument("--data", choices=("holdout", "train"), default="holdout",
                    help="holdout runs ~2 days fresher; train reaches back to 2021, "
                         "which matters because the holdout cache is only ~6 months deep")
    a = ap.parse_args()

    with open(TREND_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    tb = dict(raw.get("triple_barrier") or {})
    if a.mult is not None:
        tb["upper_mult"] = tb["lower_mult"] = float(a.mult)
    if a.holding is not None:
        tb["max_holding"] = int(a.holding)
    raw["triple_barrier"] = tb

    # Point every series at the holdout cache, which runs fresher than the
    # training CSV. Cross-market paths move together or not at all.
    if a.data == "holdout":
        h = ensure_holdout()
        raw["data_path"] = h.rel("btc")
        _override_cross_market_paths(raw, {k: h.rel(k) for k in ("eth", "sol", "dxy", "risk_proxy")
                                           if h.rel(k)})

    clip_lo, clip_hi = resolve_feature_scale_clip(raw)
    cfg = {
        "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
        "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
        "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
        "target_mode": "triple_barrier", "tb_cfg": tb, "seed": 42,
        "anchor": str(raw.get("barrier_anchor", "last_observed")),
        "weight_clip_q": raw.get("weight_clip_quantile"),
        "order_flow": bool(raw.get("order_flow", True)),
        "exclude": list(raw.get("feature_exclude") or []),
        "label_autocorr": False, "label_mode": "resolution",
    }

    df = load_frame(raw)
    ts = pd.to_datetime(df["ts"], utc=True)
    start_ts = pd.Timestamp(a.start, tz="UTC") if a.start else frozen_eval_start()
    split = int((ts < start_ts).sum())
    H = int(tb.get("max_holding", 24))
    embargo = H + cfg["seq_len"]
    warmup = span_warmup(cfg)

    print(f"data     {ts.iloc[0]:%Y-%m-%d %H:%M} .. {ts.iloc[-1]:%Y-%m-%d %H:%M}  ({len(df)} bars)")
    print(f"target   barrier touched within {H} bars at ±{tb.get('upper_mult')}σ")
    print(f"train    bars [0, {split - embargo})  -> {ts.iloc[split - embargo - 1]:%Y-%m-%d %H:%M}"
          f"   (embargo {embargo})")
    print(f"score    bars [{split}, {len(df)})    -> {start_ts:%Y-%m-%d} .. {ts.iloc[-1]:%Y-%m-%d %H:%M}\n")

    Xtr, ytr, wtr, feats, scaler = build_span(df, (0, split - embargo), warmup, None, cfg)
    Xte, yte, wte, _, _, bars = build_span(df, (split, len(df)), warmup, scaler, cfg,
                                           return_index=True)
    if len(Xte) < 30 or len(np.unique(yte)) < 2:
        print(f"not enough scorable anchors ({len(Xte)})")
        return 1

    fold_cfg = dict(raw)
    fold_cfg["stop_after_epoch"] = a.epochs
    if fold_cfg.get("class_weight") == "auto":
        b = np.bincount(ytr, minlength=2)
        fold_cfg["class_weight"] = [len(ytr) / (2 * max(b[0], 1)), len(ytr) / (2 * max(b[1], 1))]

    members = []
    for s in range(a.seeds):
        set_seed(cfg["seed"] + s)
        m, _ = train_trend_lstm(X=Xtr, y=ytr, cfg=fold_cfg, w=wtr)
        members.append(m)
    p = proba_up(TrendEnsemble(members), Xte)

    base = float(yte.mean())
    lo, hi = block_bootstrap_ci(yte.astype(float), block=H)
    print(f"{'='*72}\n── Resolution on the frozen tail ──")
    print(f"  train rows      {len(Xtr)}   scored anchors {len(yte)}")
    print(f"  first / last    {ts.iloc[bars.min()]:%Y-%m-%d %H:%M}  ..  {ts.iloc[bars.max()]:%Y-%m-%d %H:%M}")
    print(f"  base resolve    {base*100:.2f}%  (95% block CI {lo*100:.2f}-{hi*100:.2f})")
    print(f"  ROC-AUC         {roc_auc_score(yte, p):.4f}")
    print(f"  mean p          {p.mean():.4f}\n")

    print(f"  {'coverage':>10}{'n':>7}{'resolve rate':>15}{'95% block CI':>22}{'vs base':>10}")
    order = np.argsort(p)
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        sel = order[-max(1, int(len(p) * frac)):]
        rate = float(yte[sel].mean())
        l, hh = block_bootstrap_ci(yte[sel].astype(float), block=H)
        print(f"  {frac*100:9.0f}%{len(sel):7d}{rate*100:14.2f}%   [{l*100:6.2f}%,{hh*100:6.2f}%]"
              f"{(rate-base)*100:+9.2f}pp")

    # Weekly, so a single good stretch cannot carry the window.
    print(f"\n  {'week':>12}{'n':>6}{'base':>9}{'AUC':>9}{'top-decile resolve':>21}")
    # tz stripped first: to_period() drops it anyway and warns if it has to.
    wk = ts.iloc[bars].dt.tz_convert(None).dt.to_period("W")
    for w, m in pd.Series(np.arange(len(bars)), index=wk.to_numpy()).groupby(level=0):
        i = m.to_numpy()
        if len(i) < 30 or len(np.unique(yte[i])) < 2:
            continue
        k = max(1, len(i) // 10)
        top = yte[i][np.argsort(p[i])[-k:]].mean()
        print(f"  {str(w)[:10]:>12}{len(i):6d}{yte[i].mean()*100:8.1f}%"
              f"{roc_auc_score(yte[i], p[i]):9.4f}{top*100:20.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
