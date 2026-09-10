"""
Re-sweep the triple-barrier geometry, on validation, against the fixed anchor.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.barrier_sweep
    PYTHONPATH=. python -m technical_analysis_trainer.apps.barrier_sweep --folds 2 --seeds 3

The current geometry (+-2.5 sigma, 24 bars) was chosen by a val-AUC sweep run
against the OLD label, whose barriers were anchored on a close the model could
not see — a bar of noise worth 43.9% of the barrier half-width. That choice was
made on a target with a large chunk of irreducible noise baked in, so it is due
a re-run now that `barrier_anchor: last_observed` has removed it.

Three things this tool refuses to do naively.

**Compare AUC across geometries as if they were the same problem.** They are
not: a wider barrier is a different question with a different resolve rate and a
different class balance. A geometry that scores a higher AUC may simply be
asking something easier and rarer.

**Judge a target by how predictable it is rather than by what it pays.** Section
17 of plan.txt showed the trend signal sits within a basis point or two of its
costs, so the geometry has to be chosen on economics, not on AUC. Each cell is
therefore priced: a correct call earns k*sigma and a wrong one loses it, trades
run one position at a time, and the score is net return per BAR OF CAPITAL, so a
geometry that pays double but holds three times as long loses. AUC and economics
disagree on this grid, and the report says so when they do.

**Score only the trades that happened to work out.** The model never sees
anchors whose label failed to resolve, but a live account still enters those
trades and still pays for them. A 5-sigma barrier over 12 bars resolves 31% of
the time; pricing only the resolved ones ranked it first by a factor of two,
and pricing every entered trade dropped it to fifth. Both figures are printed so
the size of that correction stays visible; the ranking uses the honest one.

**Believe the winner.** With a grid this size the best cell is expected to beat
the mean by more than a standard deviation on noise alone. The report prints
`expected_max_of_noise` over the sweep's own val AUCs, and a winner inside that
band is reported as a tie, not a discovery.

Selection reads validation only. Test is not computed here at all, which is the
strongest form of the repo's rule.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.protocol import describe_boundary, trim_to_training_span
from technical_analysis_trainer.core.validation import expected_max_of_noise, walk_forward_splits
from technical_analysis_trainer.models.trend.features import (
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


def geometry_labels(df: pd.DataFrame, mult: float, holding: int, vol_window: int):
    """(labels, spans, sigma) for a geometry, over the whole frame."""
    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    sigma = lr.rolling(vol_window, min_periods=vol_window).std().to_numpy()
    labels, spans, _ = triple_barrier_labels(
        close.to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), sigma,
        upper_mult=mult, lower_mult=mult, max_holding=holding,
    )
    return labels, spans, sigma


def resolve_rate(df: pd.DataFrame, mult: float, holding: int, vol_window: int) -> tuple[float, float]:
    """(fraction of anchors that resolve, UP share among them) for a geometry."""
    labels, _, sigma = geometry_labels(df, mult, holding, vol_window)
    ok = labels >= 0
    valid = np.isfinite(sigma) & (sigma > 0)
    denom = max(int(valid.sum()), 1)
    return float(ok.sum() / denom), (float(labels[ok].mean()) if ok.any() else float("nan"))


def sequential_pnl(p, y, bars, spans, sigma, mult: float, cost: float,
                   resolve_rate: float, holding: int, timeout_edge: float = 0.0,
                   coverages=(0.05, 0.10, 0.20, 1.00)) -> dict:
    """
    Net P&L of trading these predictions with one position open at a time.

    This is the objective the sweep is actually choosing for, and it is not a
    monotone function of AUC. A geometry earns +/- mult*sigma per resolved trade
    and ties capital up for the life of the label, so a wider barrier lowers the
    accuracy needed to clear a fixed cost (break-even is 54.6% at 1.75 sigma and
    51.6% at 5.0) while holding capital two to five times longer. Ranking cells
    by AUC, or by any per-trade figure, silently ignores that trade-off; ranking
    by return per bar of occupancy does not.

    Trades are taken in confidence order but executed in time order, skipping any
    anchor that opens while an earlier position is still live — the constraint a
    single account actually has, and the one whose absence made the overlapping
    backtest in plan.txt 17.2 look like a strategy.

    TIMEOUTS ARE THE TRAP HERE, and they bite hardest exactly where the raw
    numbers look best. `y` contains only anchors whose label RESOLVED; a 5-sigma
    barrier over 12 bars resolves 31% of the time, so scoring only those assumes
    you knew in advance which two trades in three would reach a barrier. Priced
    that way 5.0 sigma / 12 bars ranked first by a factor of two; priced honestly
    it ranks fifth.

    So every cell is reported twice:

      *_resolved   P&L over resolved anchors only — optimistic, comparable to how
                   the label was scored, and NOT a strategy return.
      *_entered    P&L per trade actually entered. A timed-out trade earns
                   `timeout_edge` x the resolved edge (0 by default: the vertical
                   barrier is roughly symmetric and the model has no measured
                   skill on it) and still pays the cost and occupies the full
                   `holding` bars.

    Rank on the entered figures. The resolved ones are kept only so the size of
    the correction stays visible.
    """
    conf = np.abs(p - 0.5)
    side = np.where(p >= 0.5, 1, -1)
    width = mult * sigma[bars]
    out = {}
    for c in coverages:
        sel = np.sort(np.argsort(conf)[-max(1, int(len(p) * c)):])
        free_at, taken = -1, []
        for j in sel:
            b = int(bars[j])
            if b < free_at:
                continue
            taken.append(j)
            free_at = b + max(int(spans[b]), 1)
        if len(taken) < 20:
            out[c] = None
            continue
        t = np.array(taken, dtype=int)
        gross = side[t] * np.where(y[t] == 1, width[t], -width[t])
        hold_resolved = float(np.mean(np.maximum(spans[bars[t]], 1)))
        g = float(gross.mean())

        # Per entered trade: only `resolve_rate` of them ever reach a barrier.
        gross_entered = resolve_rate * g + (1.0 - resolve_rate) * timeout_edge * g
        occ_entered = resolve_rate * hold_resolved + (1.0 - resolve_rate) * holding
        net_entered = gross_entered - cost
        out[c] = {
            "trades": len(t),
            "acc": float((side[t] == np.where(y[t] == 1, 1, -1)).mean()),
            # Standard error of the per-trade edge. Without it the economics
            # ranking repeats the selection error it was built to avoid: on this
            # grid the best cell by net-per-bar was estimated from 81 validation
            # trades, won by a wide margin, and then lost outright on the 36-origin
            # harness (accuracy 66.67% -> 50.62%). A P&L needs a noise band as much
            # as an AUC does.
            "gross_se": float(gross.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else float("nan"),
            "gross_per_resolved": g,
            "net_per_trade_resolved": g - cost,
            "net_per_bar_resolved": (g - cost) / hold_resolved if hold_resolved else 0.0,
            "hold_resolved": hold_resolved,
            "occupancy_entered": occ_entered,
            "net_per_trade": net_entered,
            "net_per_bar": net_entered / occ_entered if occ_entered else 0.0,
        }
    return out


def evaluate_cell(df, base_cfg: dict, mult: float, holding: int,
                  folds: int, seeds: int, epochs: int, cost: float,
                  timeout_edge: float = 0.0) -> dict | None:
    """Walk-forward val AUC and net P&L for one geometry. Test is never touched."""
    cfg = dict(base_cfg)
    cfg["tb_cfg"] = {**base_cfg["tb_cfg"], "upper_mult": mult, "lower_mult": mult,
                     "max_holding": holding}
    embargo = holding + cfg["seq_len"]
    splits = walk_forward_splits(len(df), n_folds=folds, val_ratio=0.15, test_ratio=0.15,
                                 embargo=embargo)
    warmup = span_warmup(cfg)
    vol_window = int(cfg["tb_cfg"].get("vol_window", 72))
    labels_all, spans_all, sigma_all = geometry_labels(df, mult, holding, vol_window)
    # Needed before the fold loop: the P&L per ENTERED trade depends on how often
    # this geometry resolves at all, and only resolved anchors reach the model.
    rate, up_share = resolve_rate(df, mult, holding, vol_window)

    aucs, accs, ns = [], [], []
    pnl_acc: dict[float, list[dict]] = {}
    for i, sp in enumerate(splits):
        Xtr, ytr, wtr, feats, scaler = build_span(df, sp.train, warmup, None, cfg)
        Xva, yva, _, _, _, bars_va = build_span(df, sp.val, warmup, scaler, cfg,
                                                return_index=True)
        if len(Xtr) < 500 or len(Xva) < 200 or len(np.unique(yva)) < 2:
            continue
        # The frame-level labels must be the ones the windows were built against,
        # or the P&L below would be priced off a different target than the model
        # was trained on. Cheap to assert, and it has caught an alignment bug once.
        if not np.array_equal(labels_all[bars_va], yva):
            raise AssertionError(
                f"geometry {mult}sigma/{holding}b: frame labels disagree with the "
                "windowed labels — anchor indexing is wrong"
            )
        fold_cfg = dict(cfg["raw"])
        fold_cfg["stop_after_epoch"] = epochs
        if fold_cfg.get("class_weight") == "auto":
            b = np.bincount(ytr, minlength=2)
            fold_cfg["class_weight"] = [len(ytr) / (2 * max(b[0], 1)), len(ytr) / (2 * max(b[1], 1))]
        members = []
        for s in range(seeds):
            set_seed(cfg["seed"] + 313 * i + s)
            m, _ = train_trend_lstm(X=Xtr, y=ytr, cfg=fold_cfg, w=wtr)
            members.append(m)
        p = proba_up(TrendEnsemble(members), Xva)
        aucs.append(float(roc_auc_score(yva, p)))
        accs.append(float(((p >= 0.5).astype(int) == yva).mean()))
        ns.append(len(yva))
        for c, v in sequential_pnl(p, yva, bars_va, spans_all, sigma_all, mult, cost,
                                   resolve_rate=rate, holding=holding,
                                   timeout_edge=timeout_edge).items():
            if v is not None:
                pnl_acc.setdefault(c, []).append(v)

    if not aucs:
        return None
    up = up_share
    acc = float(np.mean(accs))
    mean_hold = float(np.mean(np.maximum(spans_all[labels_all >= 0], 1)))
    width = mult * float(np.nanmean(sigma_all))
    pnl = {c: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}
           for c, vs in pnl_acc.items()}
    return {
        "mult": mult, "holding": holding,
        "auc_val": float(np.mean(aucs)), "auc_val_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "acc_val": acc, "n_val": int(np.mean(ns)), "folds_used": len(aucs),
        "resolve_rate": rate, "up_share": up,
        "mean_hold": mean_hold, "width": width,
        # Accuracy that exactly covers `cost` at this barrier width. A geometry
        # whose break-even sits above what the model can deliver is unusable no
        # matter how good its AUC looks.
        "breakeven_acc": 0.5 + cost / (2 * width),
        "pnl": pnl,
        # Ranking aid across geometries, not a P&L estimate. See the module docstring.
        "edge_sigma": (2 * acc - 1) * mult * rate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Triple-barrier geometry sweep on validation")
    ap.add_argument("--mults", type=float, nargs="*", default=[1.75, 2.5, 3.5])
    ap.add_argument("--holdings", type=int, nargs="*", default=[6, 12, 24, 48])
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--cost-bp", type=float, default=8.0,
                    help="round-trip cost in bp used to price each geometry")
    ap.add_argument("--timeout-edge", type=float, default=0.0,
                    help="fraction of the resolved edge earned on timed-out trades "
                         "(0 = the conservative default; the model has no measured "
                         "skill on the vertical barrier)")
    ap.add_argument("--out", default="artifacts/barrier_sweep.json")
    a = ap.parse_args()

    with open(TREND_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    clip_lo, clip_hi = resolve_feature_scale_clip(raw)
    base_cfg = {
        "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
        "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
        "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
        "target_mode": "triple_barrier", "tb_cfg": raw.get("triple_barrier") or {},
        "seed": 42, "anchor": str(raw.get("barrier_anchor", "last_observed")),
        "weight_clip_q": raw.get("weight_clip_quantile"),
        "order_flow": bool(raw.get("order_flow", True)),
        "exclude": list(raw.get("feature_exclude") or []),
        "label_autocorr": bool(raw.get("label_autocorr", False)),
    }

    df = load_dataset(raw["data_path"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    print(f"  {describe_boundary(df)}")
    df = trim_to_training_span(df, "trend train data")
    if (base_cfg["cross_cfg"] or {}).get("enabled", False):
        df = attach_cross_market_features(df, base_cfg["cross_cfg"])
    if (base_cfg["mtf_cfg"] or {}).get("enabled", True):
        df = attach_mtf_features(df, base_cfg["mtf_cfg"])

    grid = list(itertools.product(a.mults, a.holdings))
    cost = a.cost_bp / 1e4
    print(f"anchor={base_cfg['anchor']}  {len(grid)} cells x {a.folds} folds x {a.seeds} seeds")
    print(f"cost {a.cost_bp:.0f} bp round trip; P&L is sequential (one position at a time), "
          f"on VALIDATION only\n")
    print(f"  {'k*sigma':>8}{'hold':>6}{'val AUC':>9}{'val acc':>9}{'resolve':>9}"
          f"{'width':>8}{'BE acc':>8}"
          f"{'net/res@10%':>14}{'net/entered':>14}{'net/bar':>12}{'trades':>7}")

    rows = []
    for mult, holding in grid:
        r = evaluate_cell(df, base_cfg, mult, holding, a.folds, a.seeds, a.epochs, cost,
                          timeout_edge=a.timeout_edge)
        if r is None:
            print(f"  {mult:8.2f}{holding:6d}   skipped (spans too small)")
            continue
        rows.append(r)
        c10 = r["pnl"].get(0.10)
        cells = (f"{c10['net_per_trade_resolved']*100:+13.4f}%"
                 f"{c10['net_per_trade']*100:+13.4f}%{c10['net_per_bar']*1e4:10.2f}bp"
                 f"{c10['trades']:7.0f}" if c10 else f"{'—':>14}{'—':>14}{'—':>12}{'—':>7}")
        print(f"  {mult:8.2f}{holding:6d}{r['auc_val']:9.4f}{r['acc_val']:9.4f}"
              f"{r['resolve_rate']*100:8.1f}%{r['width']*100:7.3f}%"
              f"{r['breakeven_acc']*100:7.2f}%{cells}")

    if not rows:
        print("no usable cells")
        return 1

    aucs = np.array([r["auc_val"] for r in rows])
    best = max(rows, key=lambda r: r["auc_val"])
    runner = sorted(rows, key=lambda r: -r["auc_val"])[1] if len(rows) > 1 else None
    band = expected_max_of_noise(float(aucs.mean()), float(aucs.std(ddof=1)), len(aucs))
    current = next((r for r in rows if r["mult"] == 2.5 and r["holding"] == 24), None)

    print(f"\n  best by val AUC : {best['mult']}sigma / {best['holding']} bars  ({best['auc_val']:.4f})")
    if runner:
        print(f"  runner-up       : {runner['mult']}sigma / {runner['holding']} bars  ({runner['auc_val']:.4f})"
              f"   margin {best['auc_val'] - runner['auc_val']:+.4f}")
    print(f"  sweep mean {aucs.mean():.4f} std {aucs.std(ddof=1):.4f}; best of {len(aucs)} "
          f"from noise alone would be ~{band:.4f}")
    if best["auc_val"] <= band or (runner and best["auc_val"] - runner["auc_val"] < 0.005):
        print("  ⚠ the winner is inside the noise band, or within the 0.005 seed-noise")
        print("    floor of the runner-up. Treat this as a tie and keep the incumbent:")
        print("    changing the target on this evidence is exactly the selection error")
        print("    core/validation.py exists to prevent.")
    if current:
        print(f"\n  incumbent (2.5sigma/24): val AUC {current['auc_val']:.4f}  "
              f"resolve {current['resolve_rate']*100:.1f}%  edge_sigma {current['edge_sigma']:.4f}")
    by_edge = max(rows, key=lambda r: r["edge_sigma"])
    print(f"  best by edge_sigma: {by_edge['mult']}sigma / {by_edge['holding']} bars "
          f"({by_edge['edge_sigma']:.4f}) — a ranking aid, not a P&L estimate")

    # ── The selection that matters ───────────────────────────────────────────
    # AUC says which geometry is most predictable; net-per-bar says which one
    # pays for its costs and its capital. They do not have to agree, and when
    # they disagree the second is the one a strategy lives or dies by.
    priced = [r for r in rows if r["pnl"].get(0.10)]
    if priced:
        best_pnl = max(priced, key=lambda r: r["pnl"][0.10]["net_per_bar"])
        print(f"\n  ── Ranked by net return per bar of capital, 10% coverage, "
              f"{a.cost_bp:.0f} bp, PER ENTERED TRADE ──")
        for r in sorted(priced, key=lambda r: -r["pnl"][0.10]["net_per_bar"])[:6]:
            c = r["pnl"][0.10]
            se = c.get("gross_se", float("nan"))
            print(f"    {r['mult']:.2f}sigma /{r['holding']:3d}b   "
                  f"net/bar {c['net_per_bar']*1e4:+7.2f} bp   "
                  f"net/entered {c['net_per_trade']*100:+.4f}%   "
                  f"gross {c['gross_per_resolved']*100:+.4f}% ± {se*100:.4f}  "
                  f"(t={c['gross_per_resolved']/se:5.2f}, n={c['trades']:.0f})")

        # A P&L estimated from a few dozen trades needs the same scepticism the
        # AUC band above gets. If the leader's edge is not itself distinguishable
        # from zero, the ranking is ordering noise.
        cbest = best_pnl["pnl"][0.10]
        t_best = cbest["gross_per_resolved"] / cbest["gross_se"] if cbest.get("gross_se") else float("nan")
        if not (abs(t_best) > 2):
            print(f"\n    ⚠ the leader's own gross edge is within noise (t={t_best:.2f} on "
                  f"{cbest['trades']:.0f} trades).")
            print("      Ranking on it is selection on noise. Confirm any candidate on")
            print("      apps/rolling_eval (36 origins) before touching config.yaml —")
            print("      the first time this was skipped, the winner reversed completely.")
        if best_pnl["pnl"][0.10]["net_per_trade"] <= 0:
            print("\n    ⚠ every geometry is at or below break-even after costs. The")
            print("      barrier geometry is not the binding constraint — accuracy is.")
        elif (best_pnl["mult"], best_pnl["holding"]) != (best["mult"], best["holding"]):
            print(f"\n    note: best by AUC ({best['mult']}sigma/{best['holding']}b) is NOT best by")
            print("      economics. AUC does not price the barrier width or the hold time.")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
