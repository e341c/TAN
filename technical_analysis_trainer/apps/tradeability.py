"""
Is the signal tradeable? Turns a rolling evaluation into P&L, net of costs.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.tradeability
    PYTHONPATH=. python -m technical_analysis_trainer.apps.tradeability \
        --json artifacts/rolling_eval_fixed_noac.json --cost-bp 8

AUC answers "does the ranking carry information". It does not answer "would this
have made money", and on this model the two give opposite answers. This closes
that gap, because the question gets asked after every change and deserves a
repeatable answer rather than a fresh spreadsheet each time.

Three things separate a good AUC here from a strategy, and each is a column
below:

  1. **Costs.** A triple-barrier win pays about +2.5*sigma and a loss costs the
     same, so with a ~1.14% half-width on BTC 1h, every basis point of round-trip
     cost eats ~0.22 pp of required accuracy. Break-even sits near 52.2% at 10 bp.

  2. **Capital.** The scored anchors overlap massively — mean uniqueness ~0.19,
     so 2303 scored predictions are only ~430 independent bets. Counting each
     overlapping anchor as its own trade silently assumes unlimited capital and
     unlimited leverage in whichever direction the model is currently leaning,
     which is why the overlapping P&L is dominated by a handful of trending
     months. `--sequential` (the default) instead enters only when flat, which
     is what one account can actually do.

  3. **Sample.** 480 sequential trades over three years is not many, and the
     coverage level is chosen by looking at this very table. The t-statistic and
     the block-bootstrap CI are printed so the reader can discount accordingly.

The payoff model assumes a take-profit at the barrier the label used and a stop
at its mirror, which is the trade the label literally describes. That is
optimistic about fills and pessimistic about nothing, so treat the gross numbers
as an upper bound.
"""
from __future__ import annotations

import argparse
import json
from math import erf, sqrt

import numpy as np
import yaml

from technical_analysis_trainer.apps.rolling_eval import TREND_CONFIG, load_frame
from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.metrics import block_bootstrap_ci
from technical_analysis_trainer.models.trend.features import (
    average_uniqueness,
    triple_barrier_labels,
)

COVERAGES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)


def two_sided_p(t: float) -> float:
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def load_everything(path: str):
    raw = yaml.safe_load(open(TREND_CONFIG, encoding="utf-8"))
    tb = dict(raw.get("triple_barrier") or {})
    probe = json.load(open(resolve_path(path), encoding="utf-8"))
    # Price the geometry the run was actually scored against, not whatever
    # config.yaml says today. Older files predate the stamp and fall back.
    stamped = (probe[0] or {}).get("geometry") if probe else None
    if stamped:
        tb.update(stamped)
    else:
        print("  ⚠ this file carries no geometry stamp; pricing it with config.yaml's "
              f"({tb.get('upper_mult')}σ / {tb.get('max_holding')} bars). Re-run "
              "rolling_eval if that is not the target it was scored on.")
    df = load_frame(raw)
    close = df["close"].astype(float)
    lr = np.log(close / close.shift(1))
    vw = int(tb.get("vol_window", 72))
    sigma = lr.rolling(vw, min_periods=vw).std().to_numpy()
    labels, spans, realized = triple_barrier_labels(
        close.to_numpy(np.float64), df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64), sigma,
        upper_mult=float(tb.get("upper_mult", 2.5)),
        lower_mult=float(tb.get("lower_mult", 2.5)),
        max_holding=int(tb.get("max_holding", 24)),
    )

    rows = probe
    if "bar" not in rows[0]:
        raise SystemExit(
            f"{path}: no per-anchor bar index. Re-run rolling_eval — files written "
            "before the span-alignment fix cannot be tied back to the frame."
        )
    d = {k: np.concatenate([np.asarray(r[k]) for r in rows])
         for k in ("bar", "y", "p", "prior")}
    d["thr"] = np.concatenate([np.full(len(r["p"]), r["threshold"]) for r in rows])
    d["mon"] = np.concatenate([np.full(len(r["y"]), i) for i, r in enumerate(rows)])
    d["rows"] = rows
    d["spans"] = spans
    d["mult"] = float(tb.get("upper_mult", 2.5))
    d["holding"] = int(tb.get("max_holding", 24))
    # Resolve rate over the whole frame: the share of anchors that ever reach a
    # barrier. Only these reach the model, but a live account enters the rest too.
    valid = np.isfinite(sigma) & (sigma > 0)
    d["resolve_rate"] = float((labels >= 0).sum() / max(int(valid.sum()), 1))
    d["width"] = d["mult"] * sigma[d["bar"]]
    d["uniq"] = average_uniqueness(spans, labels >= 0)
    # The label is the sign of the barrier that was hit first; the trade pays the
    # barrier width in that direction. `realized` (close of the exit bar) is kept
    # only as a cross-check — a barrier can be touched intrabar and closed through.
    d["side"] = np.where(d["p"] >= d["thr"], 1, -1)
    d["gross"] = d["side"] * np.where(d["y"] == 1, d["width"], -d["width"])
    d["gross_close"] = d["side"] * realized[d["bar"]]
    return d


def sequential(d: dict, frac: float) -> np.ndarray:
    """Indices of trades taken when only one position may be open at a time."""
    sel = np.sort(np.argsort(np.abs(d["p"] - d["thr"]))[-max(1, int(len(d["p"]) * frac)):])
    free_at, taken = -1, []
    for j in sel:
        b = int(d["bar"][j])
        if b < free_at:
            continue
        taken.append(j)
        free_at = b + max(int(d["spans"][b]), 1)
    return np.array(taken, dtype=int)


def overlapping(d: dict, frac: float) -> np.ndarray:
    return np.sort(np.argsort(np.abs(d["p"] - d["thr"]))[-max(1, int(len(d["p"]) * frac)):])


def report(d: dict, cost: float, sequential_mode: bool) -> None:
    pick = sequential if sequential_mode else overlapping
    mode = "sequential (one position at a time)" if sequential_mode else "overlapping (every anchor)"
    n_months = len(d["rows"])

    rr = d["resolve_rate"]
    print(f"\n── P&L by coverage · {mode} · cost {cost*1e4:.0f} bp round trip ──")
    print(f"{'coverage':>9}{'trades':>8}{'acc':>8}{'gross/trade':>13}{'break-even':>12}"
          f"{'net/resolved':>14}{'net/entered':>13}{'t':>7}{'p':>8}{'months+':>9}")
    for f in COVERAGES:
        t = pick(d, f)
        if len(t) < 30:
            continue
        g = d["gross"][t]
        acc = float((d["side"][t] == np.where(d["y"][t] == 1, 1, -1)).mean())
        ts = float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))
        pm = np.array([g[d["mon"][t] == i].sum() for i in range(n_months)])
        # Only `resolve_rate` of entered trades ever reach a barrier; the rest
        # time out, earn ~nothing, and still pay the cost. The model never sees
        # them, so they cannot appear in `g` — but a live account enters them.
        net_entered = rr * g.mean() - cost
        print(f"{f*100:8.0f}%{len(t):8d}{acc*100:7.2f}%{g.mean()*100:+12.4f}%"
              f"{g.mean()*1e4:9.1f} bp{(g.mean()-cost)*100:+13.4f}%{net_entered*100:+12.4f}%"
              f"{ts:7.2f}{two_sided_p(ts):8.3f}{int((pm>0).sum()):6d}/{n_months}")

    print("\n  break-even = the round-trip cost at which the gross edge is exactly consumed.")
    print("  net/resolved prices only anchors that reached a barrier — optimistic, because")
    print(f"  you cannot know which will. net/entered applies this geometry's {rr*100:.1f}%")
    print("  resolve rate, charging the timeouts their cost for ~zero gross.")
    print("  t / p are on the per-trade gross edge; months+ counts profitable months.")


def concentration(d: dict, cost: float) -> None:
    """How much of the headline comes from how few months."""
    n_months = len(d["rows"])
    for label, pick in (("overlapping", overlapping), ("sequential ", sequential)):
        t = pick(d, 0.10)
        net = d["gross"][t] - cost
        pm = np.array([net[d["mon"][t] == i].sum() * 100 for i in range(n_months)])
        top3 = float(np.sort(pm)[::-1][:3].sum())
        tot = float(pm.sum())
        share = f"{top3/tot*100:.0f}%" if abs(tot) > 1e-9 else "n/a"
        print(f"  {label} @10% coverage: total {tot:+8.2f}%  median month {np.median(pm):+.3f}%  "
              f"profitable {int((pm>0).sum())}/{n_months}  top-3 months {top3:+.2f}% ({share} of total)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Cost- and capital-aware read on a rolling evaluation")
    ap.add_argument("--json", default="artifacts/rolling_eval.json")
    ap.add_argument("--cost-bp", type=float, default=8.0,
                    help="round-trip cost in basis points (Binance perp taker ~8-10)")
    ap.add_argument("--overlapping", action="store_true",
                    help="also show the unconstrained view, which assumes unlimited capital")
    a = ap.parse_args()
    cost = a.cost_bp / 1e4

    d = load_everything(a.json)
    n = len(d["bar"])
    top10 = overlapping(d, 0.10)
    print(f"{a.json}: {n} scored anchors over {len(d['rows'])} months")
    print(f"  barrier half-width  {d['width'].mean()*100:.3f}%  (±{d['mult']}σ)   "
          f"median hold {np.median(d['spans'][d['bar']]):.0f} bars")
    print(f"  mean uniqueness     {d['uniq'][d['bar']].mean():.3f}  →  the {len(top10)} anchors in the "
          f"top decile are ~{int(len(top10)*d['uniq'][d['bar'][top10]].mean())} independent bets")
    agree = float((np.sign(d["gross"]) == np.sign(d["gross_close"])).mean())
    print(f"  barrier-fill vs close-exit sign agreement {agree*100:.2f}%  "
          "(a barrier can be touched intrabar and closed through)")

    report(d, cost, sequential_mode=True)
    if a.overlapping:
        report(d, cost, sequential_mode=False)

    print("\n── Concentration ──")
    concentration(d, cost)

    t = sequential(d, 0.10)
    lo, hi = block_bootstrap_ci(d["gross"][t] - cost, block=int(np.median(d["spans"][d["bar"][t]])))
    print(f"\n── Verdict input: sequential, top 10% coverage, {a.cost_bp:.0f} bp ──")
    print(f"  net per trade {(d['gross'][t].mean()-cost)*100:+.4f}%   "
          f"95% block CI [{lo*100:+.4f}%, {hi*100:+.4f}%]  over {len(t)} trades")
    print("  A CI that straddles zero means this has not been shown to make money,")
    print("  which is a different statement from 'it loses money'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
