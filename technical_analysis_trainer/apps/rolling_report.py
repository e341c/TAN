"""
Plotly report for a rolling-origin evaluation.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.rolling_report
    PYTHONPATH=. python -m technical_analysis_trainer.apps.rolling_report \
        --json artifacts/rolling_eval_fixed_noac.json \
        --compare artifacts/rolling_eval_fixed_ac.json

`apps/backtest.py` charts one artifact over one stretch of candles, which is the
right picture for "what did this model do here" and the wrong one for "does this
model have an edge". Section 16 of plan.txt settled that the rolling evaluation
is the measurement — 36 independently-fitted monthly models — and it had no
picture at all. This is it.

What the four panels are for, in the order the question actually gets asked:

  1. Monthly AUC. The headline. One bar per origin against the 0.5 line, because
     the claim being made is "more months above the line than a coin would give"
     and that is a thing you should be able to *see* rather than take on a
     p-value. The spread is the second half of the point: months run 0.41 to
     0.69, so any single window — including the frozen holdout — says nothing.
  2. Edge against the majority baseline. Ranking (panel 1) and decisions
     (panel 2) come apart here: AUC can sit above 0.5 every month while accuracy
     stays under "always call the common class", and on this model it does.
  3. Accuracy by coverage. Where the signal actually lives. If the model is
     worth anything it is at low coverage, and this panel says whether acting
     only on its confident calls beats acting on the base rate.
  4. Reliability. Whether the probabilities mean what they say, pooled over all
     origins, against the diagonal.

Reads the JSON that `rolling_eval.py` already writes, so it costs nothing to
re-render and never retrains anything.
"""
from __future__ import annotations

import argparse
import json
from math import comb

import numpy as np
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.metrics import block_bootstrap_ci

# Matches the block length rolling_eval uses for its own CIs: one triple-barrier
# holding window, so a resampled block spans labels that share an outcome path.
BLOCK = 24


def sign_test(k: int, m: int) -> float:
    """One-sided exact binomial: P(at least k of m above 0.5) under a coin."""
    return sum(comb(m, j) for j in range(k, m + 1)) / 2 ** m


def load(path: str) -> list[dict]:
    rows = json.load(open(resolve_path(path), encoding="utf-8"))
    if not rows:
        raise SystemExit(f"{path}: no origins in file")
    if "p" not in rows[0]:
        raise SystemExit(
            f"{path}: no per-anchor arrays — this file predates the raw-prediction "
            "dump in rolling_eval. Re-run the harness to chart it."
        )
    return rows


def summary(rows: list[dict]) -> dict:
    Y = [np.asarray(r["y"]) for r in rows]
    P = [np.asarray(r["p"]) for r in rows]
    PR = [np.asarray(r["prior"]) for r in rows]
    thr = np.concatenate([np.full(len(r["p"]), r["threshold"]) for r in rows])

    auc = np.array([roc_auc_score(Y[i], P[i]) for i in range(len(rows))])
    y, p, pr = np.concatenate(Y), np.concatenate(P), np.concatenate(PR)
    k = int((auc > 0.5).sum())
    correct = ((p >= thr).astype(int) == y).astype(float)
    lo, hi = block_bootstrap_ci(correct, block=BLOCK)
    return {
        "rows": rows, "auc": auc, "y": y, "p": p, "prior": pr, "thr": thr,
        "origin": [r["origin_ts"] for r in rows],
        "acc": np.array([r["acc"] for r in rows]),
        "majority": np.array([r["majority"] for r in rows]),
        "edge": np.array([r["edge"] for r in rows]),
        "n": np.array([r["n"] for r in rows]),
        "k": k, "m": len(auc), "p_sign": sign_test(k, len(auc)),
        "pooled_auc": float(roc_auc_score(y, p)),
        "pooled_acc": float(correct.mean()), "acc_lo": lo, "acc_hi": hi,
        "prior_auc": float(roc_auc_score(y, pr)),
    }


def coverage_curve(s: dict, fracs=(0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)):
    """Accuracy when acting only on the most confident `frac` of calls."""
    call_m = (s["p"] >= s["thr"]).astype(int)
    call_b = (s["prior"] > 0.5).astype(int)
    # Confidence is distance from the model's own fitted threshold, so the
    # ranking is the model's conviction rather than the prevailing base rate.
    order = np.argsort(np.abs(s["p"] - s["thr"]))
    out = []
    for f in fracs:
        sel = order[-max(1, int(len(order) * f)):]
        out.append((f, len(sel),
                    float((call_m[sel] == s["y"][sel]).mean()),
                    float((call_b[sel] == s["y"][sel]).mean())))
    return out


def reliability(s: dict, bins: int = 10):
    """(mean predicted, observed rate, n) per equal-count bin."""
    order = np.argsort(s["p"])
    out = []
    for k in range(bins):
        sel = order[int(k * len(order) / bins): int((k + 1) * len(order) / bins)]
        if len(sel):
            out.append((float(s["p"][sel].mean()), float(s["y"][sel].mean()), int(len(sel))))
    return out


def build_figure(a: dict, b: dict | None, name_a: str, name_b: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly ROC-AUC per origin (one freshly-fitted model each)",
            "Accuracy vs the majority baseline, per month",
            "Accuracy by coverage — acting only on the most confident calls",
            "Reliability, pooled over all origins",
        ),
        vertical_spacing=0.13, horizontal_spacing=0.08,
    )

    # ── 1. Monthly AUC ───────────────────────────────────────────────────────
    # Coloured by side of 0.5 so the sign test is legible without reading it.
    fig.add_trace(go.Bar(
        x=a["origin"], y=a["auc"] - 0.5, base=0.5, name=f"{name_a} AUC",
        marker_color=["#2ca02c" if v > 0.5 else "#d62728" for v in a["auc"]],
        customdata=np.stack([a["auc"], a["n"]], axis=-1),
        hovertemplate="origin %{x}<br>AUC %{customdata[0]:.4f}<br>n=%{customdata[1]}<extra></extra>",
    ), row=1, col=1)
    if b is not None:
        fig.add_trace(go.Scatter(
            x=b["origin"], y=b["auc"], name=f"{name_b} AUC", mode="markers",
            marker=dict(symbol="diamond", size=7, color="#1f77b4"),
            hovertemplate="origin %{x}<br>AUC %{y:.4f}<extra></extra>",
        ), row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color="white", width=1), row=1, col=1)
    fig.add_hline(y=float(a["auc"].mean()), line=dict(color="#ff7f0e", width=1, dash="dash"),
                  annotation_text=f"mean {a['auc'].mean():.4f}", row=1, col=1)

    # ── 2. Accuracy vs majority ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=a["origin"], y=a["acc"], name="accuracy", mode="lines+markers",
        line=dict(color="#2ca02c"),
        hovertemplate="origin %{x}<br>accuracy %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=a["origin"], y=a["majority"], name="majority baseline", mode="lines+markers",
        line=dict(color="#d62728", dash="dot"),
        hovertemplate="origin %{x}<br>majority %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    # ── 3. Coverage ──────────────────────────────────────────────────────────
    cov = coverage_curve(a)
    xs = [f"{int(f*100)}%" for f, *_ in cov]
    fig.add_trace(go.Scatter(
        x=xs, y=[m for *_, m, _ in cov], name="model", mode="lines+markers",
        line=dict(color="#2ca02c"),
        customdata=[n for _, n, _, _ in cov],
        hovertemplate="coverage %{x}<br>model %{y:.4f}<br>n=%{customdata}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=xs, y=[bl for *_, bl in cov], name="trailing-prior baseline",
        mode="lines+markers", line=dict(color="#9467bd", dash="dot"),
        hovertemplate="coverage %{x}<br>baseline %{y:.4f}<extra></extra>",
    ), row=2, col=1)

    # ── 4. Reliability ───────────────────────────────────────────────────────
    rel = reliability(a)
    fig.add_trace(go.Scatter(
        x=[m for m, _, _ in rel], y=[o for _, o, _ in rel], name="reliability",
        mode="lines+markers", line=dict(color="#17becf"),
        customdata=[n for _, _, n in rel],
        hovertemplate="predicted %{x:.4f}<br>observed %{y:.4f}<br>n=%{customdata}<extra></extra>",
    ), row=2, col=2)
    lohi = [min(m for m, _, _ in rel), max(m for m, _, _ in rel)]
    fig.add_trace(go.Scatter(
        x=lohi, y=lohi, name="perfect calibration", mode="lines",
        line=dict(color="white", width=1, dash="dash"), showlegend=False,
    ), row=2, col=2)

    fig.update_yaxes(title_text="ROC-AUC", row=1, col=1)
    fig.update_yaxes(title_text="accuracy", row=1, col=2)
    fig.update_yaxes(title_text="accuracy", row=2, col=1)
    fig.update_yaxes(title_text="observed UP rate", row=2, col=2)
    fig.update_xaxes(title_text="coverage (most confident share of calls)", row=2, col=1)
    fig.update_xaxes(title_text="mean predicted p(UP)", row=2, col=2)

    sub = (f"{a['m']} origins · {a['origin'][0]} → {a['origin'][-1]} · n={len(a['y'])} anchors<br>"
           f"months AUC>0.5 <b>{a['k']}/{a['m']}</b> (one-sided sign test p={a['p_sign']:.4f}) · "
           f"pooled AUC {a['pooled_auc']:.4f} · "
           f"pooled accuracy {a['pooled_acc']*100:.2f}% "
           f"(95% block CI {a['acc_lo']*100:.2f}-{a['acc_hi']*100:.2f}) · "
           f"trailing-prior baseline AUC {a['prior_auc']:.4f}")
    if b is not None:
        sub += (f"<br>compare <b>{name_b}</b>: mean AUC {b['auc'].mean():.4f}, "
                f"{b['k']}/{b['m']} months >0.5, pooled {b['pooled_auc']:.4f}")

    fig.update_layout(
        template="plotly_dark",
        title=f"Rolling-origin evaluation · {name_a}<br><sub>{sub}</sub>",
        hovermode="closest", height=900, legend=dict(groupclick="togglegroup"),
        bargap=0.15,
    )
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plotly report for a rolling-origin evaluation")
    p.add_argument("--json", default="artifacts/rolling_eval.json",
                   help="rolling_eval output to chart")
    p.add_argument("--compare", default=None, help="a second run to overlay")
    p.add_argument("--out", default="artifacts/rolling_report.html")
    p.add_argument("--no-open", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    A = summary(load(a.json))
    B = summary(load(a.compare)) if a.compare else None
    name_a = a.json.split("/")[-1].replace(".json", "")
    name_b = a.compare.split("/")[-1].replace(".json", "") if a.compare else ""

    print(f"{name_a}: {A['m']} origins, n={len(A['y'])}")
    print(f"  mean monthly AUC {A['auc'].mean():.4f}  (std {A['auc'].std(ddof=1):.4f}, "
          f"min {A['auc'].min():.4f}, max {A['auc'].max():.4f})")
    print(f"  months AUC>0.5   {A['k']}/{A['m']}  one-sided sign test p={A['p_sign']:.4f}")
    print(f"  pooled AUC       {A['pooled_auc']:.4f}   trailing-prior baseline {A['prior_auc']:.4f}")
    print(f"  pooled accuracy  {A['pooled_acc']*100:.2f}%  "
          f"(95% block CI {A['acc_lo']*100:.2f}-{A['acc_hi']*100:.2f})")
    print("\n  coverage    n     model   baseline")
    for f, n, m, bl in coverage_curve(A):
        print(f"  {f*100:7.0f}% {n:6d}  {m*100:7.2f}%  {bl*100:7.2f}%")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_figure(A, B, name_a, name_b).write_html(str(out), auto_open=not a.no_open)
    print(f"\nchart → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
