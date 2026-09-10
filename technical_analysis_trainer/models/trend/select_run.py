"""
Which trend run should be deployed, decided on validation only.

    PYTHONPATH=. python -m technical_analysis_trainer.models.trend.select_run
    PYTHONPATH=. python -m technical_analysis_trainer.models.trend.select_run --metric val_loss

Why this exists. `train_runs_v2.csv` used to record only test-side aggregates
(auc_ens, auc_gb, auc_blend, acc_ens), so the log could not support a correct
choice even in principle. What actually happened, measured on 2026-09-07 over
the twelve runs logged at that point:

    auc_ens mean 0.5243, std 0.0204
    expected_max_of_noise(0.5243, 0.0204, 12) = 0.5534
    observed best                              = 0.5586   <- run 2026-08-17T15-47-17
                                                             the one pinned in
                                                             assembly/config.yaml

The deployed run was the argmax of twelve test scores, sitting within a
whisker of where the best of twelve pure-noise draws is expected to land. This
is the same finding core/validation.py was written about after the 184-run log,
reproduced one level up: the seed lottery was fixed by ensembling, and the run
lottery replaced it.

So: selection reads `auc_val` (or `val_loss`), never a test column —
`core.validation.select_by_val` refuses the latter outright — and the noise band
over the *selection* metric is printed next to the winner, so a choice inside it
is visibly a coin flip rather than a decision.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from technical_analysis_trainer.core.validation import expected_max_of_noise, select_by_val

RUNS_CSV = os.path.join(os.path.dirname(__file__), "train_runs_v2.csv")
# Higher-is-better metrics; anything else is treated as lower-is-better.
HIGHER_IS_BETTER = {"auc_val"}


def load_runs(path: str = RUNS_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = dict(r)
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", default="auc_val", help="validation metric to select on")
    p.add_argument("--csv", default=RUNS_CSV)
    a = p.parse_args()

    runs = load_runs(a.csv)
    usable = [r for r in runs if isinstance(r.get(a.metric), float) and np.isfinite(r[a.metric])]
    if not usable:
        print(f"No run in {a.csv} records {a.metric!r}.")
        print("Runs logged before the schema change have no validation column — they")
        print("cannot be selected between, only re-run. Train at least one new run.")
        return 1

    # A validation metric is only comparable to another one measured the same
    # way. Runs before harness 2 had validation spans that began ~200 bars inside
    # their own training data, so their auc_val is inflated by an unknown amount;
    # ranking them against a correctly-measured run prefers the bug. Restrict to
    # the newest harness present rather than silently mixing.
    def harness_of(r) -> int:
        h = r.get("harness")
        try:
            return int(float(h))
        except (TypeError, ValueError):
            return 1          # unstamped == pre-fix
    newest = max(harness_of(r) for r in usable)
    stale = [r for r in usable if harness_of(r) < newest]
    if stale:
        usable = [r for r in usable if harness_of(r) == newest]
        print(f"⚠ ignoring {len(stale)} run(s) measured on an older evaluation harness "
              f"(< v{newest}).")
        print("  Their validation spans overlapped their training data, so their")
        print(f"  {a.metric} is not comparable with the current one. Re-run them under")
        print("  the current harness if you want them in the comparison.\n")
    if not usable:
        print("No run measured on the current harness. Train one before selecting.")
        return 1
    if len(usable) == 1 and stale:
        print(f"Only one run on harness v{newest}: {usable[0].get('timestamp')}.")
        print("There is nothing to select between — this is the default, not a winner.\n")

    higher = a.metric in HIGHER_IS_BETTER
    best = select_by_val(usable, metric=a.metric, lower_is_better=not higher)

    vals = np.array([r[a.metric] for r in usable], dtype=float)
    mean, std = float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    band = expected_max_of_noise(mean, std, len(vals)) if higher else None

    print(f"{len(usable)} run(s) with {a.metric}; selecting on validation only.\n")
    print(f"  {'timestamp':22}{a.metric:>10}{'anchor':>16}{'train_end':>22}")
    for r in sorted(usable, key=lambda r: r[a.metric], reverse=higher):
        mark = "  <- selected" if r is best else ""
        print(f"  {str(r.get('timestamp','?')):22}{r[a.metric]:10.4f}"
              f"{str(r.get('anchor','?')):>16}{str(r.get('train_end_ts','?'))[:19]:>22}{mark}")

    print(f"\nselected: {best.get('timestamp')}  ({a.metric}={best[a.metric]:.4f})")
    if band is not None and len(vals) > 1:
        print(f"  mean {mean:.4f}  std {std:.4f}")
        print(f"  best of {len(vals)} from noise alone would be ~{band:.4f}")
        if best[a.metric] <= band:
            print("  ⚠ the winner is INSIDE the noise band — this is not a discovery.")
            print("    Deploying it is fine; quoting its margin as an improvement is not.")
    print("\nPin it in technical_analysis_trainer/assembly/config.yaml:")
    print(f"  defaults:\n    trend_run: \"{best.get('timestamp')}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
