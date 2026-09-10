"""
The training/evaluation boundary, enforced rather than documented.

One date, `frozen_eval_start` in assembly/config.yaml, splits the world:

  * rows strictly before it may be trained on;
  * rows from it onward are the frozen evaluation set and no training run may
    read them, ever.

Why this is a module and not a convention. The previous protocol derived
"out-of-sample" two different ways, and both moved underneath you: the backtest
re-derived a walk-forward split whose indices shift every time the CSV grows,
and data_fetch/holdout.py used the artifact's export stamp, which is exact but
only tells you what a model *has already* seen — it cannot stop the next run
from training on the window you meant to evaluate on. A date fixed in config and
checked at the top of training does both: it is stable across refetches, and
`assert_training_data_is_clean` fails the run instead of quietly producing a
number that looks like out-of-sample performance and is not.

The boundary is deliberately not the last row of the CSV. Data keeps arriving;
if the boundary tracked the data, every refetch would silently hand training a
few more days of what used to be the evaluation set.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from technical_analysis_trainer.core.artifacts import resolve_path

ASSEMBLY_CONFIG = "technical_analysis_trainer/assembly/config.yaml"


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def frozen_eval_start(config_path: str | None = None) -> pd.Timestamp | None:
    """
    First timestamp reserved for evaluation, or None when no boundary is set.

    Read fresh on every call rather than cached: training runs and evaluation
    runs are separate processes, and a stale copy of this value is exactly the
    kind of drift the boundary exists to prevent.
    """
    import yaml

    with open(resolve_path(config_path or ASSEMBLY_CONFIG), encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("frozen_eval_start")
    return None if raw in (None, "", False) else _utc(raw)


def trim_to_training_span(df: pd.DataFrame, label: str = "training data") -> pd.DataFrame:
    """
    Drop everything at or after the boundary, loudly.

    Returns the frame unchanged when no boundary is configured, so a repo with
    `frozen_eval_start: null` keeps its old behaviour.
    """
    start = frozen_eval_start()
    if start is None or "ts" not in getattr(df, "columns", []):
        return df
    ts = pd.to_datetime(df["ts"], utc=True)
    keep = ts < start
    dropped = int((~keep).sum())
    if dropped:
        print(
            f"  {label}: withholding {dropped} row(s) from {start:%Y-%m-%d %H:%M} UTC onward "
            f"— frozen evaluation set, not available to training"
        )
    out = df[keep].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"{label}: every row falls on or after frozen_eval_start ({start}). "
            "Either the boundary is wrong or the data is."
        )
    return out


def assert_training_data_is_clean(df: pd.DataFrame, label: str = "training data") -> None:
    """
    Fail if a frame that should be training-only reaches into the frozen set.

    Call this *after* trimming, on the frame that actually reaches the model, so
    a trimming bug cannot pass silently.
    """
    start = frozen_eval_start()
    if start is None or "ts" not in getattr(df, "columns", []):
        return
    last = pd.to_datetime(df["ts"], utc=True).max()
    if last >= start:
        raise ValueError(
            f"{label} reaches {last} but frozen_eval_start is {start}. Training on the "
            "evaluation set would make every number reported from it meaningless."
        )


def describe_boundary(df: pd.DataFrame | None = None) -> str:
    """One line for the top of a training log or an evaluation report."""
    start = frozen_eval_start()
    if start is None:
        return "frozen_eval_start: not set — no training/evaluation boundary is enforced"
    line = f"frozen_eval_start: {start:%Y-%m-%d %H:%M} UTC (rows from here on are evaluation-only)"
    if df is not None and "ts" in getattr(df, "columns", []):
        ts = pd.to_datetime(df["ts"], utc=True)
        line += f"; data holds {int((ts >= start).sum())} evaluation row(s), last {ts.max():%Y-%m-%d %H:%M}"
    return line
