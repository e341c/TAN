from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


DEFAULT_METRICS = [
    "roc_auc",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "GBM ROC-AUC",
    "GBM F1",
    "GBM Accuracy",
    "GBM Precision",
    "GBM Recall",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive Plotly charts for trend_model/train_runs.csv"
    )
    p.add_argument(
        "--csv",
        default=str(Path(__file__).with_name("train_runs.csv")),
        help="Path to train_runs.csv (default: trend_model/train_runs.csv)",
    )
    p.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics columns to plot (default: a sensible preset; only existing columns are used).",
    )
    p.add_argument(
        "--mode",
        choices=["profile", "time"],
        default="profile",
        help=(
            "Plot mode. "
            "'profile' = one line per run (timestamp) over metrics (x=metric). "
            "'time' = one line per metric over time (x=timestamp)."
        ),
    )
    p.add_argument(
        "--latest",
        type=int,
        default=0,
        help="If > 0, only plot the latest N runs (by timestamp).",
    )
    p.add_argument(
        "--title",
        default="Train runs",
        help="Chart title",
    )
    p.add_argument(
        "--out",
        default="",
        help="If set, write interactive HTML to this path (e.g. artifacts/train_runs.html).",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="When writing --out HTML, do not open a browser window.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser()
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise SystemExit("Expected a 'timestamp' column in the CSV.")

    # Preserve the original timestamp string exactly as it appears in the CSV
    # so runs are labeled 1:1 with the model folder names.
    df["timestamp_raw"] = df["timestamp"].astype(str)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp")
    if args.latest and args.latest > 0:
        df = df.tail(args.latest)

    metrics = [m for m in args.metrics if m in df.columns]
    if not metrics:
        available = ", ".join(df.columns)
        raise SystemExit(
            "No requested metrics found in CSV. "
            f"Requested={args.metrics}. Available columns: {available}"
        )

    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    long = df.melt(
        id_vars=["timestamp", "timestamp_raw"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["timestamp", "value"])

    if args.mode == "profile":
        # one line per run; x = metric categories
        long["run"] = long["timestamp_raw"]
        # Guard against any accidental duplicate (run, metric) rows which would
        # render as multiple points at the same x on the same line.
        long = long.drop_duplicates(subset=["run", "metric"], keep="last")
        fig = px.line(
            long,
            x="metric",
            y="value",
            color="run",
            line_group="run",
            markers=True,
            category_orders={"metric": metrics},
            title=args.title or "Train runs: metric profile per run",
        )
        fig.update_layout(legend_title_text="run (timestamp)", yaxis_title="score")
    else:
        # one line per metric; x = time
        fig = px.line(
            long,
            x="timestamp",
            y="value",
            color="metric",
            markers=True,
            title=args.title or "Train runs: metrics over time",
        )
        fig.update_layout(legend_title_text="metric", yaxis_title="score")

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), auto_open=not args.no_open)
    else:
        fig.show()


if __name__ == "__main__":
    main()
