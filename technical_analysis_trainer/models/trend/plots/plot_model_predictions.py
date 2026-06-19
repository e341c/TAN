"""
Test a selected ONNX run and plot predictions interactively.

Examples:
  python -m technical_analysis_trainer.models.trend.plots.plot_model_predictions --run 2026-04-18T20-55-59
  python -m technical_analysis_trainer.models.trend.plots.plot_model_predictions --run 2026-04-18T20-55-59 --save-html artifacts/predictions.html --no-show
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from technical_analysis_trainer.models.trend.test_cli import run_evaluation

_TREND_PKG = Path(__file__).resolve().parent.parent


def main() -> None:
    default_cfg = str(_TREND_PKG / "test_config.yaml")

    ap = argparse.ArgumentParser(description="Plot model predictions for a selected ONNX run.")
    ap.add_argument(
        "--config",
        default=os.getenv("CONFIG_PATH", default_cfg),
        help="YAML as in training (data_path, window, horizon, threshold, …)",
    )
    ap.add_argument(
        "--run",
        default=os.getenv("MODEL_RUN", ""),
        metavar="TIMESTAMP",
        help="Subfolder in output_dir (e.g. 2026-04-18T20-55-59). Empty = latest by mtime.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Markers and binary metrics only when max(softmax) >= this value",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature (argmax unchanged at T=1)",
    )
    ap.add_argument(
        "--save-html",
        default="",
        metavar="PATH",
        help="If set, save chart HTML to this path.",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the chart (use with --save-html).",
    )
    args = ap.parse_args()

    fig = run_evaluation(
        cfg_path=args.config,
        run_id=args.run,
        min_confidence=float(args.min_confidence),
        temperature=float(args.temperature),
    )

    if args.save_html:
        out_path = args.save_html
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(
                os.path.join(_TREND_PKG.parent.parent.parent, out_path)
            )
        fig.write_html(out_path)
        print(f"Saved: {out_path}")

    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
