"""
Build the same feature matrix as training (cross_market → MTF → build_features_df)
and write an interactive HTML: either all features on one axis (overlay) or one subplot per feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.models.trend.features import (
    attach_cross_market_features,
    attach_mtf_features,
    build_features_df,
)

_TREND_PKG = Path(__file__).resolve().parent.parent


def _y_for_plot(values: np.ndarray, y_scale: str) -> np.ndarray:
    v = values.astype(float)
    if y_scale != "zscore":
        return v
    mu = np.nanmean(v)
    sigma = float(np.nanstd(v))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sigma


def parse_args() -> argparse.Namespace:
    default_cfg = _TREND_PKG / "config.yaml"
    p = argparse.ArgumentParser(
        description="Plot all columns from build_features_df (model inputs) vs time."
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help="YAML config (data_path, mtf, cross_market), same as train.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="artifacts/features_timeseries.html",
        help="Output HTML path (default: artifacts/features_timeseries.html).",
    )
    p.add_argument(
        "--tail",
        type=int,
        default=0,
        help="If > 0, keep only the last N rows after feature build (smaller HTML).",
    )
    p.add_argument(
        "--layout",
        choices=["overlay", "subplots"],
        default="overlay",
        help=(
            "overlay = one chart, all features as lines on the same axes (default). "
            "subplots = separate horizontal strip per feature."
        ),
    )
    p.add_argument(
        "--y-scale",
        dest="y_scale",
        choices=["zscore", "raw"],
        default="zscore",
        help="For overlay: zscore per feature or raw units.",
    )
    p.add_argument(
        "--row-height",
        type=int,
        default=90,
        help="Pixel height per subplot row when --layout subplots.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title (default: derived from config symbol / data path).",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help="After writing HTML, do not open a browser window.",
    )
    return p.parse_args()


def load_features_frame(cfg: dict):
    df = load_dataset(cfg["data_path"])
    cross_cfg = cfg.get("cross_market")
    if (cross_cfg or {}).get("enabled", False):
        df = attach_cross_market_features(df, cross_cfg)
    mtf_cfg = cfg.get("mtf")
    if (mtf_cfg or {}).get("enabled", True):
        df = attach_mtf_features(df, mtf_cfg)
    return build_features_df(df, mtf_cfg=mtf_cfg, cross_market_cfg=cross_cfg)


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    df, features = load_features_frame(cfg)
    if not features:
        raise SystemExit("No model features returned by build_features_df.")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"DataFrame missing feature columns: {missing}")

    if args.tail and args.tail > 0:
        df = df.iloc[-int(args.tail) :].copy()

    if "ts" not in df.columns:
        raise SystemExit("Expected 'ts' column after build_features_df.")

    x = df["ts"]
    n = len(features)

    title = args.title.strip()
    if not title:
        sym = str(cfg.get("symbol", "")).strip()
        base = f"{sym} — model features" if sym else "Model features (build_features_df)"
        if args.layout == "overlay":
            ys_note = " (z-score per feature)" if args.y_scale == "zscore" else " (raw units)"
            title = base + ys_note
        else:
            title = base

    if args.layout == "overlay":
        fig = go.Figure()
        for col in features:
            y_raw = df[col].to_numpy(dtype=float)
            y_plot = _y_for_plot(y_raw, args.y_scale)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_plot,
                    mode="lines",
                    name=col,
                    line=dict(width=1),
                    opacity=0.85,
                )
            )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=720,
            hovermode="x unified",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=9),
            ),
            margin=dict(r=180),
        )
        fig.update_xaxes(title_text="ts")
        y_title = "z-score (per feature, this window)" if args.y_scale == "zscore" else "value"
        fig.update_yaxes(title_text=y_title)
    else:
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=max(0.002, min(0.02, 0.8 / max(n, 1))),
            subplot_titles=list(features),
        )
        for i, col in enumerate(features, start=1):
            y = df[col].astype(float)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=col,
                    showlegend=False,
                    line=dict(width=1),
                ),
                row=i,
                col=1,
            )
        height = max(800, n * int(args.row_height))
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=height,
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="ts", row=n, col=1)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), auto_open=not args.no_open)
    print(f"Wrote {out_path} ({n} features, {len(df)} rows, layout={args.layout})")


if __name__ == "__main__":
    main()
