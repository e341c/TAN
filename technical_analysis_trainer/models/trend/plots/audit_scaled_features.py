"""
Audit feature scaling exactly like train.py: chronological split, scaler fit on train
build_features_df rows, transform + clip (see config `feature_scale_clip`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.models.trend.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
    resolve_feature_scale_clip,
)

_TREND_PKG = Path(__file__).resolve().parent.parent


def _finite_stats(a: np.ndarray) -> dict[str, float]:
    v = a.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0.0}
    av = np.abs(v)
    return {
        "n": float(n),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p01": float(np.percentile(v, 1)),
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
        "frac_abs_ge_3": float(np.mean(av >= 3.0)),
    }


def parse_args() -> argparse.Namespace:
    default_cfg = _TREND_PKG / "config.yaml"
    p = argparse.ArgumentParser(description="Audit StandardScaler + clip vs train.py pipeline.")
    p.add_argument("--config", type=str, default=str(default_cfg), help="YAML (same as train).")
    p.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/scaled_feature_audit.csv",
        help="Wide CSV: one row per feature, row-level + sequence-level stats.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    window = cfg.get("window")
    if window is None or "long" not in window:
        raise SystemExit("config.window.long is required (same as train).")

    df = load_dataset(cfg["data_path"])
    cross_cfg = cfg.get("cross_market")
    if (cross_cfg or {}).get("enabled", False):
        df = attach_cross_market_features(df, cross_cfg)
    mtf_cfg = cfg.get("mtf")
    if (mtf_cfg or {}).get("enabled", True):
        df = attach_mtf_features(df, mtf_cfg)

    n = len(df)
    val_ratio = float(cfg.get("val_ratio", 0.1))
    test_ratio = float(cfg.get("test_ratio", 0.1))
    val_idx = int(n * (1 - val_ratio - test_ratio))
    test_idx = int(n * (1 - test_ratio))

    train_df = df.iloc[:val_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx:test_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx:].reset_index(drop=True)

    seq_len = int(window["long"])
    threshold = float(cfg["threshold"])
    horizon = int(cfg.get("horizon", 3))
    clip_lo, clip_hi = resolve_feature_scale_clip(cfg)
    print(f"Clip (after StandardScaler): [{clip_lo}, {clip_hi}]")

    X_train, y_train, features, scaler, tr_feat = add_features(
        train_df,
        seq_len=seq_len,
        threshold=threshold,
        horizon=horizon,
        scaler=None,
        mtf_cfg=mtf_cfg,
        cross_market_cfg=cross_cfg,
        return_df_feat=True,
        clip_low=clip_lo,
        clip_high=clip_hi,
    )
    X_val, y_val, _, _, va_feat = add_features(
        val_df,
        seq_len=seq_len,
        threshold=threshold,
        horizon=horizon,
        scaler=scaler,
        mtf_cfg=mtf_cfg,
        cross_market_cfg=cross_cfg,
        return_df_feat=True,
        clip_low=clip_lo,
        clip_high=clip_hi,
    )
    X_test, y_test, _, _, te_feat = add_features(
        test_df,
        seq_len=seq_len,
        threshold=threshold,
        horizon=horizon,
        scaler=scaler,
        mtf_cfg=mtf_cfg,
        cross_market_cfg=cross_cfg,
        return_df_feat=True,
        clip_low=clip_lo,
        clip_high=clip_hi,
    )

    splits = {"train": tr_feat, "val": va_feat, "test": te_feat}

    rows: list[dict] = []
    F = len(features)
    print(f"Features: {F}  |  train windows: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    print(f"Scaler fit on train rows: {splits['train'].shape[0]} (after build_features_df dropna)")
    print()

    for j, name in enumerate(features):
        rec: dict = {"feature": name}
        rec["scaler_mean"] = float(scaler.mean_[j])
        rec["scaler_scale"] = float(scaler.scale_[j])

        for split_name, dff in splits.items():
            if name not in dff.columns:
                continue
            raw = dff[name].to_numpy(dtype=np.float64)
            z_pre = scaler.transform(dff[features].values.astype(np.float64))[:, j]
            z_post = np.clip(z_pre, clip_lo, clip_hi)

            rs = _finite_stats(raw)
            zp = _finite_stats(z_pre)
            zo = _finite_stats(z_post)
            z_out = (z_pre < clip_lo) | (z_pre > clip_hi)
            frac_out = float(np.mean(z_out)) if z_pre.size else np.nan
            rec[f"{split_name}_raw_std"] = rs.get("std", np.nan)
            rec[f"{split_name}_raw_p01"] = rs.get("p01", np.nan)
            rec[f"{split_name}_raw_p99"] = rs.get("p99", np.nan)
            rec[f"{split_name}_zpre_std"] = zp.get("std", np.nan)
            rec[f"{split_name}_zpre_frac_ge3"] = zp.get("frac_abs_ge_3", np.nan)
            rec[f"{split_name}_zpre_frac_outside_clip"] = frac_out
            rec[f"{split_name}_zpre_max_abs"] = (
                max(abs(zp.get("min", 0)), abs(zp.get("max", 0))) if zp.get("n") else np.nan
            )
            rec[f"{split_name}_zpost_min"] = zo.get("min", np.nan)
            rec[f"{split_name}_zpost_max"] = zo.get("max", np.nan)

        for split_name, X in (("train", X_train), ("val", X_val), ("test", X_test)):
            ch = X[:, :, j].astype(np.float64).ravel()
            st = _finite_stats(ch)
            rec[f"seq_{split_name}_std"] = st.get("std", np.nan)
            rec[f"seq_{split_name}_p01"] = st.get("p01", np.nan)
            rec[f"seq_{split_name}_p99"] = st.get("p99", np.nan)
            rec[f"seq_{split_name}_frac_ge3"] = st.get("frac_abs_ge_3", np.nan)

        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    key_out_val = "val_zpre_frac_outside_clip"
    key_out_test = "test_zpre_frac_outside_clip"
    if key_out_val in out_df.columns:
        top = out_df.nlargest(8, key_out_val)[["feature", key_out_val, key_out_test]]
        print("\nTop features by val set fraction of z_pre outside clip:")
        print(top.to_string(index=False))

    key_ge3_train = "seq_train_frac_ge3"
    if key_ge3_train in out_df.columns:
        top3 = out_df.nlargest(8, key_ge3_train)[
            ["feature", key_ge3_train, "seq_val_frac_ge3", "seq_test_frac_ge3"]
        ]
        print("\nTop features by fraction |value|>=3 in training windows:")
        print(top3.to_string(index=False))

    print("\nGlobal tensor stats (all features, clipped, sequence batches):")
    for split_name, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        print(
            f"  {split_name}: min={X.min():.4f} max={X.max():.4f} mean={X.mean():.4f} std={X.std():.4f}"
        )


if __name__ == "__main__":
    main()
