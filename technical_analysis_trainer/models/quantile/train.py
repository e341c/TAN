import datetime
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import yaml

from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.export_onnx import export_model
from technical_analysis_trainer.core.validation import (
    Split,
    walk_forward_splits,
    warn_if_stale,
)
from technical_analysis_trainer.models.quantile.eval import (
    evaluate_model,
    format_report,
    predict_z,
)
from technical_analysis_trainer.models.quantile.features import (
    DEFAULT_VOL_WINDOW,
    add_features,
)
from technical_analysis_trainer.models.quantile.model import (
    QuantileEnsemble,
    train_quantile_lstm,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slice_with_warmup(df: pd.DataFrame, start: int, end: int, warmup: int):
    """Take rows [start, end) plus the warm-up bars the indicators need."""
    lo = max(0, start - warmup)
    return df.iloc[lo:end].reset_index(drop=True), start - lo


def _build_split_arrays(df, span, warmup, scaler, cfg):
    """Build (X, y, sigma, close, har) for one index span, dropping warm-up rows."""
    start, end = span
    sub, _ = _slice_with_warmup(df, start, end, warmup)
    X, y, sig, close, har, feats, sc = add_features(
        sub,
        seq_len=cfg["seq_len"],
        horizon=cfg["horizon"],
        vol_window=cfg["vol_window"],
        scaler=scaler,
        clip=cfg["clip"],
    )

    # Drop the rows that only existed to warm up the indicators.
    drop = max(0, len(X) - (end - start))
    if drop:
        X, y, sig, close, har = X[drop:], y[drop:], sig[drop:], close[drop:], har[drop:]
    return X, y, sig, close, har, feats, sc


def run_fold(df: pd.DataFrame, split: Split, cfg: dict, fold_id: int) -> dict:
    """Train a seed ensemble on one walk-forward fold and evaluate it on test."""
    warmup = max(cfg["vol_window"], 168) + cfg["seq_len"] + 5
    quantiles = cfg["quantiles"]

    # Scaler is fitted on train only; val/test reuse it.
    Xtr, ytr, str_, ctr, htr, feats, scaler = _build_split_arrays(
        df, split.train, warmup, None, cfg
    )
    Xva, yva, sva, cva, hva, _, _ = _build_split_arrays(
        df, split.val, warmup, scaler, cfg
    )
    Xte, yte, ste, cte, hte, _, _ = _build_split_arrays(
        df, split.test, warmup, scaler, cfg
    )

    print(f"\n{'='*70}\nFold {fold_id}:  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    # ── Seed ensemble ────────────────────────────────────────────────────────
    members, seed_val_losses = [], []
    for s in range(cfg["n_seeds"]):
        set_seed(cfg["seed"] + 1000 * fold_id + s)
        m, tm = train_quantile_lstm(
            X=Xtr, y=ytr, X_val=Xva, y_val=yva, cfg={**cfg, "quantiles": quantiles}
        )
        members.append(m)
        seed_val_losses.append(tm.get("best_val_loss"))
        print(f"  seed {s}: best_val_loss={tm.get('best_val_loss'):.6f} "
              f"epoch={tm.get('best_epoch')}")

    ens = QuantileEnsemble(members)
    ens.eval()

    # ── Split-conformal calibration on val ───────────────────────────────────
    # Shift each predicted quantile by the residual quantile observed on data the
    # ensemble never trained on, so coverage is right out of sample rather than
    # right on the training distribution.
    val_pred = predict_z(ens, Xva)
    offsets = np.array(
        [float(np.quantile(yva - val_pred[:, i], q)) for i, q in enumerate(quantiles)],
        dtype=np.float32,
    )
    calibrated = QuantileEnsemble(members, offsets=offsets)
    calibrated.eval()
    print(f"  conformal offsets (z-space): {np.round(offsets, 4).tolist()}")

    res_raw = evaluate_model(ens, Xte, yte, ste, cte, quantiles, train_z=ytr, har_sigma=hte)
    res = evaluate_model(
        calibrated, Xte, yte, ste, cte, quantiles, train_z=ytr, har_sigma=hte
    )

    print(f"\n── Fold {fold_id} test ──")
    print(format_report(res))
    print(f"  (coverage before conformal: {res_raw['coverage']*100:.2f}%)")

    return {
        "fold": fold_id,
        "model": calibrated,
        "scaler": scaler,
        "features": feats,
        "offsets": offsets,
        "metrics": res,
        "coverage_precalib": res_raw["coverage"],
        "seed_val_losses": seed_val_losses,
        "n_test": len(Xte),
    }


def main() -> None:
    base_dir = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(base_dir, "config.yaml"))
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    window = raw.get("window")
    if window is None:
        raise ValueError("Config must define 'window'")

    cfg = {
        **raw,
        "seq_len": window["long"],
        "horizon": raw.get("horizon", 1),
        "vol_window": raw.get("vol_window", DEFAULT_VOL_WINDOW),
        "quantiles": raw.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95]),
        "clip": float(raw.get("feature_scale_clip", 5.0)),
        "n_seeds": int(raw.get("n_seeds", 5)),
        "seed": int(raw.get("seed", 42)),
    }

    df = load_dataset(raw["data_path"])
    warn_if_stale(df, f"quantile train data ({raw['data_path']})")
    n = len(df)

    # Embargo >= horizon so the last training label cannot peek into val/test.
    splits = walk_forward_splits(
        n,
        n_folds=int(raw.get("n_folds", 3)),
        val_ratio=raw.get("val_ratio", 0.15),
        test_ratio=raw.get("test_ratio", 0.15),
        embargo=cfg["horizon"] + cfg["seq_len"],
    )
    print(f"Dataset {n} bars → {len(splits)} walk-forward folds, "
          f"{cfg['n_seeds']} seeds each, quantiles={cfg['quantiles']}")

    folds = [run_fold(df, sp, cfg, i) for i, sp in enumerate(splits)]

    # ── Aggregate across folds ───────────────────────────────────────────────
    print(f"\n{'='*70}\n── Walk-forward summary ({len(folds)} folds) ──")
    keys = [
        "coverage", "mae_skill_vs_naive", "direction_accuracy",
        "direction_edge_vs_majority", "avg_breach_pct", "avg_interval_width_pct",
    ]
    for k in keys:
        vals = [f["metrics"][k] for f in folds if k in f["metrics"]]
        if vals:
            print(f"  {k:32} {np.mean(vals):+.4f}  (per-fold: "
                  f"{', '.join(f'{v:+.4f}' for v in vals)})")
    for suffix, label in [("_skill", "vs unconditional"), ("_skill_vs_har", "vs HAR-RV")]:
        ks = sorted({k for f in folds for k in f["metrics"] if k.endswith(suffix)})
        if ks:
            print(f"  ── pinball skill {label} ──")
            for k in ks:
                vals = [f["metrics"][k] for f in folds if k in f["metrics"]]
                print(f"    {k.replace(suffix, ''):16} {np.mean(vals):+.4f}")

    # ── Export the most recent fold (trained on the freshest data) ───────────
    best = folds[-1]
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = f"{raw['output_dir']}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": best["scaler"],
                "features": best["features"],
                "seq_len": cfg["seq_len"],
                "horizon": cfg["horizon"],
                "vol_window": cfg["vol_window"],
                "quantiles": cfg["quantiles"],
                "feature_scale_clip": cfg["clip"],
                "conformal_offsets": best["offsets"].tolist(),
                "n_seeds": cfg["n_seeds"],
                "target": "vol_normalized_log_return",
            },
            f,
        )

    export_model(
        model=best["model"],
        features=best["features"],
        eval_results=best["metrics"],
        output_dir=output_dir,
        version=timestamp,
        horizon=cfg["seq_len"],
    )

    print(f"\n✅ {raw['symbol']} quantile ensemble ({cfg['n_seeds']} seeds, "
          f"conformal-calibrated) → {output_dir}")


if __name__ == "__main__":
    main()
