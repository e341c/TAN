"""
Trend training with the evaluation protocol the 184-run log showed was missing.

What changed versus train.py, and why:

  * **Walk-forward folds** instead of one chronological split. One test period is
    one regime; three give a spread.
  * **Seed ensembling** instead of seed lottery. Across those 184 runs ROC-AUC
    was 0.5305 +/- 0.0052 with *zero* correlation to any hyperparameter
    (hidden_size: +0.000), and the best run (0.5457) sits exactly where the max
    of 184 pure-noise draws is expected (0.5439). Averaging seeds captures the
    centre instead of gambling on the tail.
  * **Selection on validation only.** Picking the best run by test score is what
    inflated the logged numbers relative to the ~51.8% seen live.
  * **Triple-barrier target** (optional, via config) and per-sample weighting by
    move size x label uniqueness.
  * **GBM on the last bar reported alongside** — it beat the LSTM by +0.0044
    ROC-AUC (t=7.13) across the log, so it is the baseline to clear, not an
    afterthought.
"""
from __future__ import annotations

import csv
import datetime
import os
import pickle
import random

import numpy as np
import torch
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.export_onnx import export_model
from technical_analysis_trainer.core.metrics import bootstrap_ci
from technical_analysis_trainer.core.validation import (
    Split,
    expected_max_of_noise,
    report_dead_features,
    report_feature_drift,
    walk_forward_splits,
    warn_if_stale,
)
from technical_analysis_trainer.models.trend.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
    resolve_feature_scale_clip,
)
from technical_analysis_trainer.models.trend.model import TrendEnsemble, train_trend_lstm
from technical_analysis_trainer.models.trend.target import build_target_spec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def proba_up(model: torch.nn.Module, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        logits = model(torch.tensor(X[i : i + batch], dtype=torch.float32))
        out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def build_span(df, span, warmup, scaler, cfg):
    start, end = span
    lo = max(0, start - warmup)
    sub = df.iloc[lo:end].reset_index(drop=True)
    X, y, feats, sc, w = add_features(
        sub,
        seq_len=cfg["seq_len"],
        threshold=cfg["threshold"],
        horizon=cfg["horizon"],
        scaler=scaler,
        mtf_cfg=cfg["mtf_cfg"],
        cross_market_cfg=cfg["cross_cfg"],
        clip_low=cfg["clip_lo"],
        clip_high=cfg["clip_hi"],
        target_mode=cfg["target_mode"],
        tb_cfg=cfg["tb_cfg"],
        return_weights=True,
    )
    drop = max(0, len(X) - (end - start))
    if drop:
        X, y, w = X[drop:], y[drop:], w[drop:]
    return X, y, w, feats, sc


def run_fold(df, split: Split, cfg: dict, fold_id: int) -> dict:
    warmup = cfg["seq_len"] + 200
    Xtr, ytr, wtr, feats, scaler = build_span(df, split.train, warmup, None, cfg)
    Xva, yva, wva, _, _ = build_span(df, split.val, warmup, scaler, cfg)
    Xte, yte, wte, _, _ = build_span(df, split.test, warmup, scaler, cfg)

    print(f"\n{'='*70}\nFold {fold_id}: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        print("  ⚠ empty split — skipping fold")
        return {}
    print(f"  class balance train: {np.bincount(ytr, minlength=2)}  "
          f"test: {np.bincount(yte, minlength=2)}")

    # Input health, before a single epoch is spent. Both checks exist because
    # both failures shipped silently once: constant features that inference then
    # feeds unscaled, and non-stationary features that walk the operating point
    # off 0.5 until the model emits one class forever.
    report_dead_features(scaler, feats, label=f"fold {fold_id} train")
    report_feature_drift(Xtr, Xte, feats, label=f"fold {fold_id} test")

    fold_cfg = dict(cfg["raw"])
    if fold_cfg.get("class_weight") == "auto":
        b = np.bincount(ytr, minlength=2)
        n0, n1 = max(b[0], 1), max(b[1], 1)
        fold_cfg["class_weight"] = [len(ytr) / (2 * n0), len(ytr) / (2 * n1)]

    members, val_losses, seed_aucs, kept_epochs = [], [], [], []
    for s in range(cfg["n_seeds"]):
        set_seed(cfg["seed"] + 1000 * fold_id + s)
        m, tm = train_trend_lstm(
            X=Xtr, y=ytr, X_val=Xva, y_val=yva, cfg=fold_cfg, w=wtr, w_val=wva
        )
        members.append(m)
        val_losses.append(tm.get("best_val_loss"))
        kept_epochs.append(int(tm.get("best_epoch") or 0))
        # Per-seed test AUC is recorded for the noise analysis only — it is
        # never used to choose anything.
        seed_aucs.append(float(roc_auc_score(yte, proba_up(m, Xte))))
        # `epoch` is the checkpoint actually kept, out of `epochs_run` trained.
        # A run that keeps epoch 1 of 26 has not trained — it has early-stopped
        # onto its initialisation, which is what a flat LR at full rate produces.
        print(f"  seed {s}: val_auc={tm.get('best_val_auc') or float('nan'):.4f} "
              f"val_loss={tm.get('best_val_loss'):.4f} "
              f"epoch={tm.get('best_epoch')}/{tm.get('epochs_run')} "
              f"(test auc {seed_aucs[-1]:.4f})")

    # A checkpoint kept at epoch 1 is the model's initialisation — the optimiser
    # found nothing validation preferred over random weights — and averaging such
    # a member in dilutes the rest of the ensemble. `min_epochs` in config.yaml
    # should make this unreachable, so reaching it means that setting is off.
    min_epochs = int(cfg["raw"].get("min_epochs", 3))
    untrained = [i for i, e in enumerate(kept_epochs) if e < max(2, min_epochs)]
    if untrained:
        print(
            f"  ⚠ {len(untrained)}/{len(kept_epochs)} seed(s) kept an epoch before "
            f"{max(2, min_epochs)} (seeds {untrained}) — those members are barely "
            f"past initialisation. Check min_epochs / warmup_epochs."
        )
    print(f"  kept epochs: {kept_epochs}  (min_epochs={min_epochs})")

    ens = TrendEnsemble(members)
    p_ens = proba_up(ens, Xte)
    auc_ens = float(roc_auc_score(yte, p_ens))
    acc_ens = float(accuracy_score(yte, (p_ens >= 0.5).astype(int)))

    # Validation AUC. Any choice between configurations — barrier width, feature
    # set, architecture — must be made on this number, never on the test one.
    auc_val = float(roc_auc_score(yva, proba_up(ens, Xva)))

    # GBM on the last bar of the window — the baseline that beat the LSTM.
    gb = GradientBoostingClassifier(n_estimators=cfg["raw"].get("n_estimators", 100))
    gb.fit(Xtr[:, -1, :], ytr, sample_weight=wtr)
    p_gb = gb.predict_proba(Xte[:, -1, :])[:, 1]
    auc_gb = float(roc_auc_score(yte, p_gb))

    # Blend: the two disagree run-to-run (corr 0.08 across the log), so their
    # errors are close to independent and averaging usually helps.
    p_blend = 0.5 * p_ens + 0.5 * p_gb
    auc_blend = float(roc_auc_score(yte, p_blend))

    majority = float(max(np.mean(yte), 1 - np.mean(yte)))
    correct = ((p_ens >= 0.5).astype(int) == yte).astype(float)

    print(f"\n  ── Fold {fold_id} ──")
    print(f"    VAL auc        : {auc_val:.4f}   <- select configurations on this")
    print(f"    seed AUCs      : {[round(a, 4) for a in seed_aucs]}")
    print(f"    LSTM ensemble  : auc {auc_ens:.4f}  acc {acc_ens:.4f} "
          f"(95% CI {bootstrap_ci(correct)[0]:.4f}-{bootstrap_ci(correct)[1]:.4f})")
    print(f"    GBM last-bar   : auc {auc_gb:.4f}")
    print(f"    Blend 50/50    : auc {auc_blend:.4f}")
    print(f"    majority base  : {majority:.4f}   edge {acc_ens - majority:+.4f}")

    # Precision in the top-confidence decile: with a small edge, the only way to
    # use it is to act rarely and only where the model is most sure.
    k = max(1, len(p_ens) // 10)
    top = np.argsort(np.abs(p_ens - 0.5))[-k:]
    prec_top = float(accuracy_score(yte[top], (p_ens[top] >= 0.5).astype(int)))
    print(f"    top-decile acc : {prec_top:.4f}  (n={k})")

    return {
        "fold": fold_id, "model": ens, "scaler": scaler, "features": feats,
        "auc_val": auc_val,
        "auc_ens": auc_ens, "acc_ens": acc_ens, "auc_gb": auc_gb,
        "auc_blend": auc_blend, "seed_aucs": seed_aucs, "majority": majority,
        "top_decile_acc": prec_top, "n_test": len(Xte),
        "val_losses": val_losses,
    }


def main() -> None:
    base_dir = os.path.dirname(__file__)
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(base_dir, "config.yaml"))
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    window = raw["window"]
    clip_lo, clip_hi = resolve_feature_scale_clip(raw)
    tb_cfg = raw.get("triple_barrier") or {}
    cfg = {
        "raw": raw,
        "seq_len": window["long"],
        "threshold": raw["threshold"],
        "horizon": raw.get("horizon", 1),
        "mtf_cfg": raw.get("mtf"),
        "cross_cfg": raw.get("cross_market"),
        "clip_lo": clip_lo, "clip_hi": clip_hi,
        "target_mode": raw.get("target_mode", "next_bar"),
        "tb_cfg": tb_cfg,
        "n_seeds": int(raw.get("n_seeds", 5)),
        "seed": int(raw.get("seed", 42)) if str(raw.get("seed")) != "auto" else 42,
    }

    df = load_dataset(raw["data_path"])
    warn_if_stale(df, f"trend train data ({raw['data_path']})")
    if (cfg["cross_cfg"] or {}).get("enabled", False):
        df = attach_cross_market_features(df, cfg["cross_cfg"])
    if (cfg["mtf_cfg"] or {}).get("enabled", True):
        df = attach_mtf_features(df, cfg["mtf_cfg"])

    lookahead = (
        int(tb_cfg.get("max_holding", 24))
        if cfg["target_mode"] == "triple_barrier"
        else cfg["horizon"]
    )
    splits = walk_forward_splits(
        len(df),
        n_folds=int(raw.get("n_folds", 3)),
        val_ratio=raw.get("val_ratio", 0.15),
        test_ratio=raw.get("test_ratio", 0.15),
        # Labels look `lookahead` bars ahead; without an embargo the last train
        # labels overlap the val period and the split leaks.
        embargo=lookahead + cfg["seq_len"],
    )
    print(f"target_mode={cfg['target_mode']}  folds={len(splits)}  "
          f"seeds={cfg['n_seeds']}  embargo={lookahead + cfg['seq_len']}")

    folds = [f for f in (run_fold(df, sp, cfg, i) for i, sp in enumerate(splits)) if f]
    if not folds:
        raise SystemExit("no usable folds")

    print(f"\n{'='*70}\n── Walk-forward summary ({len(folds)} folds) ──")
    for k, label in [
        ("auc_val", "VAL AUC (for selection)"),
        ("auc_ens", "LSTM ensemble AUC"), ("auc_gb", "GBM AUC"),
        ("auc_blend", "Blend AUC"), ("acc_ens", "Ensemble accuracy"),
        ("majority", "Majority baseline"), ("top_decile_acc", "Top-decile accuracy"),
    ]:
        vals = [f[k] for f in folds]
        print(f"  {label:22} {np.mean(vals):.4f}  "
              f"(per-fold: {', '.join(f'{v:.4f}' for v in vals)})")

    # Single-seed spread vs the ensemble: quantifies how much of the old run log
    # was seed lottery rather than signal.
    all_seed = [a for f in folds for a in f["seed_aucs"]]
    if len(all_seed) > 1:
        m, s = float(np.mean(all_seed)), float(np.std(all_seed))
        print(f"\n  single-seed AUC: mean {m:.4f} std {s:.4f}")
        print(f"  best-of-{len(all_seed)} expected from noise alone: "
              f"{expected_max_of_noise(m, s, len(all_seed)):.4f} "
              f"(observed best {max(all_seed):.4f})")
        print(f"  ensembled AUC:   {np.mean([f['auc_ens'] for f in folds]):.4f}")

    best = folds[-1]
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(raw["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": best["scaler"], "features": best["features"],
                "scale_clip_min": clip_lo, "scale_clip_max": clip_hi,
                "target_mode": cfg["target_mode"], "n_seeds": cfg["n_seeds"],
                # Full target definition travels with the artifact. Reading it
                # from a config at inference lets the two drift: a triple-barrier
                # model served under a next_bar config emits "UP"/"DOWN" that
                # mean a different thing than every consumer assumes.
                "target_spec": build_target_spec(
                    cfg["target_mode"], tb_cfg, horizon=cfg["horizon"]
                ),
            },
            f,
        )
    export_model(
        model=best["model"], features=best["features"],
        eval_results={k: v for k, v in best.items()
                      if isinstance(v, (int, float, str, list))},
        output_dir=output_dir, version=timestamp, horizon=cfg["seq_len"],
    )

    runs_csv = os.path.join(base_dir, "train_runs_v2.csv")
    row = {
        "timestamp": timestamp, "target_mode": cfg["target_mode"],
        "n_folds": len(folds), "n_seeds": cfg["n_seeds"],
        "auc_ens": np.mean([f["auc_ens"] for f in folds]),
        "auc_gb": np.mean([f["auc_gb"] for f in folds]),
        "auc_blend": np.mean([f["auc_blend"] for f in folds]),
        "acc_ens": np.mean([f["acc_ens"] for f in folds]),
        "majority": np.mean([f["majority"] for f in folds]),
        "top_decile_acc": np.mean([f["top_decile_acc"] for f in folds]),
    }
    write_header = not os.path.isfile(runs_csv)
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            wcsv.writeheader()
        wcsv.writerow(row)
    print(f"\n✅ exported → {output_dir}\n📋 logged → {runs_csv}")


if __name__ == "__main__":
    main()
