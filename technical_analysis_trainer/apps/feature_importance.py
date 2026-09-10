"""
Permutation importance for the trend features, measured on validation only.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.feature_importance
    PYTHONPATH=. python -m technical_analysis_trainer.apps.feature_importance --seeds 3

The model carries 58 inputs against roughly 4500 *effective* training labels
(22k rows at an average label uniqueness of ~0.2). That ratio is the argument
for pruning: every input that contributes nothing still costs capacity, and the
fold logs show some of them are actively unstable — xmkt_corr_main_eth_{24,72,168}
shift -0.87 to -1.66 training sd between train and test.

Method, and the two decisions that make it trustworthy:

  * **Validation, never test.** Selecting a feature set is selection, and the
    repo's rule is that selection reads validation only. core.validation.select_by_val
    refuses test metrics outright; this follows the same rule by construction.

  * **Clustered, not per-feature.** The feature set contains obvious collinear
    blocks — three amihud windows, three range_per_vol windows, three
    correlation windows, four taker-imbalance windows. Permuting one member of a
    correlated block understates its importance, because the model reads the
    same information from its neighbours. Group importance permutes the whole
    block at once, which is the number that answers "can this block go?".

Both are reported: single-feature importance for ranking, group importance for
the drop decision.

Permutation shuffles a feature across samples while preserving its shape over
the window, so the marginal distribution is unchanged and only its relationship
to the label is destroyed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.protocol import describe_boundary, trim_to_training_span
from technical_analysis_trainer.core.validation import walk_forward_splits
from technical_analysis_trainer.models.trend.features import (
    attach_cross_market_features,
    attach_mtf_features,
    resolve_feature_scale_clip,
)
from technical_analysis_trainer.models.trend.model import TrendEnsemble, train_trend_lstm
from technical_analysis_trainer.models.trend.train_v2 import (
    build_span,
    proba_up,
    set_seed,
    span_warmup,
)

TREND_CONFIG = "technical_analysis_trainer/models/trend/config.yaml"


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    """
    Collinear blocks, keyed by a name that says what the block measures.

    Grouping is by construction rather than by a correlation threshold: these
    families are the same statistic at several lookbacks, which is exactly the
    case where single-feature permutation lies.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for f in features:
        if f.startswith("amihud_"):
            groups["amihud (illiquidity, 3 windows)"].append(f)
        elif f.startswith("range_per_vol_"):
            groups["range_per_vol (3 windows)"].append(f)
        elif f.startswith("avg_trade_size_"):
            groups["avg_trade_size (3 windows)"].append(f)
        elif f.startswith("vol_z_"):
            groups["vol_z (3 windows)"].append(f)
        elif f.startswith("xmkt_corr_main_eth_"):
            groups["xmkt_corr_main_eth (3 windows)"].append(f)
        elif f.startswith("taker_"):
            groups["taker order flow (6)"].append(f)
        elif f.startswith("mtf_4h_"):
            groups["mtf 4h block"].append(f)
        elif f.startswith("mtf_1d_"):
            groups["mtf 1d block"].append(f)
        elif f.startswith("xmkt_log_ret_") or f.startswith("xmkt_"):
            groups["xmkt returns/spreads"].append(f)
        elif f.startswith("funding_rate"):
            groups["funding"].append(f)
        elif f in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            groups["calendar"].append(f)
        elif f in ("roc_3", "roc_6", "roc_12", "price_roc", "macd_hist_norm"):
            groups["momentum"].append(f)
        else:
            groups[f].append(f)
    return dict(groups)


def permutation_importance(
    ens, X: np.ndarray, y: np.ndarray, features: list[str],
    blocks: dict[str, list[str]], rng: np.random.Generator, repeats: int = 3,
) -> dict[str, float]:
    """AUC lost when each block is shuffled across samples. Higher = more useful."""
    base = float(roc_auc_score(y, proba_up(ens, X)))
    out: dict[str, float] = {}
    index = {f: i for i, f in enumerate(features)}
    for name, cols in blocks.items():
        idx = [index[c] for c in cols if c in index]
        if not idx:
            continue
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            # Whole block permuted together and with the same permutation, so
            # the block's internal structure survives and only its link to the
            # label is broken.
            Xp[:, :, idx] = Xp[perm][:, :, idx]
            drops.append(base - float(roc_auc_score(y, proba_up(ens, Xp))))
        out[name] = float(np.mean(drops))
    out["__baseline_auc__"] = base
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Permutation importance on validation")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="artifacts/feature_importance.json")
    # The pruning decision has to be re-measured on the FULL set, not on the
    # survivors of the previous prune — otherwise it only ranks what already
    # passed, and a feature dropped in error can never come back.
    ap.add_argument("--no-exclude", action="store_true",
                    help="ignore config feature_exclude and rank every available input")
    a = ap.parse_args()

    with open(TREND_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    clip_lo, clip_hi = resolve_feature_scale_clip(raw)
    tb = raw.get("triple_barrier") or {}
    cfg = {
        "raw": raw, "seq_len": raw["window"]["long"], "threshold": raw["threshold"],
        "horizon": raw.get("horizon", 1), "mtf_cfg": raw.get("mtf"),
        "cross_cfg": raw.get("cross_market"), "clip_lo": clip_lo, "clip_hi": clip_hi,
        "target_mode": raw.get("target_mode", "triple_barrier"), "tb_cfg": tb,
        "seed": 42, "anchor": str(raw.get("barrier_anchor", "last_observed")),
        "weight_clip_q": raw.get("weight_clip_quantile"),
        "order_flow": bool(raw.get("order_flow", True)),
        "exclude": [] if a.no_exclude else list(raw.get("feature_exclude") or []),
        "label_autocorr": bool(raw.get("label_autocorr", False)),
    }

    df = load_dataset(raw["data_path"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    print(f"  {describe_boundary(df)}")
    df = trim_to_training_span(df, "trend train data")
    if (cfg["cross_cfg"] or {}).get("enabled", False):
        df = attach_cross_market_features(df, cfg["cross_cfg"])
    if (cfg["mtf_cfg"] or {}).get("enabled", True):
        df = attach_mtf_features(df, cfg["mtf_cfg"])

    embargo = int(tb.get("max_holding", 24)) + cfg["seq_len"]
    splits = walk_forward_splits(len(df), n_folds=int(raw.get("n_folds", 3)),
                                 val_ratio=raw.get("val_ratio", 0.15),
                                 test_ratio=raw.get("test_ratio", 0.15), embargo=embargo)
    warmup = span_warmup(cfg)
    rng = np.random.default_rng(0)
    per_fold: list[dict[str, float]] = []

    for i, sp in enumerate(splits):
        Xtr, ytr, wtr, feats, scaler = build_span(df, sp.train, warmup, None, cfg)
        Xva, yva, _, _, _ = build_span(df, sp.val, warmup, scaler, cfg)
        if len(Xtr) == 0 or len(Xva) == 0:
            continue
        fold_cfg = dict(raw)
        fold_cfg["stop_after_epoch"] = a.epochs
        if fold_cfg.get("class_weight") == "auto":
            b = np.bincount(ytr, minlength=2)
            fold_cfg["class_weight"] = [len(ytr) / (2 * max(b[0], 1)), len(ytr) / (2 * max(b[1], 1))]
        members = []
        for s in range(a.seeds):
            set_seed(cfg["seed"] + 500 * i + s)
            m, _ = train_trend_lstm(X=Xtr, y=ytr, cfg=fold_cfg, w=wtr)
            members.append(m)
        ens = TrendEnsemble(members)

        blocks = feature_groups(feats)
        imp = permutation_importance(ens, Xva, yva, feats, blocks, rng, a.repeats)
        per_fold.append(imp)
        print(f"  fold {i}: val AUC {imp['__baseline_auc__']:.4f}  "
              f"({len(feats)} features in {len(blocks)} blocks)")

    if not per_fold:
        print("no usable folds")
        return 1

    names = [k for k in per_fold[0] if not k.startswith("__")]
    mean = {n: float(np.mean([f.get(n, np.nan) for f in per_fold])) for n in names}
    spread = {n: float(np.std([f.get(n, np.nan) for f in per_fold], ddof=1)) if len(per_fold) > 1 else 0.0
              for n in names}

    print(f"\n{'='*76}\n── Block importance (val AUC lost when shuffled, mean of {len(per_fold)} folds) ──")
    print(f"  {'block':38}{'mean':>9}{'std':>9}{'per-fold':>22}")
    for n in sorted(names, key=lambda k: -mean[k]):
        pf = ", ".join(f"{f.get(n, float('nan')):+.3f}" for f in per_fold)
        print(f"  {n:38}{mean[n]:+9.4f}{spread[n]:9.4f}   {pf}")

    # Seed-to-seed AUC noise on this data is ~0.005 (README). A block whose mean
    # importance sits inside that band is not measurably doing anything, and a
    # negative mean means the model did better without it on average.
    NOISE = 0.005
    droppable = [n for n in names if mean[n] < NOISE]
    print(f"\n  blocks with mean importance below the {NOISE} seed-noise floor: {len(droppable)}")
    for n in sorted(droppable, key=lambda k: mean[k]):
        print(f"    {n:38}{mean[n]:+.4f}")
    print("\n  These are candidates, not a verdict: importance is measured on")
    print("  validation with a 3-epoch budget, and a block that is useless to this")
    print("  model may still matter to a different one. Confirm any drop with")
    print("  apps/rolling_eval.py before believing it.")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_fold": per_fold, "mean": mean, "std": spread}, f, indent=2)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
