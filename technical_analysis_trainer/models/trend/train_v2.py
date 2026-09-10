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
import pandas as pd
import torch
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from technical_analysis_trainer.core.calibration import (
    Calibrator,
    best_threshold,
    brier,
    fit_platt,
    reliability_table,
)
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.export_onnx import export_model
from technical_analysis_trainer.core.metrics import block_bootstrap_ci
from technical_analysis_trainer.core.protocol import (
    assert_training_data_is_clean,
    describe_boundary,
    frozen_eval_start,
    trim_to_training_span,
)
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
from technical_analysis_trainer.models.trend.target import (
    ANCHOR_LAST_OBSERVED,
    build_target_spec,
)


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


# Which evaluation apparatus produced a run's metrics.
#
# Bump this whenever a change makes new val/test numbers incomparable with old
# ones. Runs logged before 2026-09-08 were measured with a `build_span` that
# trimmed by row count instead of by bar, so their validation spans began ~200
# bars inside their own training data and their `auc_val` is inflated by an
# unknown amount. select_run.py compares runs on auc_val; without this column it
# would compare across that break and silently prefer the inflated ones — which
# it did, picking a pre-fix run over the first correctly-measured one.
#
#   1  count-based span trim, warmup seq_len+200  (up to and including 2026-09-07)
#   2  exact bar-indexed trim, warmup seq_len+400 (models/trend/test_spans.py)
HARNESS_VERSION = 2


def span_warmup(cfg) -> int:
    """
    Bars of lead-in a span needs before its first row is usable.

    This was `seq_len + 200` at five call sites. 200 is not enough: the longest
    rolling features run 168 bars, the MTF block resamples to 1d and 4h on top of
    that, and build_features_df drops every row any of them is still NaN on. With
    200 the first usable anchor of a monthly span landed 186 bars after the span
    started, so a quarter of each evaluation window silently produced no rows.
    400 saturates — measured by sweeping the value and watching where the first
    anchor stops moving — and the extra lead-in costs only feature computation.
    """
    return int(cfg["seq_len"]) + 400


def build_span(df, span, warmup, scaler, cfg, return_index: bool = False):
    """
    Build windows for the bars in `[start, end)`, with `warmup` bars of lead-in.

    The lead-in exists so the rolling features and the sequence window are warm
    at `start`; those extra bars must not end up in the span itself. That trim
    used to be done by count — drop `len(X) - (end - start)` rows from the left —
    which silently assumed `add_features` returns one row per bar. It does not:
    it skips unresolved labels, labels that resolve past the frame, and windows
    that still contain warmup NaNs. On a one-month span with this feature set
    that is ~51% of candidate bars, so `len(X)` came out BELOW the bar count, the
    trim computed zero, and every val/test span silently began ~200 bars early —
    inside the preceding split. Evaluation spans overlapped the data they were
    supposed to be held out from, and the per-bar prior in rolling_eval was lined
    up against the wrong bars.

    Now `add_features` reports the anchor bar of every row it kept, and the trim
    is an exact mask on those positions.
    """
    start, end = span
    lo = max(0, start - warmup)
    sub = df.iloc[lo:end].reset_index(drop=True)
    X, y, feats, sc, w, idx = add_features(
        sub,
        return_index=True,
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
        anchor=cfg["anchor"],
        order_flow=cfg["order_flow"],
        exclude=cfg["exclude"],
        label_autocorr=cfg["label_autocorr"],
        # "direction" (which barrier) or "resolution" (whether any was touched).
        # Defaulted rather than required so every existing caller is unchanged.
        label_mode=cfg.get("label_mode", "direction"),
    )
    bars = lo + idx                      # anchor position in `df`, not in `sub`
    keep = bars >= start                 # `end` is already the frame's right edge
    X, y, w, bars = X[keep], y[keep], w[keep], bars[keep]
    w = clip_weights(w, cfg.get("weight_clip_q"))
    if return_index:
        return X, y, w, feats, sc, bars
    return X, y, w, feats, sc


def clip_weights(w: np.ndarray, q: float | None) -> np.ndarray:
    """
    Winsorise per-sample weights at quantile `q`, renormalised to mean 1.

    raw_w = |realised move| x uniqueness is heavy-tailed: on fold 2 the top 10%
    of samples carried 39.1% of the total weight (median 0.625, max 23.06), so
    the effective training sample was far smaller than the row count suggested.
    `None` keeps the historical behaviour, which is the default until a sweep
    says otherwise — this is a knob, not a fix that has been demonstrated here.
    """
    if q is None or not len(w):
        return w
    cap = float(np.quantile(w, float(q)))
    out = np.minimum(w, cap)
    m = float(out.mean())
    return (out / m).astype(np.float32) if m > 0 else w


def run_fold(df, split: Split, cfg: dict, fold_id: int) -> dict:
    warmup = span_warmup(cfg)
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
    # Block bootstrap: triple-barrier labels overlap (median span 6 bars against
    # max_holding 24), so an i.i.d. resample reports an interval roughly
    # sqrt(1/uniqueness) too narrow. Blocks of max_holding preserve the overlap.
    block = int((cfg["tb_cfg"] or {}).get("max_holding", 24)) if cfg["target_mode"] == "triple_barrier" else max(1, cfg["horizon"])
    acc_lo, acc_hi = block_bootstrap_ci(correct, block=block)

    print(f"\n  ── Fold {fold_id} ──")
    print(f"    VAL auc        : {auc_val:.4f}   <- select configurations on this")
    print(f"    seed AUCs      : {[round(a, 4) for a in seed_aucs]}")
    print(f"    LSTM ensemble  : auc {auc_ens:.4f}  acc {acc_ens:.4f} "
          f"(95% block CI {acc_lo:.4f}-{acc_hi:.4f}, blocks of {block})")
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
        "acc_ci95": [acc_lo, acc_hi], "block": block,
        "val_losses": val_losses,
        # Kept epochs feed the final refit's stopping budget; see train_final().
        "kept_epochs": kept_epochs,
        "val_loss": float(np.mean([v for v in val_losses if v is not None])) if val_losses else None,
    }


def train_final(df, cfg: dict, embargo: int, folds: list[dict]) -> dict:
    """
    Fit the artifact that actually ships, on everything training is allowed.

    The walk-forward folds exist to ESTIMATE performance and to choose a
    configuration. Exporting one of them — `folds[-1]`, as this file used to —
    ships a model fitted on whatever that fold's train span happened to be. For
    the 2026-08-17 run that span ended 2025-05-02, so production served weights
    and a scaler that had never seen the previous sixteen months, and its
    StandardScaler was centred on a market a year and a half stale.

    Layout, chronological:

        [0 .. C)          train
        [C+e .. n)        calibration — Platt scaling and the decision threshold

    There is deliberately no validation block. Recency of the weights was the
    largest single defect in the previous artifact, so every bar that is not
    needed for calibration goes into training, and the stopping point comes from
    the folds instead of from a held-out split: `stop_after_epoch` is set to the
    median epoch the folds' own early stopping kept. `epochs` still sets the
    cosine horizon, so the learning-rate path is the one the folds ran.

    Trade-off, stated because it is a real one: a fold-derived epoch budget is a
    weaker stopping rule than early stopping on this model's own validation set.
    It is bought with roughly five months of extra training data, which on the
    evidence in plan.txt is the more expensive of the two to give up.

    The honest caveat the caller must report: this model has no test set of its
    own. Its expected performance is the walk-forward estimate, not anything
    measured here.
    """
    n = len(df)
    # ~8% for calibration: enough samples for a two-parameter fit, recent enough
    # that the base rate it corrects toward is the one being served into.
    c = int(n * 0.92)
    warmup = span_warmup(cfg)

    Xtr, ytr, wtr, feats, scaler = build_span(df, (0, c), warmup, None, cfg)
    Xcal, ycal, wcal, _, _ = build_span(df, (c + embargo, n), warmup, scaler, cfg)

    kept = [e for f in folds for e in f.get("kept_epochs", []) if e]
    n_ep = int(np.median(kept)) if kept else int(cfg["raw"].get("epochs", 60))

    print(f"\n{'='*70}\n── Final refit (the artifact that ships) ──")
    print(f"  train {len(Xtr)}  calibration {len(Xcal)}")
    print(f"  train span ends {df['ts'].iloc[c]:%Y-%m-%d}, "
          f"calibration runs {df['ts'].iloc[c + embargo]:%Y-%m-%d} .. {df['ts'].iloc[n - 1]:%Y-%m-%d}")
    print(f"  epoch budget {n_ep} (median of fold kept epochs {kept})")
    if len(Xtr) == 0 or len(Xcal) == 0:
        raise SystemExit("final refit: an empty span — check frozen_eval_start and the data range")
    print(f"  class balance train {np.bincount(ytr, minlength=2)}  "
          f"cal {np.bincount(ycal, minlength=2)}")
    report_dead_features(scaler, feats, label="final train")

    fold_cfg = dict(cfg["raw"])
    fold_cfg["stop_after_epoch"] = n_ep
    if fold_cfg.get("class_weight") == "auto":
        bal = np.bincount(ytr, minlength=2)
        n0, n1 = max(bal[0], 1), max(bal[1], 1)
        fold_cfg["class_weight"] = [len(ytr) / (2 * n0), len(ytr) / (2 * n1)]

    members = []
    for s in range(cfg["n_seeds"]):
        set_seed(cfg["seed"] + 99000 + s)
        m, tm = train_trend_lstm(X=Xtr, y=ytr, cfg=fold_cfg, w=wtr)
        members.append(m)
        print(f"  seed {s}: trained {tm.get('epochs_run')} epoch(s)")
    ens = TrendEnsemble(members)

    # ── Calibration, on a slice no member was fitted on ───────────────────────
    p_cal = proba_up(ens, Xcal)
    cal = fit_platt(p_cal, ycal, sample_weight=wcal)
    p_adj = cal(p_cal)
    thr = best_threshold(p_adj, ycal)
    # `prior_ref` is the base rate this calibration was fitted at; serving
    # re-centres from it onto the rate prevailing then. Off by default since
    # 2026-09-08: the 18-month evidence that enabled it was measured with a
    # misaligned prior (see core.calibration.apply_prior_shift), and the 36-month
    # re-measurement leaves it undemonstrated. Default False so a config without
    # the key does not silently switch it back on.
    prior_ref = float(ycal.mean())
    use_shift = bool(cfg["raw"].get("prior_shift", False))
    cal = Calibrator(a=cal.a, b=cal.b, n=cal.n, threshold=thr,
                     prior_ref=prior_ref, prior_shift=use_shift)

    print(f"\n  ── Calibration ({len(ycal)} held-out samples) ──")
    print(f"    Platt a={cal.a:.4f} b={cal.b:.4f}   decision threshold {thr:.3f}")
    print(f"    prior_ref {prior_ref:.4f}   prior shift {'ON' if use_shift else 'off'} "
          "(serving re-centres onto the trailing resolved base rate)")
    print(f"    mean p(UP): raw {p_cal.mean():.4f} -> calibrated {p_adj.mean():.4f}   "
          f"actual {ycal.mean():.4f}")
    print(f"    Brier:      raw {brier(p_cal, ycal):.4f} -> calibrated {brier(p_adj, ycal):.4f}")
    print(f"    AUC on calibration slice: {roc_auc_score(ycal, p_cal):.4f} "
          "(unchanged by calibration — a monotone map cannot reorder)")
    print(f"    {'mean p':>10}{'actual':>10}{'n':>7}")
    for mp, obs, cnt in reliability_table(p_adj, ycal):
        print(f"    {mp:10.4f}{obs:10.4f}{cnt:7d}")

    return {
        "model": ens, "scaler": scaler, "features": feats, "calibrator": cal,
        "n_train": len(Xtr), "n_cal": len(Xcal), "epoch_budget": n_ep,
        "train_end_ts": str(df["ts"].iloc[c]),
        "cal_start_ts": str(df["ts"].iloc[c + embargo]),
        "data_end_ts": str(df["ts"].iloc[n - 1]),
        "cal_auc": float(roc_auc_score(ycal, p_cal)) if len(np.unique(ycal)) > 1 else float("nan"),
        "cal_brier_raw": brier(p_cal, ycal), "cal_brier": brier(p_adj, ycal),
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
        "anchor": str(raw.get("barrier_anchor", ANCHOR_LAST_OBSERVED)),
        "weight_clip_q": raw.get("weight_clip_quantile"),
        "order_flow": bool(raw.get("order_flow", True)),
        "exclude": list(raw.get("feature_exclude") or []),
        "label_autocorr": bool(raw.get("label_autocorr", False)),
    }

    df = load_dataset(raw["data_path"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    warn_if_stale(df, f"trend train data ({raw['data_path']})")
    # The evaluation set is withheld here, before a single feature is computed,
    # so nothing downstream can reach it by accident.
    print(f"  {describe_boundary(df)}")
    df = trim_to_training_span(df, "trend train data")
    assert_training_data_is_clean(df, "trend train data")
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
    embargo = lookahead + cfg["seq_len"]
    print(f"target_mode={cfg['target_mode']}  anchor={cfg['anchor']}  folds={len(splits)}  "
          f"seeds={cfg['n_seeds']}  embargo={embargo}")

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

    # The artifact is refitted on everything training may see — it is not one of
    # the folds. See train_final() for why.
    final = train_final(df, cfg, embargo, folds)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(raw["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    spec = build_target_spec(
        cfg["target_mode"], tb_cfg, horizon=cfg["horizon"], anchor=cfg["anchor"]
    )
    with open(f"{output_dir}/scaler_{timestamp}.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": final["scaler"], "features": final["features"],
                "scale_clip_min": clip_lo, "scale_clip_max": clip_hi,
                "target_mode": cfg["target_mode"], "n_seeds": cfg["n_seeds"],
                # Full target definition travels with the artifact. Reading it
                # from a config at inference lets the two drift: a triple-barrier
                # model served under a next_bar config emits "UP"/"DOWN" that
                # mean a different thing than every consumer assumes.
                "target_spec": spec,
                # Probability correction and decision cut, both fitted on a slice
                # no member trained on. Carried here so serving cannot forget to
                # apply them or apply a stale pair.
                "calibrator": final["calibrator"].to_dict(),
            },
            f,
        )

    fold_summary = {
        "n_folds": len(folds),
        "auc_val_mean": float(np.mean([f["auc_val"] for f in folds])),
        "auc_ens_mean": float(np.mean([f["auc_ens"] for f in folds])),
        "acc_ens_mean": float(np.mean([f["acc_ens"] for f in folds])),
        "majority_mean": float(np.mean([f["majority"] for f in folds])),
        "per_fold_auc_val": [f["auc_val"] for f in folds],
        "per_fold_auc_ens": [f["auc_ens"] for f in folds],
        "per_fold_acc_ci95": [f["acc_ci95"] for f in folds],
        # Said explicitly because the number people quote from meta.json used to
        # be a single fold's test score for a model trained on that fold's span.
        "note": (
            "The shipped model is a refit on the full permitted span and has no test "
            "set of its own. These walk-forward numbers are its performance estimate."
        ),
    }
    export_model(
        model=final["model"], features=final["features"],
        eval_results=fold_summary,
        output_dir=output_dir, version=timestamp, horizon=cfg["seq_len"],
        provenance={
            "trained_on": raw["data_path"],
            "frozen_eval_start": str(frozen_eval_start()),
            "train_end_ts": final["train_end_ts"],
            "calibration_start_ts": final["cal_start_ts"],
            "data_end_ts": final["data_end_ts"],
            "n_train": final["n_train"], "n_cal": final["n_cal"],
            "epoch_budget": final["epoch_budget"],
            "embargo": embargo,
            "target_spec": spec,
            "calibrator": final["calibrator"].to_dict(),
            "cal_auc": final["cal_auc"],
            "cal_brier_raw": final["cal_brier_raw"], "cal_brier": final["cal_brier"],
        },
    )

    runs_csv = os.path.join(base_dir, "train_runs_v2.csv")
    row = {
        "timestamp": timestamp, "target_mode": cfg["target_mode"],
        "anchor": cfg["anchor"],
        "n_folds": len(folds), "n_seeds": cfg["n_seeds"],
        # Validation metrics first, and recorded at all: the log used to hold
        # only test-side aggregates, so nobody could select a run correctly even
        # if they wanted to — and the deployed run turned out to be the argmax of
        # 12 test scores, sitting exactly at the noise-max expectation.
        "auc_val": np.mean([f["auc_val"] for f in folds]),
        "val_loss": np.mean([f["val_loss"] for f in folds if f["val_loss"] is not None]),
        "auc_ens": np.mean([f["auc_ens"] for f in folds]),
        "auc_gb": np.mean([f["auc_gb"] for f in folds]),
        "auc_blend": np.mean([f["auc_blend"] for f in folds]),
        "acc_ens": np.mean([f["acc_ens"] for f in folds]),
        "majority": np.mean([f["majority"] for f in folds]),
        "top_decile_acc": np.mean([f["top_decile_acc"] for f in folds]),
        "train_end_ts": final["train_end_ts"],
        "frozen_eval_start": str(frozen_eval_start()),
        "harness": HARNESS_VERSION,
    }
    write_header = not os.path.isfile(runs_csv)
    if not write_header:
        # The schema grew (auc_val, anchor, provenance). Rewriting the old rows
        # with blanks beats silently dropping the new columns on append.
        with open(runs_csv, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        if old and set(old[0]) != set(row):
            with open(runs_csv, "w", newline="", encoding="utf-8") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(row))
                wcsv.writeheader()
                for r in old:
                    wcsv.writerow({k: r.get(k, "") for k in row})
    with open(runs_csv, "a", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            wcsv.writeheader()
        wcsv.writerow(row)
    print(f"\n✅ exported → {output_dir}\n📋 logged → {runs_csv}")
    print(f"   train ends {final['train_end_ts']}, frozen eval from {frozen_eval_start()}")


if __name__ == "__main__":
    main()
