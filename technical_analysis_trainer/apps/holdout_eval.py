"""
Score both deployed models on candles they were never trained on.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.holdout_eval
    PYTHONPATH=. python -m technical_analysis_trainer.apps.holdout_eval --refresh
    PYTHONPATH=. python -m technical_analysis_trainer.apps.holdout_eval --offline --model trend

Every other accuracy number in this repo comes from `data/train/*.csv`, which is
the series the artifacts were fitted on. Even the backtest's "out-of-sample"
block only *estimates* the boundary by re-deriving the walk-forward split, and
that estimate moves every time the CSV grows. This tool removes the guess:
`data_fetch/holdout.py` fetches candles later than the artifacts' export stamps,
and only those anchors are scored. A bar that closed after the ONNX file was
written cannot be in its training set.

Both models are graded against their own target, with the same baselines the
training reports use, so the numbers are directly comparable to `meta.json`:

  * **quantile** — interval coverage against nominal, pinball loss per level
    against the unconditional-quantile baseline, q50 against persistence.
  * **trend** — triple-barrier accuracy and ROC-AUC against the majority class,
    via `apps/backtest.py` restricted to post-cutoff anchors.

A gap between these numbers and `meta.json` is the honest cost of deployment:
`meta.json` reports a held-out *fold* from the same period the model was fitted
in, this reports a later period the model has never seen.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from technical_analysis_trainer.core.artifacts import ArtifactStore
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.metrics import bootstrap_ci, pinball_loss, skill_score
from technical_analysis_trainer.data_fetch.holdout import Holdout, ensure_holdout, model_cutoff
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.quantile.predict import load_bundle, predict_from_candles

# Raw bars handed to predict_from_candles per anchor. Must clear the 168-bar
# HAR-RV window plus seq_len; 400 matches assembly/config.yaml's binance_limit,
# so each anchor is scored on the same amount of history production gives it.
QUANTILE_HISTORY = 400


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ── Quantile ─────────────────────────────────────────────────────────────────


def eval_quantile(h: Holdout, run_id: str | None = None, limit: int | None = None) -> dict:
    """
    Replay the served quantile path over every unseen anchor.

    One `predict_from_candles` call per anchor on a trimmed frame, rather than a
    batched pass over precomputed features: this is the code the API and the bot
    run, and the point of the exercise is to grade what is actually deployed.
    `drop_last=False` because the frame is already trimmed so its final row *is*
    the closed bar to anchor on — the convention `test_backfill_predictions.py`
    pins.
    """
    df = load_dataset(str(h.btc))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = model_cutoff("quantile", h.runs.get("quantile", ""))
    store = ArtifactStore(package_models_dir("quantile"))
    bundle, run = load_bundle(store, run_id)
    horizon = int(bundle.get("horizon", 1))
    levels = [float(q) for q in (bundle.get("quantiles") or [0.1, 0.5, 0.9])]

    # An anchor is scorable when it has enough history behind it and its horizon
    # bar has already closed ahead of it.
    ts = df["ts"].to_numpy()
    idx = np.arange(QUANTILE_HISTORY, len(df) - horizon)
    idx = idx[pd.to_datetime(ts[idx], utc=True) > cutoff]
    if limit:
        idx = idx[-limit:]
    if len(idx) == 0:
        raise ValueError(f"no unseen quantile anchors after {cutoff}")

    rows = []
    for i in idx:
        window = df.iloc[i - QUANTILE_HISTORY + 1 : i + 1].reset_index(drop=True)
        preds, _ = predict_from_candles(window, store, run_id=run.run_id, drop_last=False)
        rows.append(
            {
                "ts": df["ts"].iloc[i],
                "last_close": preds["last_close"],
                "sigma": preds["sigma"],
                "log_returns": preds["log_returns"],
                # Realised outcome: the close `horizon` bars later.
                "true_close": float(df["close"].iloc[i + horizon]),
            }
        )

    f = pd.DataFrame(rows)
    last_close = f["last_close"].to_numpy()
    true_lr = np.log(f["true_close"].to_numpy() / last_close)
    pred_lr = np.vstack(f["log_returns"].to_numpy())  # (n, n_levels)
    sigma = f["sigma"].to_numpy()

    out: dict = {
        "n": len(f),
        "run_id": run.run_id,
        "levels": levels,
        "first": f["ts"].iloc[0],
        "last": f["ts"].iloc[-1],
        "horizon": horizon,
    }

    # Pinball per level, against the two baselines models/quantile/eval.py uses.
    # A bare pinball number is unreadable — it mostly measures how volatile the
    # window happened to be. Both baselines are fitted on pre-cutoff bars only,
    # so neither sees the period it is being scored on.
    ref_z, har_sigma = _baselines(df, bundle, cutoff)
    base_lr = np.quantile(ref_z, levels)[None, :] * sigma[:, None]
    har_lr = None
    har_at_anchor = har_sigma[idx]
    if np.isfinite(har_at_anchor).all():
        har_lr = np.quantile(ref_z, levels)[None, :] * har_at_anchor[:, None]
    for j, q in enumerate(levels):
        out[f"pinball_q{q}"] = pinball_loss(true_lr, pred_lr[:, j], q)
        b = pinball_loss(true_lr, base_lr[:, j], q)
        out[f"pinball_q{q}_baseline"] = b
        out[f"pinball_q{q}_skill"] = skill_score(out[f"pinball_q{q}"], b)
        if har_lr is not None:
            hb = pinball_loss(true_lr, har_lr[:, j], q)
            out[f"pinball_q{q}_har"] = hb
            out[f"pinball_q{q}_skill_vs_har"] = skill_score(out[f"pinball_q{q}"], hb)
    out["has_har"] = har_lr is not None

    # Interval calibration on the outermost trained levels — the same band
    # models/quantile/eval.py reports, so `nominal` matches meta.json.
    lo, hi = pred_lr[:, 0], pred_lr[:, -1]
    inside = (true_lr >= lo) & (true_lr <= hi)
    out["coverage"] = float(inside.mean())
    out["coverage_nominal"] = levels[-1] - levels[0]
    out["coverage_ci95"] = bootstrap_ci(inside.astype(float))
    out["breach_below_rate"] = float((true_lr < lo).mean())
    out["breach_below_nominal"] = levels[0]
    out["breach_above_rate"] = float((true_lr > hi).mean())
    out["breach_above_nominal"] = 1.0 - levels[-1]
    out["avg_interval_width_pct"] = float(np.mean(hi - lo) * 100)
    out["crossing_rate"] = float(np.mean(np.any(np.diff(pred_lr, axis=1) < 0, axis=1)))

    # Point forecast vs persistence, in price space.
    mid = levels.index(0.5) if 0.5 in levels else len(levels) // 2
    pred_price = last_close * np.exp(pred_lr[:, mid])
    true_price = f["true_close"].to_numpy()
    abs_err = np.abs(pred_price - true_price)
    naive_err = np.abs(last_close - true_price)
    out["mae_q50"] = float(abs_err.mean())
    out["mae_naive"] = float(naive_err.mean())
    out["mae_skill_vs_naive"] = skill_score(out["mae_q50"], out["mae_naive"])
    out["mae_skill_ci95"] = bootstrap_ci(naive_err - abs_err)

    correct = ((true_lr > 0) == (pred_lr[:, mid] > 0)).astype(float)
    out["direction_accuracy"] = float(correct.mean())
    out["direction_accuracy_ci95"] = bootstrap_ci(correct)
    majority = max(float((true_lr > 0).mean()), float((true_lr <= 0).mean()))
    out["direction_majority_baseline"] = majority
    out["direction_edge_vs_majority"] = out["direction_accuracy"] - majority
    return out


def _baselines(df: pd.DataFrame, bundle: dict, cutoff: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference z distribution and HAR-RV sigma for the baselines.

    The runs do not persist the training z sample, so the unconditional
    distribution is re-estimated from the holdout file's own pre-cutoff bars.
    That is information available before the first scored anchor, so it gives the
    baseline no advantage the model does not also have — and it is the same
    quantity `models/quantile/eval.py` calls `train_z`.

    Returns (pre-cutoff z sample, har_sigma aligned to `df`'s rows).
    """
    from technical_analysis_trainer.models.quantile.features import (
        DEFAULT_VOL_WINDOW,
        build_features_df,
    )

    vol_window = int(bundle.get("vol_window", DEFAULT_VOL_WINDOW))
    horizon = int(bundle.get("horizon", 1))
    feat = build_features_df(df.reset_index(drop=True), vol_window=vol_window)
    fwd_lr = np.log(feat["close"].shift(-horizon) / feat["close"])
    z = fwd_lr / feat["sigma"]
    pre = pd.to_datetime(feat["ts"], utc=True) <= cutoff
    ref_z = z[pre].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(ref_z) < 100:
        raise ValueError(
            f"only {len(ref_z)} pre-cutoff bars available for the baseline; "
            "fetch more history (data_fetch/holdout.py --months)"
        )
    return ref_z, feat["har_sigma"].to_numpy(dtype=np.float64)


def format_quantile(r: dict) -> str:
    lines = [
        "══ Quantile ══",
        f"  run             {r['run_id']}",
        f"  anchors         {r['n']}  ({r['first']:%Y-%m-%d %H:%M} .. {r['last']:%Y-%m-%d %H:%M} UTC)",
        f"  horizon         {r['horizon']}h   levels {r['levels']}",
        "",
        "── Intervals ──",
        f"  coverage        {_pct(r['coverage'])}  (nominal {_pct(r['coverage_nominal'])}, "
        f"95% CI {_pct(r['coverage_ci95'][0])}–{_pct(r['coverage_ci95'][1])})",
        f"  breach below    {_pct(r['breach_below_rate'])}  (nominal {_pct(r['breach_below_nominal'])})",
        f"  breach above    {_pct(r['breach_above_rate'])}  (nominal {_pct(r['breach_above_nominal'])})",
        f"  avg width       {r['avg_interval_width_pct']:.3f}% of price",
        f"  crossings       {_pct(r['crossing_rate'])}",
        "",
        "── Pinball by level ──",
        "  baselines fitted on pre-cutoff bars only: uncond = unconditional z quantiles,",
        "  HAR = same shape scaled by a HAR-RV volatility forecast instead of the network",
        "",
        f"  {'level':<8}{'model':>12}{'uncond':>12}{'skill':>9}{'HAR':>12}{'skill':>9}",
    ]
    for q in r["levels"]:
        row = f"  {q:<8}{r[f'pinball_q{q}']:12.6f}{r[f'pinball_q{q}_baseline']:12.6f}" \
              f"{r[f'pinball_q{q}_skill']:+9.4f}"
        if r.get("has_har"):
            row += f"{r[f'pinball_q{q}_har']:12.6f}{r[f'pinball_q{q}_skill_vs_har']:+9.4f}"
        else:
            row += f"{'—':>12}{'—':>9}"
        lines.append(row)
    lines += [
        "",
        "── Point forecast (q50) ──",
        f"  MAE             {r['mae_q50']:.2f}   vs naive {r['mae_naive']:.2f}",
        f"  skill vs naive  {r['mae_skill_vs_naive']:+.4f}  "
        f"(95% CI on per-sample gain: {r['mae_skill_ci95'][0]:+.2f}–{r['mae_skill_ci95'][1]:+.2f})",
        "",
        "── Direction ──",
        f"  accuracy        {_pct(r['direction_accuracy'])}  "
        f"(95% CI {_pct(r['direction_accuracy_ci95'][0])}–{_pct(r['direction_accuracy_ci95'][1])})",
        f"  majority base   {_pct(r['direction_majority_baseline'])}",
        f"  edge            {r['direction_edge_vs_majority'] * 100:+.2f} pp",
    ]
    return "\n".join(lines)


# ── Trend ────────────────────────────────────────────────────────────────────


def model_window_start(h: Holdout, model: str) -> pd.Timestamp:
    """
    First unseen bar for ONE model, rather than the joint cutoff.

    `h.cutoff` is the max across served models so their reports cover the same
    window. That is right for a side-by-side, and wrong when one model has been
    retrained under the frozen boundary and another has not: the retrained model
    would be graded on a window shortened by its stale neighbour. Per-model, the
    trend model gets everything after its own training span.
    """
    return model_cutoff(model, h.runs.get(model, ""))


def trend_backtest_config(h: Holdout, stride: int = 1, start: pd.Timestamp | None = None) -> dict:
    """
    A backtest config pinned to the holdout CSVs and to post-cutoff anchors.

    `range.start` is the cutoff, so no anchor the model trained on is scored. The
    benchmark and macro paths move with it — leaving them on the training CSVs
    would feed the model a live BTC series alongside a SOL series that stopped
    months ago, which is exactly the forward-fill trap the report warns about.
    """
    from technical_analysis_trainer.apps.backtest import load_config

    cfg = load_config()
    data = cfg.setdefault("data", {})
    data["btc"] = h.rel("btc")
    for key in ("eth", "sol", "dxy", "risk_proxy"):
        rel = h.rel(key)
        if rel:
            data[key] = rel
    begin = (start if start is not None else h.cutoff)
    cfg["range"] = {"start": begin.isoformat(), "end": None, "last_bars": None, "stride": stride}
    cfg["unseen_from"] = begin.isoformat()
    return cfg


def eval_trend(h: Holdout, run_id: str | None = None, stride: int = 1):
    from technical_analysis_trainer.apps.backtest import run_backtest

    start = model_window_start(h, "trend")
    return run_backtest(trend_backtest_config(h, stride, start=start), run_id=run_id)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the deployed models on candles postdating their export stamps"
    )
    p.add_argument("--model", choices=["quantile", "trend", "both"], default="both")
    p.add_argument("--refresh", action="store_true", help="refetch the holdout candles first")
    p.add_argument("--offline", action="store_true", help="use the cache as-is, never fetch")
    p.add_argument("--max-age-hours", type=float, default=12.0)
    p.add_argument("--stride", type=int, default=1, help="thin the trend anchors")
    p.add_argument("--limit", type=int, default=None, help="score at most N quantile anchors")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    try:
        h = ensure_holdout(refresh=a.refresh, offline=a.offline, max_age_hours=a.max_age_hours)
    except FileNotFoundError as e:
        print(f"holdout ERROR: {e}")
        return 1

    lo, hi = h.span()
    print(
        f"\nUnseen window: {h.cutoff:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} UTC "
        f"({h.unseen_bars()} bars); {lo:%Y-%m-%d} onwards loaded for feature warm-up.\n"
    )

    failed = False
    if a.model in ("quantile", "both"):
        try:
            print(format_quantile(eval_quantile(h, limit=a.limit)))
        except (FileNotFoundError, ValueError) as e:
            print(f"══ Quantile ══\n  ERROR: {e}")
            failed = True
        print()

    if a.model in ("trend", "both"):
        from technical_analysis_trainer.apps.backtest import format_report

        try:
            print("══ Trend ══")
            print(format_report(eval_trend(h, stride=a.stride)))
        except (FileNotFoundError, ValueError) as e:
            print(f"  ERROR: {e}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
