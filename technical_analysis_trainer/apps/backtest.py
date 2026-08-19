"""
Replay the deployed trend model over historical candles and draw what it predicted.

    PYTHONPATH=. python -m technical_analysis_trainer.apps.backtest
    PYTHONPATH=. python -m technical_analysis_trainer.apps.backtest --start 2026-01-01
    PYTHONPATH=. python -m technical_analysis_trainer.apps.backtest --last-bars 2000 --no-open

Prints a metrics report to the console and writes an interactive Plotly chart in
which every prediction is a line running from its anchor candle to the barrier the
model predicted, 24 bars later.

Why this exists rather than `models/trend/test_cli.py`: that tool grades **next-bar
sign**, and the served model is a **triple-barrier** model ("does price touch
+2.5*sigma before -2.5*sigma within 24 bars"). Scoring it as next-bar sign measures
something the model was never trained to do. Ground truth here comes from
`features.triple_barrier_labels`, the same function that generated the training
labels, so the backtest grades the model against its own target.

Three details that decide whether a backtest like this tells the truth:

  * **Anchor convention.** `add_features` pairs window `X[i-seq_len:i]` with
    `labels[i]`: the model sees bars up to `i-1` and the barrier originates at
    `close[i]`. `predict.py` indexes identically. This module follows that, so a
    line starts on the bar *after* the last bar the model saw — which is exactly
    where the barrier levels are anchored.

  * **One feature pass, not one per anchor.** Every rolling window here is
    backward-looking, so computing features once over the whole series gives the
    same values as rebuilding them per anchor, for a fraction of the cost. The one
    exception is MACD (EWM, never fully forgets), which differs marginally in the
    first few hundred bars; the feature warm-up discards those anyway.
    `test_backtest.py` pins this against the live path rather than asserting it.

  * **Unresolved anchors.** When neither barrier is touched within `max_holding`,
    `triple_barrier_labels` returns -1 and training drops the sample. Accuracy here
    is computed on resolved anchors only, and the excluded count is reported rather
    than quietly folded away.

Known data gap, surfaced in the report rather than left silent: the SOL benchmark
CSV ends 2026-04-17, so `xmkt_log_ret_sol` flattens to a constant for anchors after
that date. Same class of failure that killed the ETH cross-market features.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from technical_analysis_trainer.assembly.orchestrator import _run_id_for, load_assembly_config
from technical_analysis_trainer.core.artifacts import ArtifactStore, resolve_path
from technical_analysis_trainer.core.dataset import load_dataset
from technical_analysis_trainer.core.metrics import bootstrap_ci
from technical_analysis_trainer.core.validation import dead_features
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.trend.features import (
    add_features,
    attach_cross_market_features,
    attach_mtf_features,
    triple_barrier_labels,
)
from technical_analysis_trainer.models.trend.predict import (
    backfill_missing_features_with_scaler_means,
    load_target_spec,
)
from technical_analysis_trainer.models.trend.target import horizon_label

DEFAULT_CONFIG = Path(__file__).resolve().parent / "backtest_config.yaml"
# ONNX batch size. The exported graph has a dynamic batch axis (see
# core/export_onnx.py), so the whole range goes through in a handful of calls
# instead of test_cli.py's one session.run per bar.
BATCH = 4096


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _utc(ts: Any) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """One row per anchor, plus the run metadata needed to read them."""

    frame: pd.DataFrame          # ts, close, sigma, p_up, pred, actual, barrier, span...
    candles: pd.DataFrame        # OHLC over the reported range, for the chart
    run_id: str
    spec: dict[str, Any]
    features: list[str]
    dead: list[str]
    symbol: str
    interval: str
    bar: pd.Timedelta
    oos_start: pd.Timestamp | None   # estimated first out-of-sample anchor

    @property
    def resolved(self) -> pd.DataFrame:
        """Anchors whose barrier actually resolved — the ones accuracy is computed on."""
        return self.frame[self.frame["actual"] >= 0]

    @property
    def out_of_sample(self) -> pd.DataFrame:
        """Resolved anchors the model almost certainly did not train on."""
        res = self.resolved
        if self.oos_start is None:
            return res
        return res[pd.to_datetime(res["ts"], utc=True) >= self.oos_start]


# ── Core ─────────────────────────────────────────────────────────────────────


def _barrier_sigma(close: pd.Series, vol_window: int) -> np.ndarray:
    """Volatility the barriers are scaled by — identical to add_features' computation."""
    lr = np.log(close.astype(float) / close.astype(float).shift(1))
    return lr.rolling(vol_window, min_periods=vol_window).std().to_numpy()


def run_backtest(cfg: dict[str, Any], *, run_id: str | None = None) -> BacktestResult:
    """Score every anchor in the configured range. No network, no database."""
    symbol = str(cfg.get("symbol", "BTCUSDT"))
    interval = str(cfg.get("interval", "1h"))
    data_cfg = cfg.get("data") or {}
    trend_cfg_path = data_cfg.get("trend_config") or "technical_analysis_trainer/models/trend/config.yaml"
    with open(resolve_path(trend_cfg_path), encoding="utf-8") as f:
        trend_cfg = yaml.safe_load(f)

    # Which run to score. Falling through to the assembly config means "whatever
    # production is serving" is the default, which is the question people
    # actually want answered.
    if run_id is None:
        run_id = cfg.get("trend_run") or _run_id_for("trend", load_assembly_config())

    store = ArtifactStore(package_models_dir("trend"))
    scaler, features, clip_lo, clip_hi, run = store.load_trend_scaler_bundle(run_id)
    # Target semantics come from the artifact, never from config — the deployed
    # model defines what its own output means (see models/trend/target.py).
    spec = load_target_spec(store, run.run_id)
    seq_len = int((trend_cfg.get("window") or {}).get("long", 24))

    # ── Candles + cross-market/MTF, exactly as training assembled them ────────
    btc_path = data_cfg.get("btc") or "technical_analysis_trainer/data/train/btcusdt_1h.csv"
    df = load_dataset(str(resolve_path(btc_path)))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cross_cfg = trend_cfg.get("cross_market")
    mtf_cfg = trend_cfg.get("mtf")
    if (cross_cfg or {}).get("enabled", False):
        df = attach_cross_market_features(df, cross_cfg)
    if (mtf_cfg or {}).get("enabled", True):
        df = attach_mtf_features(df, mtf_cfg)

    # ── Features once over the whole series ───────────────────────────────────
    *_, df_feat = add_features(
        df,
        seq_len=seq_len,
        threshold=float(trend_cfg.get("threshold", 0.001)),
        horizon=int(trend_cfg.get("horizon", 1)),
        scaler=scaler,
        mtf_cfg=mtf_cfg,
        cross_market_cfg=cross_cfg,
        return_df_feat=True,
        clip_low=clip_lo,
        clip_high=clip_hi,
    )
    df_feat, _ = backfill_missing_features_with_scaler_means(df_feat, features, scaler)
    df_feat = df_feat.reset_index(drop=True)

    X = np.clip(
        scaler.transform(df_feat[features].values.astype(np.float32)), clip_lo, clip_hi
    ).astype(np.float32)
    # Same guard the live path applies: a feature that was constant while the
    # scaler was fitted is not standardised by it, so its raw value would reach
    # the model. Pin those to the training constant.
    dead = dead_features(scaler, features)
    if dead:
        X[:, [features.index(f) for f in dead]] = 0.0

    # ── Ground truth from the model's own target definition ───────────────────
    tb = trend_cfg.get("triple_barrier") or {}
    upper = float(spec.get("upper_mult", tb.get("upper_mult", 2.5)))
    lower = float(spec.get("lower_mult", tb.get("lower_mult", 2.5)))
    max_holding = int(spec.get("max_holding", tb.get("max_holding", 24)))
    vol_window = int(spec.get("vol_window", tb.get("vol_window", 72)))

    close = df_feat["close"].astype(float)
    sigma = _barrier_sigma(close, vol_window)
    labels, spans, _realized = triple_barrier_labels(
        close.to_numpy(dtype=np.float64),
        df_feat["high"].to_numpy(dtype=np.float64),
        df_feat["low"].to_numpy(dtype=np.float64),
        sigma,
        upper_mult=upper,
        lower_mult=lower,
        max_holding=max_holding,
    )

    # ── Anchors ───────────────────────────────────────────────────────────────
    # Window X[i-seq_len:i] predicts the barrier starting at close[i], so the
    # first usable anchor is i = seq_len and the last is the bar whose barrier
    # window still fits inside the data.
    n = len(df_feat)
    first, last = seq_len, n - max_holding - 1
    if last <= first:
        raise ValueError(f"not enough history: {n} feature rows, need > {seq_len + max_holding + 1}")

    idx = np.arange(first, last + 1)
    idx = _apply_range(idx, df_feat["ts"], cfg.get("range") or {})
    if len(idx) == 0:
        raise ValueError("configured range selects no anchors — check range.start / range.end")

    # ── Batched inference ─────────────────────────────────────────────────────
    # windows[k] == X[k : k + seq_len], so the window ending before anchor i is
    # windows[i - seq_len]. Same slice add_features and predict.py use.
    windows = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))[:, 0]
    batch = np.ascontiguousarray(windows[idx - seq_len])
    sess = ort.InferenceSession(str(run.onnx_path))
    name = sess.get_inputs()[0].name
    logits = np.concatenate(
        [
            np.asarray(sess.run(None, {name: batch[i : i + BATCH]})[0], dtype=np.float64)
            for i in range(0, len(batch), BATCH)
        ]
    )
    # TrendEnsemble emits log(mean(softmax(members))), so softmaxing here returns
    # the ensemble probability unchanged — exactly what run_trend_onnx does.
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_up = (e / e.sum(axis=1, keepdims=True))[:, 1]
    pred = (p_up >= 0.5).astype(int)

    anchor_close = close.to_numpy()[idx]
    anchor_sigma = sigma[idx]
    # The level the model named: up-barrier when it says UP, down-barrier when DOWN.
    barrier = anchor_close * np.exp(np.where(pred == 1, upper, -lower) * anchor_sigma)

    frame = pd.DataFrame(
        {
            "i": idx,
            "ts": df_feat["ts"].to_numpy()[idx],
            # The last bar the model actually saw, one before the barrier origin.
            "window_end_ts": df_feat["ts"].to_numpy()[idx - 1],
            "close": anchor_close,
            "sigma": anchor_sigma,
            "p_up": p_up,
            "pred": pred,
            "actual": labels[idx],
            "span": spans[idx],
            "barrier": barrier,
        }
    )
    frame["correct"] = np.where(frame["actual"] >= 0, frame["pred"] == frame["actual"], np.nan)

    bar = pd.Timedelta(hours=1) if interval == "1h" else pd.Timedelta(interval)
    oos_start = _estimated_oos_start(df, trend_cfg, seq_len, max_holding)
    lo_ts, hi_ts = frame["ts"].iloc[0], frame["ts"].iloc[-1] + max_holding * bar
    candles = df_feat[(df_feat["ts"] >= lo_ts - seq_len * bar) & (df_feat["ts"] <= hi_ts)][
        ["ts", "open", "high", "low", "close"]
    ].reset_index(drop=True)

    return BacktestResult(
        frame=frame, candles=candles, run_id=run.run_id, spec=spec, features=features,
        dead=dead, symbol=symbol, interval=interval, bar=bar, oos_start=oos_start,
    )


def _estimated_oos_start(
    df: pd.DataFrame, trend_cfg: dict[str, Any], seq_len: int, max_holding: int
) -> pd.Timestamp | None:
    """
    First timestamp the exported model was almost certainly *not* trained on.

    Without this the tool invites its own worst misreading: run it over all history
    and it reports accuracy ~58% / AUC 0.62, when most of that range is the model's
    own training data. The genuine out-of-sample number over the last fold's test
    window is several points lower.

    `train_v2.py` exports `folds[-1]`, whose test window is the final `test_ratio`
    of the series, so recomputing the same walk-forward split recovers the
    boundary. It is an *estimate*: the CSV has grown since training, which shifts
    every split index, so it is reported as approximate and never used to filter
    the headline metrics.
    """
    try:
        from technical_analysis_trainer.core.validation import walk_forward_splits

        splits = walk_forward_splits(
            len(df),
            n_folds=int(trend_cfg.get("n_folds", 3)),
            val_ratio=float(trend_cfg.get("val_ratio", 0.15)),
            test_ratio=float(trend_cfg.get("test_ratio", 0.15)),
            embargo=max_holding + seq_len,
        )
        if not splits:
            return None
        return _utc(df["ts"].iloc[splits[-1].test[0]])
    except Exception:  # noqa: BLE001 - a missing estimate must not break the report
        return None


def _apply_range(idx: np.ndarray, ts: pd.Series, rng: dict[str, Any]) -> np.ndarray:
    """Narrow the anchor set to the configured window. start/end beat last_bars."""
    times = pd.to_datetime(ts.to_numpy()[idx], utc=True)
    start, end = rng.get("start"), rng.get("end")
    keep = np.ones(len(idx), dtype=bool)
    if start:
        keep &= times >= _utc(start)
    if end:
        keep &= times <= _utc(end)
    idx = idx[keep]
    if not start and not end:
        last_bars = rng.get("last_bars")
        if last_bars:
            idx = idx[-int(last_bars):]
    stride = max(1, int(rng.get("stride", 1) or 1))
    return idx[::stride] if stride > 1 else idx


# ── Console report ───────────────────────────────────────────────────────────


def format_report(r: BacktestResult) -> str:
    """Every metric next to its baseline, in the repo's `── Section ──` style."""
    def pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    f, res = r.frame, r.resolved
    names = {0: "DOWN", 1: "UP"}
    lines = [
        "── Setup ──",
        f"  model run       {r.run_id}   ({len(r.features)} features)",
        f"  target          {r.spec.get('description', '?')}",
        f"  horizon         {horizon_label(r.spec)}",
        f"  symbol          {r.symbol} {r.interval}",
        f"  range           {f['ts'].iloc[0]}  ..  {f['ts'].iloc[-1]}",
        f"  anchors         {len(f)}",
    ]
    if r.dead:
        lines.append(f"  ⚠ dead features {', '.join(r.dead)}  (constant when the scaler was fitted)")
    flat = _flat_cross_market(f)
    if flat:
        lines.append(f"  ⚠ stale inputs  {flat}")

    unresolved = int((f["actual"] < 0).sum())
    lines += [
        "",
        "── Coverage ──",
        f"  scored          {len(res)}  of {len(f)} anchors",
        f"  unresolved      {unresolved}  ({pct(unresolved / max(len(f), 1))}) — neither barrier "
        f"touched within {r.spec.get('max_holding', '?')} bars, dropped in training too",
    ]

    if len(res) == 0 or res["actual"].nunique() < 2:
        return "\n".join(lines + ["", "  not enough resolved anchors for metrics"])

    y, p, prob = res["actual"].to_numpy(), res["pred"].to_numpy(), res["p_up"].to_numpy()
    correct = (y == p).astype(float)
    acc = float(correct.mean())
    lo, hi = bootstrap_ci(correct)
    majority = float(max(y.mean(), 1 - y.mean()))
    prec, rec, _f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    in_sample = len(res) - len(r.out_of_sample)
    if in_sample > 0:
        lines += [
            "",
            "── ⚠ In-sample overlap ──",
            f"  {in_sample} of {len(res)} scored anchors ({pct(in_sample / len(res))}) fall before "
            f"{r.oos_start:%Y-%m-%d}, which the model was almost certainly trained on.",
            "  Metrics below are therefore optimistic. See the out-of-sample block at the end;",
            "  the boundary is estimated by re-deriving the walk-forward split, so it is approximate.",
        ]

    lines += [
        "",
        "── Classification (whole selected range) ──",
        f"  accuracy        {pct(acc)}  (95% CI {pct(lo)}–{pct(hi)})",
        f"  ROC-AUC         {roc_auc_score(y, prob):.4f}",
        f"  majority base   {pct(majority)}",
        f"  edge            {(acc - majority) * 100:+.2f} pp",
        "",
        f"  {'':6}{'precision':>11}{'recall':>9}{'support':>9}",
        f"  {names[0]:6}{prec[0]:11.4f}{rec[0]:9.4f}{sup[0]:9d}",
        f"  {names[1]:6}{prec[1]:11.4f}{rec[1]:9.4f}{sup[1]:9d}",
        "",
        f"  confusion       actual DOWN / UP",
        f"    pred DOWN     {tn:6d} {fn:6d}",
        f"    pred UP       {fp:6d} {tp:6d}",
        "",
        f"  predicted       DOWN {pct((p == 0).mean())}   UP {pct((p == 1).mean())}",
        f"  actual          DOWN {pct((y == 0).mean())}   UP {pct((y == 1).mean())}",
    ]

    # Confidence is distance from 0.5: a small edge is usually concentrated in the
    # bars the model is most sure about, which a single accuracy number hides.
    conf = np.abs(prob - 0.5)
    order = np.argsort(conf)
    lines += ["", "── Accuracy by confidence decile ──",
              f"  {'decile':<8}{'|p-0.5|':>12}{'n':>7}{'accuracy':>11}"]
    for d in range(10):
        sel = order[int(d * len(order) / 10) : int((d + 1) * len(order) / 10)]
        if len(sel) == 0:
            continue
        lines.append(
            f"  {d + 1:<8}{conf[sel].mean():12.4f}{len(sel):7d}{pct(float(correct[sel].mean())):>11}"
        )
    top = order[-max(1, len(order) // 10):]
    lines.append(f"  top decile accuracy {pct(float(correct[top].mean()))}  (n={len(top)})")

    # The number to actually quote, when the range reaches back into training data.
    oos = r.out_of_sample
    if in_sample > 0 and len(oos) > 30 and oos["actual"].nunique() > 1:
        yo, po, probo = oos["actual"].to_numpy(), oos["pred"].to_numpy(), oos["p_up"].to_numpy()
        co = (yo == po).astype(float)
        olo, ohi = bootstrap_ci(co)
        omaj = float(max(yo.mean(), 1 - yo.mean()))
        lines += [
            "",
            f"── Out-of-sample only (from {r.oos_start:%Y-%m-%d}, estimated) ──",
            f"  n               {len(oos)}",
            f"  accuracy        {pct(float(co.mean()))}  (95% CI {pct(olo)}–{pct(ohi)})",
            f"  ROC-AUC         {roc_auc_score(yo, probo):.4f}",
            f"  majority base   {pct(omaj)}",
            f"  edge            {(float(co.mean()) - omaj) * 100:+.2f} pp",
        ]
    return "\n".join(lines)


def _flat_cross_market(frame: pd.DataFrame) -> str:
    """SOL's CSV stops in Apr 2026; say so instead of letting a dead input pass."""
    last = pd.Timestamp(frame["ts"].iloc[-1])
    sol_end = pd.Timestamp("2026-04-17", tz="UTC")
    if last > sol_end:
        return f"solusdt CSV ends {sol_end.date()}, xmkt_log_ret_sol is constant after that"
    return ""


# ── Chart ────────────────────────────────────────────────────────────────────


def build_figure(r: BacktestResult, chart_cfg: dict[str, Any]):
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    # ISO strings rather than Timestamps on the time axis. models/trend/test_cli.py
    # documents why numeric x breaks Candlestick in some Plotly/browser combos, and
    # raw Timestamps additionally make write_image() fail (kaleido's orjson cannot
    # serialise them) even though write_html() copes.
    from technical_analysis_trainer.models.trend.test_cli import (
        _coerce_float64_series,
        _ts_series_for_plotly,
    )

    def iso(t: Any) -> str:
        return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    max_holding = int(r.spec.get("max_holding", 24))
    show_unresolved = bool(chart_cfg.get("show_unresolved", True))
    max_lines = int(chart_cfg.get("max_lines", 750) or 0)

    f = r.frame if show_unresolved else r.frame[r.frame["actual"] >= 0]
    if max_lines and len(f) > max_lines:
        f = f.tail(max_lines)
    lo_ts = pd.Timestamp(f["ts"].iloc[0]) - 24 * r.bar
    candles = r.candles[r.candles["ts"] >= lo_ts]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=_ts_series_for_plotly(candles["ts"]),
            open=_coerce_float64_series(candles["open"]),
            high=_coerce_float64_series(candles["high"]),
            low=_coerce_float64_series(candles["low"]),
            close=_coerce_float64_series(candles["close"]),
            name=r.symbol,
        )
    )

    # Dark24 has exactly 24 colours and the horizon is exactly 24 bars, so no two
    # lines that overlap in time ever share one — which is the whole point of
    # colouring them differently.
    palette = qualitative.Dark24
    # Solid = the barrier the model named was hit first; dashed = the other one
    # won; dotted = nothing was touched inside the holding window.
    dash_for = {1: "solid", 0: "dash", -1: "dot"}
    label_for = {1: "correct", 0: "wrong", -1: "unresolved"}
    seen_groups: set[str] = set()

    for n, row in enumerate(f.itertuples(index=False)):
        actual = int(row.actual)
        outcome = -1 if actual < 0 else int(actual == int(row.pred))
        group = f"{'UP' if row.pred == 1 else 'DOWN'} · {label_for[outcome]}"
        pred_name = "UP" if row.pred == 1 else "DOWN"
        act_name = {0: "DOWN", 1: "UP"}.get(actual, "none touched")
        end_ts = pd.Timestamp(row.ts) + max_holding * r.bar
        fig.add_trace(
            go.Scatter(
                x=[iso(row.ts), iso(end_ts)],
                y=[float(row.close), float(row.barrier)],
                mode="lines",
                line=dict(color=palette[n % len(palette)], width=1.4, dash=dash_for[outcome]),
                legendgroup=group,
                name=group,
                showlegend=group not in seen_groups,
                hovertemplate=(
                    f"anchor {pd.Timestamp(row.ts):%Y-%m-%d %H:%M} UTC<br>"
                    f"window ends {pd.Timestamp(row.window_end_ts):%Y-%m-%d %H:%M}<br>"
                    f"p(UP) {row.p_up:.4f} → <b>{pred_name}</b><br>"
                    f"close {row.close:,.2f} → barrier {row.barrier:,.2f}"
                    f" ({(row.barrier / row.close - 1) * 100:+.2f}%)<br>"
                    f"actual: {act_name}"
                    + (f" after {int(row.span)} bars" if actual >= 0 else "")
                    + "<extra></extra>"
                ),
            )
        )
        seen_groups.add(group)

    res = r.resolved
    acc = float((res["pred"] == res["actual"]).mean()) if len(res) else float("nan")
    fig.update_layout(
        template=str(chart_cfg.get("template", "plotly_dark")),
        title=(
            f"{r.symbol} {r.interval} · trend run {r.run_id} · {horizon_label(r.spec)}<br>"
            f"<sub>{len(f)} of {len(r.frame)} anchors drawn · accuracy {acc * 100:.2f}% "
            f"on {len(res)} resolved · each line spans {max_holding} bars from its anchor "
            f"close to the predicted barrier</sub>"
        ),
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis_title="price",
        hovermode="closest",
        legend=dict(groupclick="togglegroup"),
    )
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest the trend model over historical candles")
    p.add_argument("--config", default=os.getenv("CONFIG_PATH", str(DEFAULT_CONFIG)))
    p.add_argument("--run", default=None, help="trend run id (default: what production serves)")
    p.add_argument("--start", default=None, help="ISO date/datetime UTC, overrides last_bars")
    p.add_argument("--end", default=None)
    p.add_argument("--last-bars", type=int, default=None,
                   help="score the most recent N anchors; 0 = the whole history")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--out", default=None, help="HTML output path")
    p.add_argument("--no-open", action="store_true", help="write the chart but do not open it")
    p.add_argument("--no-chart", action="store_true", help="console report only")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cfg = load_config(a.config)
    rng = cfg.setdefault("range", {})
    if a.start is not None:
        rng["start"] = a.start
    if a.end is not None:
        rng["end"] = a.end
    if a.last_bars is not None:
        rng["last_bars"] = a.last_bars
        rng["start"] = rng["end"] = None
    if a.stride is not None:
        rng["stride"] = a.stride
    chart_cfg = cfg.setdefault("chart", {})
    if a.out:
        chart_cfg["out"] = a.out

    try:
        result = run_backtest(cfg, run_id=a.run)
    except (FileNotFoundError, ValueError) as e:
        print(f"backtest ERROR: {e}")
        return 1

    print(format_report(result))

    if not a.no_chart:
        out = resolve_path(chart_cfg.get("out") or "artifacts/backtest.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        auto_open = bool(chart_cfg.get("open", True)) and not a.no_open
        build_figure(result, chart_cfg).write_html(str(out), auto_open=auto_open)
        print(f"\nChart → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
