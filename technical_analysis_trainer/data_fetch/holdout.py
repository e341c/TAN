"""
Candles the deployed models cannot have trained on, fetched and cached.

    python -m technical_analysis_trainer.data_fetch.holdout            # fetch if stale
    python -m technical_analysis_trainer.data_fetch.holdout --refresh  # fetch unconditionally
    python -m technical_analysis_trainer.data_fetch.holdout --status   # report, never fetch

Why this exists: every data-driven test and report in this repo read
`data/train/*.csv`, which is the exact series the exported models were fitted
on. Accuracy measured there is a memory test. This module supplies the other
thing — bars that did not exist yet when the artifact was written.

**The cutoff is the model's own run stamp.** Run ids are export timestamps
(`2026-08-17T15-47-17`), so a candle whose `ts` is later than the stamp provably
postdates training: the file was already on disk before that bar closed. It
needs no bookkeeping in `meta.json` and it cannot drift. It is conservative —
training data actually ended a few hours earlier — and conservative is the right
direction for a leak guard. With several models in play the cutoff is the
*latest* stamp among them, so one holdout window is out-of-sample for all of
them.

Everything lands in `data/holdout/` under the same basenames as `data/train/`,
so a path override is the only change a caller needs.

Fetching is cached, not live-per-run: a test suite that hits Binance on every
invocation is slow, rate-limited and fails offline, and its numbers move under
you between two runs of the same code. `ensure_holdout()` refetches only when
the cache is missing, older than `max_age_hours`, or too short to cover the
cutoff plus feature warm-up.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from technical_analysis_trainer.core.artifacts import resolve_path

HOLDOUT_DIR = "technical_analysis_trainer/data/holdout"

# How much history to keep before the cutoff. The binding constraints are the
# 168-bar cross-market correlation window, the 14-day MTF daily window and the
# 168-bar HAR-RV window on the quantile side; 6 months clears all of them with
# room for the cutoff to move as models are retrained.
DEFAULT_MONTHS = 6
# Bars of pre-cutoff history the holdout must contain before it is usable.
WARMUP_BARS = 1000
# Crypto is 24/7 on a 1h grid, so anything older than this is a stale cache.
DEFAULT_MAX_AGE_HOURS = 12

SYMBOLS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}
# FRED series -> (basename, tolerated staleness in days). Same split as
# data_fetch/macro_fred.py: DTWEXBGS is weekly with a release lag, VIXCLS daily.
MACRO = {"dxy": ("DTWEXBGS", "macro_dtwexbgs_1d", 14), "risk_proxy": ("VIXCLS", "macro_vixcls_1d", 5)}

RUN_STAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ── Cutoff ───────────────────────────────────────────────────────────────────


def run_stamp(run_id: str) -> pd.Timestamp:
    """Parse an export-stamp run id into the instant the artifact was written."""
    try:
        return pd.Timestamp(datetime.strptime(run_id, RUN_STAMP_FORMAT), tz="UTC")
    except ValueError as e:
        raise ValueError(
            f"run id {run_id!r} is not an export stamp ({RUN_STAMP_FORMAT}), so it cannot "
            "date the training data. Pass an explicit --cutoff."
        ) from e


def training_cutoff(models: tuple[str, ...] = ("trend", "quantile")) -> tuple[pd.Timestamp, dict[str, str]]:
    """
    First timestamp no served model can have trained on, and the runs it came from.

    Per model, the tightest bound available on what it actually saw:

      * `provenance.train_end_ts` in meta.json — what the run really trained on.
        Exact, and it is the reason a model exported today can still be evaluated
        on August: it was exported on 2026-09-07 but forbidden to read past
        2026-08-01, so its export stamp is a *useless* bound while its
        provenance is the true one.
      * otherwise the export stamp — a run id is the instant the artifact was
        written, so a later candle provably postdates it. The only bound
        available for artifacts predating provenance, and a conservative one.

    The overall cutoff is the max across models, so one window is out-of-sample
    for every model in the report and their numbers are comparable.
    """
    from technical_analysis_trainer.assembly.orchestrator import (
        _run_id_for,
        load_assembly_config,
    )
    from technical_analysis_trainer.core.artifacts import ArtifactStore
    from technical_analysis_trainer.models._paths import package_models_dir

    assembly = load_assembly_config()
    runs: dict[str, str] = {}
    for m in models:
        # Unpinned in assembly/config.yaml means production serves whichever run
        # has the newest scaler, so the cutoff has to follow the same rule.
        runs[m] = str(_run_id_for(m, assembly) or ArtifactStore(package_models_dir(m)).latest_run_id())
    return max(model_cutoff(m, runs[m]) for m in models), runs


def model_cutoff(model: str, run_id: str) -> pd.Timestamp:
    """Tightest defensible bound on what one artifact was trained on."""
    from technical_analysis_trainer.core.artifacts import ArtifactStore
    from technical_analysis_trainer.models._paths import package_models_dir

    try:
        run = ArtifactStore(package_models_dir(model)).get_run(run_id)
        meta = run.run_dir / f"meta_{run_id}.json"
        if meta.is_file():
            with open(meta, encoding="utf-8") as f:
                prov = (json.load(f) or {}).get("provenance") or {}
            # `frozen_eval_start` first, and it is not the same as train_end_ts.
            # The weights stop at train_end_ts, but the calibrator and the
            # decision threshold are fitted on the span between the two, and both
            # ship inside the artifact. Anchors in that gap are unseen by the
            # network and seen by the correction applied on top of it, so grading
            # there measures a pipeline that has partly read its own answers.
            for key in ("frozen_eval_start", "train_end_ts"):
                end = prov.get(key)
                if end and str(end).lower() != "none":
                    return _utc(end)
    except (OSError, ValueError, KeyError, FileNotFoundError) as e:
        # Narrow on purpose. A bare `except Exception` here hid a NameError for a
        # missing helper and silently returned the export stamp, which looks like
        # a legitimate answer — the report then covered zero bars instead of five
        # weeks. A malformed artifact should degrade to the stamp; a bug should not.
        print(f"  ⚠ {model}/{run_id}: could not read provenance ({e}); using the export stamp")
    return run_stamp(run_id)


# ── Cache ────────────────────────────────────────────────────────────────────


@dataclass
class Holdout:
    """Where the unseen candles live and what they cover."""

    cutoff: pd.Timestamp
    runs: dict[str, str]
    dir: Path
    paths: dict[str, Path]
    fetched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def btc(self) -> Path:
        return self.paths["btc"]

    def rel(self, key: str) -> str | None:
        """Repo-relative path, which is what the configs and CLIs take."""
        from technical_analysis_trainer.core.artifacts import repo_root

        p = self.paths.get(key)
        return None if p is None else str(p.relative_to(repo_root()))

    def span(self, key: str = "btc") -> tuple[pd.Timestamp, pd.Timestamp]:
        ts = _timestamps(self.paths[key])
        return ts.min(), ts.max()

    def unseen_bars(self, key: str = "btc") -> int:
        ts = _timestamps(self.paths[key])
        return int((ts > self.cutoff).sum())


def _timestamps(path: Path) -> pd.Series:
    return pd.to_datetime(pd.read_csv(path, usecols=["ts"])["ts"], utc=True)


def _cache_verdict(
    path: Path, cutoff: pd.Timestamp, max_age_hours: float, warmup_bars: int
) -> str | None:
    """Reason the cache cannot be used, or None when it is good."""
    if not path.is_file():
        return "missing"
    try:
        ts = _timestamps(path)
    except Exception as e:  # noqa: BLE001 - a corrupt cache is just a cache miss
        return f"unreadable ({e})"
    if ts.empty:
        return "empty"
    age_h = (pd.Timestamp.now(tz="UTC") - ts.max()).total_seconds() / 3600
    if age_h > max_age_hours:
        return f"stale (ends {ts.max():%Y-%m-%d %H:%M}, {age_h:.0f}h ago)"
    if ts.max() <= cutoff:
        return f"no unseen bars (ends {ts.max():%Y-%m-%d %H:%M}, cutoff {cutoff:%Y-%m-%d %H:%M})"
    # Features are built over the whole series, so the file must reach back well
    # before the cutoff or the first unseen anchors are scored on warm-up NaNs.
    if int((ts <= cutoff).sum()) < warmup_bars:
        return f"too little warm-up ({int((ts <= cutoff).sum())} bars before the cutoff, need {warmup_bars})"
    return None


def _macro_verdict(path: Path, tolerance_days: int) -> str | None:
    if not path.is_file():
        return "missing"
    try:
        ts = _timestamps(path)
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({e})"
    if ts.empty:
        return "empty"
    age_d = (pd.Timestamp.now(tz="UTC") - ts.max()).days
    return f"stale (ends {ts.max():%Y-%m-%d}, {age_d}d ago)" if age_d > tolerance_days else None


# ── Fetch ────────────────────────────────────────────────────────────────────


def _fetch_klines(symbol: str, out: Path, months: int, with_funding: bool) -> None:
    from technical_analysis_trainer.data_fetch.binance_klines import (
        fetch_funding_rate,
        fetch_klines,
        merge_funding_rate,
    )

    df = fetch_klines(symbol, "1h", months)
    if with_funding:
        # The trend model has funding_rate features; a BTC file without them
        # would be silently backfilled with scaler means at scoring time.
        df = merge_funding_rate(df, fetch_funding_rate(symbol, months))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  ✅ {out.name}  {len(df)} rows  {df['ts'].min()} .. {df['ts'].max()}")


def _fetch_macro(series_id: str, out: Path) -> None:
    from technical_analysis_trainer.data_fetch.macro_fred import fetch_series

    df = fetch_series(series_id, start="2018-01-01")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  ✅ {out.name}  {len(df)} rows  ends {df['ts'].max():%Y-%m-%d}")


# ── Entry point ──────────────────────────────────────────────────────────────


def ensure_holdout(
    *,
    cutoff: pd.Timestamp | None = None,
    months: int = DEFAULT_MONTHS,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    warmup_bars: int = WARMUP_BARS,
    refresh: bool = False,
    offline: bool = False,
    quiet: bool = False,
) -> Holdout:
    """
    Return a `Holdout` pointing at candles later than the models' export stamps.

    Fetches only what the cache cannot supply. `offline=True` never fetches and
    raises if the crypto cache is unusable; `refresh=True` refetches everything.
    """
    resolved_cutoff, runs = training_cutoff()
    if cutoff is not None:
        resolved_cutoff = cutoff
    root = resolve_path(HOLDOUT_DIR)
    paths = {k: root / f"{sym.lower()}_1h.csv" for k, sym in SYMBOLS.items()}
    paths |= {k: root / f"{base}.csv" for k, (_, base, _) in MACRO.items()}

    h = Holdout(cutoff=resolved_cutoff, runs=runs, dir=root, paths=paths)
    say = (lambda *a: None) if quiet else print

    say(f"Holdout cutoff {resolved_cutoff:%Y-%m-%d %H:%M} UTC "
        f"({', '.join(f'{m}={r}' for m, r in runs.items())})")

    for key, symbol in SYMBOLS.items():
        path = paths[key]
        why = "forced by --refresh" if refresh else _cache_verdict(
            path, resolved_cutoff, max_age_hours, warmup_bars
        )
        if why is None:
            continue
        if offline:
            if key == "btc":
                raise FileNotFoundError(
                    f"holdout cache for {symbol} is unusable ({why}) and --offline forbids "
                    f"fetching. Run: python -m {__name__}"
                )
            h.warnings.append(f"{symbol}: {why} (offline, not refetched)")
            continue
        say(f"  fetching {symbol}: cache {why}")
        _fetch_klines(symbol, path, months, with_funding=(key == "btc"))
        h.fetched.append(symbol)

    for key, (series_id, _base, tol) in MACRO.items():
        path = paths[key]
        why = "forced by --refresh" if refresh else _macro_verdict(path, tol)
        if why is None:
            continue
        if offline:
            h.warnings.append(f"{series_id}: {why} (offline, not refetched)")
            continue
        try:
            say(f"  fetching {series_id}: cache {why}")
            _fetch_macro(series_id, path)
            h.fetched.append(series_id)
        except Exception as e:  # noqa: BLE001
            # FRED needs network and, for heavy use, an API key. A missing macro
            # series flattens two of 52 features rather than breaking the run, so
            # it is a warning — but a loud one, because the failure is silent
            # downstream (attach_cross_market_features forward-fills).
            h.warnings.append(f"{series_id}: fetch failed ({e}); xmkt_log_ret_{key} will be flat")

    # Drop macro entries the cache never supplied, so callers do not point the
    # trend config at a file that is not there.
    for key in list(MACRO):
        if not paths[key].is_file():
            h.paths.pop(key, None)

    if not quiet:
        lo, hi = h.span()
        print(f"  holdout {lo:%Y-%m-%d} .. {hi:%Y-%m-%d %H:%M}  "
              f"({h.unseen_bars()} bars after the cutoff)")
        for w in h.warnings:
            print(f"  ⚠ {w}")
    return h


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh", action="store_true", help="refetch even when the cache is fresh")
    p.add_argument("--status", action="store_true", help="report the cache, never fetch")
    p.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    p.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    a = p.parse_args()

    if a.status:
        cutoff, runs = training_cutoff()
        root = resolve_path(HOLDOUT_DIR)
        print(f"cutoff {cutoff:%Y-%m-%d %H:%M} UTC  ({', '.join(f'{m}={r}' for m, r in runs.items())})")
        print(f"dir    {root}")
        for key, sym in SYMBOLS.items():
            path = root / f"{sym.lower()}_1h.csv"
            why = _cache_verdict(path, cutoff, a.max_age_hours, WARMUP_BARS)
            if why is None:
                ts = _timestamps(path)
                print(f"  ok      {path.name:22} rows={len(ts):6} unseen={int((ts > cutoff).sum()):5} "
                      f"last={ts.max():%Y-%m-%d %H:%M}")
            else:
                print(f"  REFETCH {path.name:22} {why}")
        for key, (series_id, base, tol) in MACRO.items():
            path = root / f"{base}.csv"
            why = _macro_verdict(path, tol)
            print(f"  {'ok     ' if why is None else 'REFETCH'} {path.name:22} {why or ''}")
        return 0

    ensure_holdout(months=a.months, max_age_hours=a.max_age_hours, refresh=a.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
