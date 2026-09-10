"""
Collect Binance futures positioning data — the one thing that must start today.

    PYTHONPATH=. python -m technical_analysis_trainer.data_fetch.binance_derivatives
    PYTHONPATH=. python -m technical_analysis_trainer.data_fetch.binance_derivatives --status

WHY THIS IS URGENT AND NOTHING ELSE HERE IS.

Plan section 11 set the exit criterion for direction work: if order-flow features
do not move validation AUC, the answer is a different DATA CLASS, not a different
model. Section 19.5 recorded that five approaches all landed at AUC ~0.52, and
14.4 showed the free order-flow proxy (taker buy volume, already in the klines)
gives nothing measurable. So the next real attempt at direction needs order-book
and positioning data.

That data cannot be backfilled. Measured against the live API on 2026-09-09:

    startTime 30 days ago   -> accepted
    startTime 31 days ago   -> HTTP 400, "parameter 'startTime' is invalid"
    oldest 1h point on offer -> 2026-08-19, about 21 days back
    /fapi/v1/allForceOrders  -> 404, the public liquidation endpoint is gone

There is no archive. Every series below exists only from the moment something
starts writing it down, so the length of the eventual training set is decided by
when this first runs, not by how much effort goes in later. A month of delay is a
month that can never be recovered — which is the entire argument for running this
before any further modelling.

WHAT IT COLLECTS (1h, matching the model's bar)

    open_interest            sumOpenInterest / sumOpenInterestValue
    top_trader_account_ratio topLongShortAccountRatio    (accounts)
    top_trader_position_ratio topLongShortPositionRatio  (position size)
    global_account_ratio     globalLongShortAccountRatio (all accounts)
    taker_buy_sell_ratio     takerlongshortRatio         (aggressive flow)
    basis / basis_rate       perp vs index, the carry signal

Not collected, and worth stating so nobody assumes otherwise:

    liquidations   the REST endpoint is gone; forced flow is only available live
                   on the !forceOrder websocket stream, which is a daemon rather
                   than a cron job. Worth adding, separately.
    book depth     snapshots only, no history at all, also a daemon.

HOW IT BEHAVES

Append-only and idempotent: it reads what is already on disk, fetches only what
is missing, merges on `ts` and de-duplicates. Running it twice changes nothing;
running it hourly keeps the file current. It never truncates, because the data
it holds is irreplaceable.

It also reports gaps. If collection lapses for more than the retention window the
hole is permanent, so a silent cron failure is the main risk this file exists to
survive — `--status` prints coverage and the largest gap so that failure is
visible before it has cost a month.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests

BASE = "https://fapi.binance.com"
DEFAULT_OUT = "technical_analysis_trainer/data/derivatives"
# Binance rejects startTime older than this outright. Measured, not assumed.
RETENTION_DAYS = 30
MAX_LIMIT = 500

# endpoint -> (path, extra params, {api field: our column})
SERIES: dict[str, tuple[str, dict, dict]] = {
    "open_interest": (
        "/futures/data/openInterestHist", {},
        {"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"},
    ),
    "top_account_ratio": (
        "/futures/data/topLongShortAccountRatio", {},
        {"longShortRatio": "top_account_ls_ratio", "longAccount": "top_account_long"},
    ),
    "top_position_ratio": (
        "/futures/data/topLongShortPositionRatio", {},
        {"longShortRatio": "top_position_ls_ratio", "longAccount": "top_position_long"},
    ),
    "global_account_ratio": (
        "/futures/data/globalLongShortAccountRatio", {},
        {"longShortRatio": "global_account_ls_ratio", "longAccount": "global_account_long"},
    ),
    "taker_ratio": (
        "/futures/data/takerlongshortRatio", {},
        {"buySellRatio": "taker_buy_sell_ratio", "buyVol": "taker_buy_vol",
         "sellVol": "taker_sell_vol"},
    ),
    # `annualizedBasisRate` is deliberately not taken: it comes back as an empty
    # string for PERPETUAL (there is no expiry to annualise against), so it would
    # be a column of NaN pretending to be a feature. `futuresPrice` is the useful
    # half — perp mark against index is the carry signal in directly usable form.
    "basis": (
        "/futures/data/basis", {"contractType": "PERPETUAL"},
        {"basis": "basis", "basisRate": "basis_rate",
         "indexPrice": "index_price", "futuresPrice": "futures_price"},
    ),
}


def _fetch(path: str, params: dict) -> list[dict]:
    for attempt in range(4):
        r = requests.get(BASE + path, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        # 418/429 are rate limits; everything else is worth surfacing.
        if r.status_code in (418, 429):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"{path} -> HTTP {r.status_code}: {r.text[:160]}")
    raise RuntimeError(f"{path}: rate-limited after 4 attempts")


def fetch_series(name: str, symbol: str, period: str, start: pd.Timestamp | None) -> pd.DataFrame:
    """One series, paged forward from `start` (or as far back as retention allows)."""
    path, extra, cols = SERIES[name]
    # `basis` keys the instrument as `pair`; the rest use `symbol`.
    key = "pair" if name == "basis" else "symbol"
    # Everything here is tz-aware UTC. Mixing naive and aware timestamps is the
    # kind of thing that raises on the first run and never again, so it is kept
    # uniform rather than converted at the comparison sites.
    now = pd.Timestamp.now(tz="UTC")
    floor = now - pd.Timedelta(days=RETENTION_DAYS - 1)
    cursor = max(start, floor) if start is not None else floor

    frames, guard = [], 0
    while guard < 60:
        guard += 1
        params = {key: symbol, "period": period, "limit": MAX_LIMIT,
                  "startTime": int(cursor.timestamp() * 1000),
                  # /futures/data/basis rejects a request without endTime
                  # ("Mandatory parameter 'endtime' ... malformed"). The others
                  # accept it, so it is sent unconditionally.
                  "endTime": int(now.timestamp() * 1000), **extra}
        data = _fetch(path, params)
        if not data:
            break
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        keep = {k: v for k, v in cols.items() if k in df.columns}
        out = df[["ts"] + list(keep)].rename(columns=keep)
        for c in keep.values():
            out[c] = pd.to_numeric(out[c], errors="coerce")
        frames.append(out)
        last = out["ts"].max()
        if len(data) < MAX_LIMIT or last <= cursor:
            break
        cursor = last + pd.Timedelta(seconds=1)
        time.sleep(0.25)

    if not frames:
        return pd.DataFrame(columns=["ts"])
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))


def merge_history(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Fold a freshly-fetched frame into the stored history without losing anything.

    This is the only place irreplaceable data can be destroyed, so the rules are
    explicit rather than emergent from a `concat`:

      * every timestamp in either frame survives — the file is append-only;
      * every column in either frame survives, so adding a series later does not
        erase what was collected before it existed;
      * on a conflicting timestamp the NEW value wins, because the most recent
        bars are provisional and a re-fetch is a correction — but only where the
        new frame actually has a value. A NaN from an endpoint that failed this
        run must never overwrite a real number already on disk.

    A correction, because the comment here previously claimed otherwise: the
    obvious `concat` + `groupby("ts").last()` does NOT have that last bug.
    `GroupBy.last()` skips NaN by default, so a failed endpoint's NaN loses to
    the stored value on its own. The two implementations agree on every case in
    test_derivatives.py, and the tests pass against both.

    This version is kept anyway, for reasons that are about maintenance rather
    than correctness: the NaN-skipping is an easily-missed default rather than a
    stated intent, and it is not shared by the neighbouring idioms — `.nth(-1)`
    and `.tail(1)` both propagate the NaN and would silently erase a column. An
    explicit `where(notna)` says what the file needs and cannot be broken by
    someone swapping in a similar-looking aggregation.
    """
    if existing is None or existing.empty:
        return new.sort_values("ts").reset_index(drop=True)
    if new is None or new.empty:
        return existing.sort_values("ts").reset_index(drop=True)

    old = existing.set_index("ts")
    fresh = new.set_index("ts")
    combined = old.reindex(old.index.union(fresh.index))
    for col in fresh.columns:
        incoming = fresh[col].reindex(combined.index)
        if col in combined.columns:
            # `.where(notna)` keeps the stored value wherever the fetch had none.
            combined[col] = incoming.where(incoming.notna(), combined[col])
        else:
            combined[col] = incoming
    return combined.rename_axis("ts").reset_index().sort_values("ts").reset_index(drop=True)


def collect(symbol: str, period: str, out_dir: str) -> pd.DataFrame:
    path = os.path.join(out_dir, f"{symbol.lower()}_{period}_derivatives.csv")
    os.makedirs(out_dir, exist_ok=True)

    existing = pd.DataFrame()
    if os.path.isfile(path):
        existing = pd.read_csv(path)
        existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
    # Re-fetch the last day even when the file is current: the most recent bars
    # can still be revised, and overlapping costs one request.
    start = None
    if len(existing):
        start = existing["ts"].max() - pd.Timedelta(days=1)

    merged: pd.DataFrame | None = None
    for name in SERIES:
        try:
            s = fetch_series(name, symbol, period, start)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ {name}: {type(e).__name__}: {e}")
            continue
        got = "" if s.empty else f"{s['ts'].min():%Y-%m-%d %H:%M} .. {s['ts'].max():%Y-%m-%d %H:%M}"
        print(f"  {name:24s} {len(s):5d} rows  {got}")
        if s.empty:
            continue
        merged = s if merged is None else merged.merge(s, on="ts", how="outer")

    if merged is None or merged.empty:
        print("  nothing fetched")
        return existing

    combined = merge_history(existing, merged)
    combined.to_csv(path, index=False)

    added = len(combined) - len(existing)
    print(f"\n  {path}")
    print(f"  {len(combined)} rows total ({added:+d})  "
          f"{combined['ts'].min():%Y-%m-%d %H:%M} .. {combined['ts'].max():%Y-%m-%d %H:%M}")
    return combined


def report(symbol: str, period: str, out_dir: str) -> int:
    path = os.path.join(out_dir, f"{symbol.lower()}_{period}_derivatives.csv")
    if not os.path.isfile(path):
        print(f"no file at {path} — nothing has been collected yet.")
        print("Every hour before the first run is an hour that cannot be recovered.")
        return 1
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    span = df["ts"].max() - df["ts"].min()
    step = pd.Timedelta(period)
    gaps = df["ts"].diff().dropna()
    big = gaps[gaps > step * 1.5]
    age = pd.Timestamp.now(tz="UTC") - df["ts"].max()

    print(f"{path}")
    print(f"  rows        {len(df)}")
    print(f"  span        {df['ts'].min():%Y-%m-%d %H:%M} .. {df['ts'].max():%Y-%m-%d %H:%M}"
          f"  ({span.days}d {span.components.hours}h)")
    print(f"  last point  {age.total_seconds()/3600:.1f}h old")
    print(f"  gaps        {len(big)}"
          + (f", largest {big.max()}" if len(big) else ""))
    print(f"  columns     {', '.join(c for c in df.columns if c != 'ts')}")
    if age > pd.Timedelta(days=RETENTION_DAYS):
        print(f"\n  ⚠ the last point is older than the {RETENTION_DAYS}-day retention window.")
        print("    Everything between then and now is gone permanently.")
    elif age > pd.Timedelta(days=2):
        print(f"\n  ⚠ collection is {age.days} days behind. It is still recoverable, but only")
        print(f"    for another {RETENTION_DAYS - age.days} days.")
    # A rough sense of when this is worth modelling on.
    months = span.days / 30.0
    print(f"\n  usable history: {months:.1f} months. The rolling harness wants ~18 monthly")
    print("  origins plus a training span, so this becomes testable at roughly 2 years.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--period", default="1h",
                   help="5m/15m/30m/1h/2h/4h/6h/12h/1d — 1h matches the trend model")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--status", action="store_true", help="report coverage and gaps, fetch nothing")
    a = p.parse_args()

    if a.status:
        return report(a.symbol, a.period, a.out)
    print(f"collecting {a.symbol} {a.period} futures positioning data")
    print(f"(retention is {RETENTION_DAYS} days — anything older is already lost)\n")
    collect(a.symbol, a.period, a.out)
    print()
    return report(a.symbol, a.period, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
