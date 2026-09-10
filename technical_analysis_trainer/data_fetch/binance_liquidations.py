"""
Stream Binance futures liquidations. Live-only — there is no history to fetch.

    PYTHONPATH=. python -m technical_analysis_trainer.data_fetch.binance_liquidations --run
    PYTHONPATH=. python -m technical_analysis_trainer.data_fetch.binance_liquidations --aggregate
    PYTHONPATH=. python -m technical_analysis_trainer.data_fetch.binance_liquidations --status

Forced flow is the most-cited short-horizon predictor this repo does not have.
A liquidation is not a trade someone chose to make: it is the exchange closing a
position at market because margin ran out, so it is mechanically price-moving and
mechanically clustered — which is exactly the shape of thing a 24-bar barrier
model could use.

WHY THIS IS A DAEMON AND NOT A CRON JOB

`/fapi/v1/allForceOrders` returns 404; the public REST endpoint for historical
liquidations is gone (measured 2026-09-09). The only source is the websocket
stream, which sends events as they happen and remembers nothing. Where
`binance_derivatives.py` can recover from a 29-day outage, this cannot recover
from a 29-SECOND one. Anything that happens while nothing is listening is gone.

ZERO IS A MEASUREMENT, NOT A GAP

This is the design constraint that shapes the file. A calm hour genuinely has no
liquidations, and a dead collector also produces no rows. Those must not look
alike, or the eventual feature is "liquidations, except when the process was
down, where the value is a confident zero". So a heartbeat is written on a fixed
interval whether or not anything arrived, and coverage is computed from
heartbeats rather than from events. An hour with a heartbeat and no events is
real data; an hour with neither is a hole and is reported as one.

A CANARY, BECAUSE A PROCESS HEARTBEAT IS NOT ENOUGH

The first version had exactly the bug it was written to prevent, and only found
it by accident. Connecting to fstream.binance.com from one network succeeded —
`websockets` reported an open socket, state 1 — and then delivered nothing at
all, for any stream, including `btcusdt@aggTrade` which normally fires several
times a second. Spot (`stream.binance.com:9443`) worked from the same machine and
REST to fapi worked, so nothing looked broken. A process heartbeat would have
written "alive, no liquidations" every minute forever and called it data.

So the collector subscribes to `markPrice@1s` alongside the liquidation stream
purely as a liveness signal: the exchange sends it once a second, unconditionally.
A heartbeat is only written while the canary is fresh. If the canary goes quiet
the socket is dead however healthy it looks, coverage stops being claimed, and
the reconnect loop takes over. "Covered" therefore means the exchange was
demonstrably talking to us, not merely that this process was running.

SIDE SEMANTICS, WHICH ARE EASY TO INVERT

The stream's `S` field is the side of the LIQUIDATING order, not of the position
that died. A forced SELL closes a LONG. So:

    S = SELL  ->  a long was liquidated  ->  long_liquidations
    S = BUY   ->  a short was liquidated ->  short_liquidations

Getting this backwards would flip the sign of the feature while leaving every
magnitude plausible, so it is asserted in the tests rather than trusted.

STORAGE

Raw events are the source of truth and are appended as they arrive; the hourly
aggregate is derived and can always be rebuilt with --aggregate. Raw volume is
small (BTCUSDT liquidations are order hundreds per day, not millions).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import signal
from datetime import datetime, timezone

import pandas as pd

WS_URL = "wss://fstream.binance.com/ws"
COMBINED_URL = "wss://fstream.binance.com/stream"
# The canary must be seen inside this window for coverage to be claimed.
CANARY_STALE_SECONDS = 15
DEFAULT_OUT = "technical_analysis_trainer/data/derivatives"
# How often to record "the collector was alive at this moment" even in silence.
HEARTBEAT_SECONDS = 60
RAW_FIELDS = ["ts", "symbol", "side", "position_closed", "price", "qty", "quote"]


def _paths(symbol: str, out_dir: str) -> tuple[str, str, str]:
    s = symbol.lower()
    return (os.path.join(out_dir, f"{s}_liquidations_raw.csv"),
            os.path.join(out_dir, f"{s}_liquidations_heartbeat.csv"),
            os.path.join(out_dir, f"{s}_1h_liquidations.csv"))


def _append(path: str, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
        # Liquidations arrive in bursts and the process can be killed at any
        # moment; an event held in a buffer is an event that never existed.
        f.flush()
        os.fsync(f.fileno())


def parse_event(msg: dict) -> dict | None:
    """One forceOrder event -> one raw row, or None if it is not one."""
    if msg.get("e") != "forceOrder":
        return None
    o = msg.get("o") or {}
    side = str(o.get("S", "")).upper()
    qty = float(o.get("z") or o.get("q") or 0.0)     # accumulated filled qty
    price = float(o.get("ap") or o.get("p") or 0.0)  # average fill price
    return {
        "ts": pd.to_datetime(int(o.get("T") or msg.get("E")), unit="ms", utc=True).isoformat(),
        "symbol": o.get("s", ""),
        "side": side,
        # See the module docstring: a forced SELL is a LONG being closed.
        "position_closed": "long" if side == "SELL" else "short",
        "price": price,
        "qty": qty,
        "quote": price * qty,
    }


async def stream(symbol: str, out_dir: str) -> None:
    import websockets

    raw_path, hb_path, _ = _paths(symbol, out_dir)
    s = symbol.lower()
    # forceOrder is the payload; markPrice@1s is the canary (see the docstring).
    url = f"{COMBINED_URL}?streams={s}@forceOrder/{s}@markPrice@1s"
    # 0.0 = the canary has NEVER been seen. Deliberately not seeded with the
    # current time on connect: an open socket is not evidence, and a grace
    # period would write one heartbeat for an unproven connection — which is
    # precisely the false coverage this whole mechanism exists to prevent.
    last_canary = 0.0
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:      # not available on all platforms
            pass

    async def heartbeat():
        warned = False
        while not stop.is_set():
            fresh = (loop.time() - last_canary) < CANARY_STALE_SECONDS
            if fresh:
                _append(hb_path, ["ts"], [{"ts": datetime.now(timezone.utc).isoformat()}])
                warned = False
            elif not warned:
                # Deliberately NOT writing a heartbeat: an unproven hour must
                # look like a hole, not like a quiet market.
                print("  ⚠ canary stale — the socket is open but the exchange is "
                      "not sending. Coverage is NOT being claimed for this period.")
                warned = True
            try:
                await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                pass

    hb = asyncio.create_task(heartbeat())
    backoff, seen = 1, 0
    print(f"streaming {url}")
    print(f"  raw       -> {raw_path}")
    print(f"  heartbeat -> {hb_path}  (every {HEARTBEAT_SECONDS}s)")
    try:
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    print(f"  connected {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
                    backoff = 1
                    while not stop.is_set():
                        envelope = json.loads(await ws.recv())
                        # /stream?streams=... wraps payloads as {stream, data}.
                        msg = envelope.get("data", envelope)
                        if msg.get("e") == "markPriceUpdate":
                            last_canary = loop.time()
                            continue
                        row = parse_event(msg)
                        if row is None:
                            continue
                        _append(raw_path, RAW_FIELDS, [row])
                        seen += 1
                        print(f"  {row['ts'][:19]}  {row['position_closed']:5s} "
                              f"{row['qty']:>10.4f} @ {row['price']:>10.2f}  "
                              f"= {row['quote']:>12,.0f} USDT")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                if stop.is_set():
                    break
                # A reconnect gap is a permanent hole, so it is logged loudly
                # rather than swallowed.
                print(f"  ⚠ disconnected ({type(e).__name__}: {e}); retry in {backoff}s")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60)
    finally:
        hb.cancel()
        print(f"\nstopped after {seen} liquidation event(s)")


def aggregate(symbol: str, out_dir: str) -> pd.DataFrame:
    """Hourly aggregate, rebuilt from raw. Hours with a heartbeat but no events are zeros."""
    raw_path, hb_path, out_path = _paths(symbol, out_dir)
    if not os.path.isfile(hb_path):
        print("no heartbeat file — nothing has been collected yet")
        return pd.DataFrame()

    hb = pd.read_csv(hb_path)
    hb["ts"] = pd.to_datetime(hb["ts"], utc=True)
    covered = set(hb["ts"].dt.floor("h"))

    raw = None
    if os.path.isfile(raw_path):
        raw = pd.read_csv(raw_path)
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
        raw["hour"] = raw["ts"].dt.floor("h")
        # Receiving an event IS proof the exchange was talking to us, so an hour
        # holding one is covered whether or not a heartbeat happened to land in
        # it. Without this, an event arriving before the first beat of a session
        # would be dropped on the floor.
        covered |= set(raw["hour"])

    hours = pd.DatetimeIndex(sorted(covered))
    frame = pd.DataFrame(index=hours).rename_axis("ts")

    if raw is not None:
        for side in ("long", "short"):
            s = raw[raw["position_closed"] == side].groupby("hour")
            frame[f"liq_{side}_quote"] = s["quote"].sum()
            frame[f"liq_{side}_count"] = s["quote"].size()
        frame["liq_max_quote"] = raw.groupby("hour")["quote"].max()
    for c in ("liq_long_quote", "liq_short_quote", "liq_long_count",
              "liq_short_count", "liq_max_quote"):
        if c not in frame.columns:
            frame[c] = 0.0
    # A covered hour with no liquidations is a real zero, not a missing value.
    frame = frame.fillna(0.0)
    frame["liq_net_quote"] = frame["liq_short_quote"] - frame["liq_long_quote"]
    frame["liq_total_quote"] = frame["liq_short_quote"] + frame["liq_long_quote"]

    out = frame.reset_index()
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"{out_path}: {len(out)} covered hour(s)")
    return out


def status(symbol: str, out_dir: str) -> int:
    raw_path, hb_path, out_path = _paths(symbol, out_dir)
    if not os.path.isfile(hb_path):
        print(f"no heartbeat at {hb_path} — the collector has never run.")
        print("Liquidations cannot be backfilled; until this runs, there is no data.")
        return 1
    hb = pd.read_csv(hb_path)
    hb["ts"] = pd.to_datetime(hb["ts"], utc=True)
    age = pd.Timestamp.now(tz="UTC") - hb["ts"].max()
    gaps = hb["ts"].diff().dropna()
    big = gaps[gaps > pd.Timedelta(seconds=HEARTBEAT_SECONDS * 3)]
    covered = hb["ts"].dt.floor("h").nunique()

    print(f"heartbeat   {hb['ts'].min():%Y-%m-%d %H:%M} .. {hb['ts'].max():%Y-%m-%d %H:%M}"
          f"  ({len(hb)} beats)")
    print(f"  last beat {age.total_seconds()/60:.1f} min old"
          + ("  ⚠ collector is not running" if age > pd.Timedelta(minutes=5) else ""))
    print(f"  hours with coverage {covered}")
    print(f"  outages     {len(big)}" + (f", longest {big.max()}" if len(big) else ""))
    if os.path.isfile(raw_path):
        raw = pd.read_csv(raw_path)
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
        print(f"events      {len(raw)}  "
              f"({(raw['position_closed'] == 'long').sum()} long, "
              f"{(raw['position_closed'] == 'short').sum()} short)")
        if len(raw):
            print(f"  notional  {raw['quote'].sum():,.0f} USDT   "
                  f"largest {raw['quote'].max():,.0f}")
    else:
        print("events      0 (no liquidations seen yet — with a live heartbeat that is data)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--run", action="store_true", help="stream until interrupted")
    p.add_argument("--seconds", type=float, default=None,
                   help="with --run, stop after this long (for smoke tests)")
    p.add_argument("--aggregate", action="store_true", help="rebuild the hourly CSV from raw")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()

    if a.aggregate:
        aggregate(a.symbol, a.out)
        return 0
    if a.run:
        async def go():
            t = asyncio.create_task(stream(a.symbol, a.out))
            if a.seconds:
                await asyncio.sleep(a.seconds)
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            else:
                await t
        asyncio.run(go())
        aggregate(a.symbol, a.out)
        return 0
    return status(a.symbol, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
