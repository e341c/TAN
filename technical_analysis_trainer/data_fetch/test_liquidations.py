"""
Tests for the liquidation collector. Offline — no network.

Run: python -m technical_analysis_trainer.data_fetch.test_liquidations

These matter more than usual because the collector could not be verified against
a live stream: fstream.binance.com accepts a websocket connection from the
development machine and then sends nothing at all, for any stream, while spot and
REST both work. So the wire format is taken from Binance's documented schema and
pinned here, and the two pieces of logic that are silently invertible are
asserted rather than eyeballed:

  * SIDE. `S` is the side of the liquidating ORDER, not of the position that
    died — a forced SELL closes a LONG. Inverting this flips the sign of the
    eventual feature while leaving every magnitude plausible, so it would survive
    any amount of staring at the CSV.

  * ZERO vs MISSING. An hour with coverage and no liquidations is a real zero; an
    hour with no coverage is a hole. If those ever collapse into each other the
    feature becomes "liquidations, except when the collector was down, where it
    is confidently zero" — which is worse than not having the feature.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from technical_analysis_trainer.data_fetch.binance_liquidations import (
    _append,
    _paths,
    aggregate,
    parse_event,
)

# A real forceOrder payload, shaped as Binance documents it.
def _event(side: str, qty: str, price: str, ms: int, symbol: str = "BTCUSDT") -> dict:
    return {"e": "forceOrder", "E": ms,
            "o": {"s": symbol, "S": side, "o": "LIMIT", "f": "IOC", "q": qty, "p": price,
                  "ap": price, "X": "FILLED", "l": qty, "z": qty, "T": ms}}


def test_forced_sell_is_a_long_being_liquidated() -> None:
    """The sign convention. Inverting it is invisible in the data."""
    row = parse_event(_event("SELL", "0.5", "60000", 1788000000000))
    assert row is not None
    assert row["position_closed"] == "long", (
        f"a forced SELL closes a LONG position, got {row['position_closed']!r}"
    )
    row = parse_event(_event("BUY", "0.5", "60000", 1788000000000))
    assert row["position_closed"] == "short", (
        f"a forced BUY closes a SHORT position, got {row['position_closed']!r}"
    )


def test_notional_and_timestamp() -> None:
    row = parse_event(_event("SELL", "0.25", "60000", 1788000000000))
    assert row["quote"] == 15000.0, f"quote should be price*qty, got {row['quote']}"
    assert row["ts"].startswith("2026-"), f"unexpected ts {row['ts']}"
    assert pd.Timestamp(row["ts"]).tz is not None, "timestamps must be tz-aware UTC"


def test_non_liquidation_messages_are_ignored() -> None:
    """The socket also carries the markPrice canary; it must never become a row."""
    assert parse_event({"e": "markPriceUpdate", "s": "BTCUSDT", "p": "60000"}) is None
    assert parse_event({"e": "aggTrade", "s": "BTCUSDT"}) is None
    assert parse_event({}) is None


def test_covered_hour_with_no_events_is_a_real_zero() -> None:
    with tempfile.TemporaryDirectory() as d:
        raw, hb, out = _paths("BTCUSDT", d)
        # Two covered hours; liquidations in the first only.
        _append(hb, ["ts"], [{"ts": "2026-09-01T00:10:00+00:00"},
                             {"ts": "2026-09-01T01:10:00+00:00"}])
        _append(raw, list(parse_event(_event("SELL", "1", "60000", 1)).keys()),
                [parse_event(_event("SELL", "1", "60000", 1788222000000))])
        df = aggregate("BTCUSDT", d)
        assert len(df) == 2, f"expected both covered hours, got {len(df)}"
        assert df["liq_long_quote"].isna().sum() == 0, "a covered hour must not be NaN"
        assert (df["liq_total_quote"] == 0).sum() == 1, (
            "the covered hour with no liquidations should be exactly zero"
        )
        assert os.path.isfile(out)


def test_uncovered_hours_are_absent_not_zero() -> None:
    """The hole must stay a hole. A gap in coverage must not appear as a quiet hour."""
    with tempfile.TemporaryDirectory() as d:
        _, hb, _ = _paths("BTCUSDT", d)
        # 00:00 and 05:00 covered; 01:00-04:00 the collector was down.
        _append(hb, ["ts"], [{"ts": "2026-09-01T00:10:00+00:00"},
                             {"ts": "2026-09-01T05:10:00+00:00"}])
        df = aggregate("BTCUSDT", d)
        hours = pd.to_datetime(df["ts"], utc=True).dt.hour.tolist()
        assert hours == [0, 5], f"uncovered hours leaked in as zeros: {hours}"


def test_side_field_is_preserved_for_audit() -> None:
    """Keep the raw side too, so the derived column can be re-checked later."""
    row = parse_event(_event("SELL", "1", "60000", 1788000000000))
    assert row["side"] == "SELL"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
