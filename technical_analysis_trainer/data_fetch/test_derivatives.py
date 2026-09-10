"""
Tests for the derivatives collector's merge. Offline — no network.

Run: python -m technical_analysis_trainer.data_fetch.test_derivatives

The collected series cannot be re-fetched: Binance keeps 30 days and rejects any
`startTime` older than that, so anything this file loses is lost permanently.
The HTTP layer is not what needs testing — a failed request is loud and the next
run repairs it. The merge is the dangerous part, because every way it can go
wrong is silent:

  * dropping a timestamp that was on disk,
  * dropping a column that was on disk when a later run fetches fewer series,
  * letting a NaN from a partial fetch overwrite a real stored value.

The third is the subtle one. These tests were written believing that a `concat`
followed by `groupby("ts").last()` got it wrong; it does not. `GroupBy.last()`
skips NaN by default, so the stored value survives a failed endpoint without any
help. Running the suite against that implementation passes all six.

That is recorded rather than quietly deleted, because it changes what these
tests are for. They are not defending against a bug that was there — they pin
behaviour that currently holds by way of an easily-missed pandas default, and
which `.nth(-1)` or `.tail(1)` would silently break. That makes them a guard on
a future edit, not evidence of a past fix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from technical_analysis_trainer.data_fetch.binance_derivatives import merge_history


def _frame(hours, **cols):
    ts = pd.to_datetime(["2026-09-01T00:00:00Z"], utc=True)[0] + pd.to_timedelta(hours, unit="h")
    return pd.DataFrame({"ts": ts, **cols})


def test_merge_is_append_only() -> None:
    old = _frame([0, 1, 2], open_interest=[10.0, 11.0, 12.0])
    new = _frame([3, 4], open_interest=[13.0, 14.0])
    out = merge_history(old, new)
    assert len(out) == 5, f"expected 5 rows, got {len(out)}"
    assert out["ts"].is_monotonic_increasing
    assert list(out["open_interest"]) == [10, 11, 12, 13, 14]


def test_merge_is_idempotent() -> None:
    old = _frame([0, 1, 2], open_interest=[10.0, 11.0, 12.0])
    once = merge_history(old, old.copy())
    twice = merge_history(once, old.copy())
    assert len(once) == 3 and len(twice) == 3, "re-merging the same frame added rows"
    assert once.equals(twice)


def test_new_values_correct_provisional_ones() -> None:
    """The latest bars can be revised, so an overlapping re-fetch must win."""
    old = _frame([0, 1], open_interest=[10.0, 99.0])
    new = _frame([1, 2], open_interest=[11.0, 12.0])
    out = merge_history(old, new)
    assert list(out["open_interest"]) == [10, 11, 12], (
        f"a re-fetch did not correct the provisional value: {list(out['open_interest'])}"
    )


def test_a_partial_fetch_cannot_blank_stored_values() -> None:
    """
    The failure this function exists for.

    One endpoint returns nothing while the others succeed. The merged frame has
    that column as NaN over the overlap. A `groupby.last()` would take the new
    row wholesale and erase 30 days of a series that cannot be re-fetched.
    """
    old = _frame([0, 1], open_interest=[10.0, 11.0], basis=[-1.0, -2.0])
    new = _frame([1, 2], open_interest=[np.nan, np.nan], basis=[-2.5, -3.0])
    out = merge_history(old, new)
    oi = out.set_index("ts")["open_interest"]
    assert oi.iloc[0] == 10.0 and oi.iloc[1] == 11.0, (
        f"a NaN from a failed endpoint overwrote stored data: {list(oi)}"
    )
    assert np.isnan(oi.iloc[2]), "the genuinely-missing new row should stay NaN"
    assert list(out["basis"]) == [-1.0, -2.5, -3.0], "the succeeding series should update"


def test_columns_are_never_dropped() -> None:
    """Adding or losing a series later must not erase what was already collected."""
    old = _frame([0, 1], open_interest=[10.0, 11.0], retired_series=[1.0, 2.0])
    new = _frame([1, 2], open_interest=[11.0, 12.0], brand_new_series=[7.0, 8.0])
    out = merge_history(old, new)
    for col in ("open_interest", "retired_series", "brand_new_series"):
        assert col in out.columns, f"{col} was dropped by the merge"
    assert list(out["retired_series"].iloc[:2]) == [1.0, 2.0], "old series lost its values"


def test_empty_inputs_are_handled() -> None:
    old = _frame([0, 1], open_interest=[10.0, 11.0])
    assert len(merge_history(pd.DataFrame(), old)) == 2
    assert len(merge_history(old, pd.DataFrame())) == 2


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
