"""
Trend target semantics — what question a given trend model actually answers.

`predicted_trend: "DOWN"` is the same string under both targets, but it means
very different things:

  next_bar        the next bar's return is below -threshold        (1 bar)
  triple_barrier  price touches -k*sigma before +k*sigma           (up to N bars)

At current BTC volatility that is a 0.1% move within an hour versus a ~0.35%
move within a day — a 3.5x threshold over a 24x horizon, behind an identical
payload. Anything acting on the signal (the trading bot's entry rules, the
accuracy panel, the UI badge) needs the spec, not just the label.

The spec is written into the model artifact at training time and read back at
inference, so a config edit can never silently reinterpret a deployed model.
"""
from __future__ import annotations

import json
from typing import Any

NEXT_BAR = "next_bar"
TRIPLE_BARRIER = "triple_barrier"


# Where the barrier is anchored, relative to the window the model sees.
#
#   LAST_OBSERVED  barriers start at the close of the last bar IN the window.
#                  The model can see the price its barriers are measured from,
#                  and the bot can place that level the moment it gets a signal.
#   NEXT_CLOSE     barriers start at the close of the bar AFTER the window
#                  (the original convention). The anchor price is unobserved:
#                  measured on BTC 1h in Aug 2026, one bar of noise is 43.9% of
#                  the 2.5-sigma barrier half-width, so nearly half the target's
#                  variance was unpredictable by construction. At serving time
#                  that bar has not closed yet, so the level the bot places is
#                  not the level the label used.
ANCHOR_LAST_OBSERVED = "last_observed"
ANCHOR_NEXT_CLOSE = "next_close"


def build_target_spec(
    target_mode: str,
    tb_cfg: dict[str, Any] | None = None,
    *,
    horizon: int = 1,
    threshold: float = 0.001,
    bar_hours: float = 1.0,
    anchor: str = ANCHOR_LAST_OBSERVED,
) -> dict[str, Any]:
    """Serialisable description of what the model predicts."""
    tb = tb_cfg or {}
    if target_mode == TRIPLE_BARRIER:
        max_holding = int(tb.get("max_holding", 24))
        return {
            "mode": TRIPLE_BARRIER,
            "upper_mult": float(tb.get("upper_mult", 2.5)),
            "lower_mult": float(tb.get("lower_mult", 2.5)),
            "max_holding": max_holding,
            "vol_window": int(tb.get("vol_window", 72)),
            "horizon_bars": max_holding,
            "horizon_hours": max_holding * bar_hours,
            "anchor": anchor,
            "description": (
                f"P(price touches +{float(tb.get('upper_mult', 2.5))}*sigma before "
                f"-{float(tb.get('lower_mult', 2.5))}*sigma within {max_holding} bars, "
                f"from the {'last observed close' if anchor == ANCHOR_LAST_OBSERVED else 'next close'})"
            ),
        }
    return {
        "mode": NEXT_BAR,
        "threshold": float(threshold),
        "horizon_bars": int(horizon),
        "horizon_hours": int(horizon) * bar_hours,
        "anchor": anchor,
        "description": f"P(return over {horizon} bar(s) exceeds +/-{threshold})",
    }


def spec_anchor(spec: dict[str, Any] | None) -> str:
    """
    Anchor convention for a spec, defaulting to the legacy one.

    An artifact exported before the anchor was recorded was trained with the
    barrier on the *next* close, so a missing key must read as NEXT_CLOSE.
    Defaulting the other way would silently re-index every window a bar late.
    """
    return str((spec or {}).get("anchor") or ANCHOR_NEXT_CLOSE)


def default_spec() -> dict[str, Any]:
    """
    Assumed target for artifacts predating the spec.

    Every run before 2026-08-16 was trained on next-bar sign, so that is the
    honest default — not the currently-configured mode.
    """
    return build_target_spec(NEXT_BAR, anchor=ANCHOR_NEXT_CLOSE)


def spec_from_bundle(bundle: dict[str, Any] | None) -> dict[str, Any]:
    """Read the target spec out of a trend scaler bundle, tolerating old ones."""
    if not isinstance(bundle, dict):
        return default_spec()
    spec = bundle.get("target_spec")
    if isinstance(spec, dict) and spec.get("mode"):
        return spec
    mode = bundle.get("target_mode")
    if mode == TRIPLE_BARRIER:
        # Bundle knows the mode but predates the full spec: fall back to the
        # trained defaults rather than to next_bar, which would be wrong.
        return build_target_spec(TRIPLE_BARRIER, {}, anchor=ANCHOR_NEXT_CLOSE)
    return default_spec()


def spec_to_json(spec: dict[str, Any] | None) -> str | None:
    return json.dumps(spec, ensure_ascii=False) if spec else None


def spec_from_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            out = json.loads(raw)
            if isinstance(out, dict) and out.get("mode"):
                return out
        except (ValueError, TypeError):
            pass
    return default_spec()


def horizon_label(spec: dict[str, Any] | None) -> str:
    """Short human-readable horizon, e.g. '24h barrier +/-2.5s' or '1h'."""
    s = spec or default_spec()
    if s.get("mode") == TRIPLE_BARRIER:
        return f"{int(s.get('horizon_hours', 24))}h barrier ±{s.get('upper_mult', 2.5)}σ"
    return f"{int(s.get('horizon_hours', 1))}h"
