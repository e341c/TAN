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


def build_target_spec(
    target_mode: str,
    tb_cfg: dict[str, Any] | None = None,
    *,
    horizon: int = 1,
    threshold: float = 0.001,
    bar_hours: float = 1.0,
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
            "description": (
                f"P(price touches +{float(tb.get('upper_mult', 2.5))}*sigma before "
                f"-{float(tb.get('lower_mult', 2.5))}*sigma within {max_holding} bars)"
            ),
        }
    return {
        "mode": NEXT_BAR,
        "threshold": float(threshold),
        "horizon_bars": int(horizon),
        "horizon_hours": int(horizon) * bar_hours,
        "description": f"P(return over {horizon} bar(s) exceeds +/-{threshold})",
    }


def default_spec() -> dict[str, Any]:
    """
    Assumed target for artifacts predating the spec.

    Every run before 2026-08-16 was trained on next-bar sign, so that is the
    honest default — not the currently-configured mode.
    """
    return build_target_spec(NEXT_BAR)


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
        return build_target_spec(TRIPLE_BARRIER, {})
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
