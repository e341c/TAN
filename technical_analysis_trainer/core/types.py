from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PredictContext:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    trend_config: dict[str, Any] | None = None
    trend_config_path: str | None = None


@dataclass
class QuantilePayload:
    ts: Any
    q10: float
    q50: float
    q90: float
    log_returns: list[float] | None = None
    last_close: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "q10": self.q10,
            "q50": self.q50,
            "q90": self.q90,
            "log_returns": self.log_returns,
            "last_close": self.last_close,
        }


@dataclass
class TrendPayload:
    predicted_trend: str
    probabilities: dict[str, float]
    logits: list[float]
    run_id: str | None = None
    display: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "predicted_trend": self.predicted_trend,
            "probabilities": self.probabilities,
            "logits": self.logits,
        }
        out.update(self.display)
        return out


@dataclass
class ModelOutput:
    name: str
    payload: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionSnapshot:
    candles: pd.DataFrame
    outputs: dict[str, ModelOutput]
    symbol: str = "BTCUSDT"
    interval: str = "1h"

    def quantile_dict(self) -> dict[str, Any] | None:
        out = self.outputs.get("quantile")
        return out.payload if out else None

    def trend_dict(self) -> dict[str, Any] | None:
        out = self.outputs.get("trend")
        return out.payload if out else None

    def trend_meta(self) -> dict[str, Any]:
        out = self.outputs.get("trend")
        return out.meta if out else {}
