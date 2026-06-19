from __future__ import annotations

from technical_analysis_trainer.models.base import ModelPlugin
from technical_analysis_trainer.models.quantile.plugin import QuantilePlugin
from technical_analysis_trainer.models.trend.plugin import TrendPlugin

REGISTRY: dict[str, type[ModelPlugin]] = {
    "quantile": QuantilePlugin,
    "trend": TrendPlugin,
}


def get_plugin(name: str) -> ModelPlugin:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model plugin: {name}")
    return REGISTRY[name]()
