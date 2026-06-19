from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from technical_analysis_trainer.core.artifacts import resolve_path
from technical_analysis_trainer.core.types import ModelOutput, PredictContext, PredictionSnapshot
from technical_analysis_trainer.models.registry import get_plugin
from technical_analysis_trainer.models.trend.plugin import load_trend_config
from technical_analysis_trainer.runtime.binance import fetch_last_candles


def _assembly_config_path() -> Path:
    env = os.environ.get("ASSEMBLY_CONFIG", "").strip()
    if env:
        return resolve_path(env)
    return Path(__file__).resolve().parent / "config.yaml"


def load_assembly_config() -> dict[str, Any]:
    with open(_assembly_config_path(), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _enabled_models(cfg: dict[str, Any]) -> list[str]:
    return list(cfg.get("enabled_models") or ["quantile", "trend"])


def _run_id_for(model_name: str, cfg: dict[str, Any]) -> str | None:
    defaults = cfg.get("defaults") or {}
    if model_name == "quantile":
        return os.environ.get("QUANTILE_MODEL_RUN") or defaults.get("quantile_run")
    if model_name == "trend":
        env = os.environ.get("TREND_MODEL_RUN", "").strip()
        return env or defaults.get("trend_run")
    return None


class PredictionOrchestrator:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_assembly_config()
        self._plugins: dict[str, Any] = {}

    def _ensure_loaded(self, name: str) -> None:
        if name in self._plugins:
            return
        plugin = get_plugin(name)
        plugin.load(_run_id_for(name, self.config))
        self._plugins[name] = plugin

    def predict_sync(
        self,
        candles: pd.DataFrame,
        *,
        symbol: str | None = None,
        interval: str | None = None,
    ) -> PredictionSnapshot:
        symbol = symbol or str(self.config.get("symbol", "BTCUSDT"))
        interval = interval or str(self.config.get("interval", "1h"))
        trend_cfg = load_trend_config(self.config.get("trend_config_path"))
        ctx = PredictContext(
            symbol=symbol,
            interval=interval,
            trend_config=trend_cfg,
        )
        outputs: dict[str, ModelOutput] = {}
        for name in _enabled_models(self.config):
            self._ensure_loaded(name)
            outputs[name] = self._plugins[name].predict(candles, context=ctx)
        return PredictionSnapshot(candles=candles, outputs=outputs, symbol=symbol, interval=interval)


_orchestrator: PredictionOrchestrator | None = None


def get_orchestrator() -> PredictionOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PredictionOrchestrator()
    return _orchestrator


async def build_snapshot(
    symbol: str | None = None,
    interval: str | None = None,
    limit: int | None = None,
    candles_df: pd.DataFrame | None = None,
) -> PredictionSnapshot:
    cfg = load_assembly_config()
    symbol = symbol or str(cfg.get("symbol", "BTCUSDT"))
    interval = interval or str(cfg.get("interval", "1h"))
    if limit is None:
        limit = int(cfg.get("binance_limit", 45))

    if candles_df is None:
        candles_df = await fetch_last_candles(symbol, interval, limit)
    if candles_df is None or len(candles_df) == 0:
        raise ValueError("Empty or missing candle dataframe")

    orch = get_orchestrator()
    return await asyncio.to_thread(
        orch.predict_sync,
        candles_df,
        symbol=symbol,
        interval=interval,
    )
