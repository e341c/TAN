from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from technical_analysis_trainer.core.artifacts import ArtifactStore, resolve_path
from technical_analysis_trainer.core.types import ModelOutput, PredictContext
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.trend.predict import build_trend_x_window, run_trend_onnx


def default_trend_config_path() -> Path:
    env = os.environ.get("TREND_REALTIME_CONFIG", "").strip()
    if env:
        return resolve_path(env)
    return Path(__file__).resolve().parent / "test_config.yaml"


def load_trend_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = resolve_path(path) if path else default_trend_config_path()
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TrendPlugin:
    name = "trend"

    def __init__(self) -> None:
        self._store: ArtifactStore | None = None
        self._run_id: str | None = None
        self._cfg: dict[str, Any] | None = None

    def load(self, run_id: str | None = None, *, repo_root: Path | None = None) -> None:
        self._run_id = run_id or os.environ.get("TREND_MODEL_RUN", "").strip() or None
        self._store = ArtifactStore(package_models_dir("trend"))
        if self._run_id:
            self._store.get_run(self._run_id)

    def predict(self, candles: pd.DataFrame, *, context: PredictContext) -> ModelOutput:
        if self._store is None:
            self.load(self._run_id)
        assert self._store is not None
        cfg = context.trend_config or load_trend_config(context.trend_config_path)
        x_window, meta = build_trend_x_window(cfg, self._store, run_id=self._run_id)
        payload = run_trend_onnx(
            x_window, str(meta["onnx_path"]), target_spec=meta.get("target_spec")
        )
        return ModelOutput(name=self.name, payload=payload, meta=meta)
