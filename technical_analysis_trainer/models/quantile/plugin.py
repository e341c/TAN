from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from technical_analysis_trainer.core.artifacts import ArtifactStore
from technical_analysis_trainer.core.types import ModelOutput, PredictContext
from technical_analysis_trainer.models._paths import package_models_dir
from technical_analysis_trainer.models.quantile.predict import predict_from_candles


class QuantilePlugin:
    name = "quantile"

    def __init__(self) -> None:
        self._store: ArtifactStore | None = None
        # None → take seq_len from the run's saved bundle, so a model trained
        # with a different window cannot be served with the wrong one.
        self._seq_len: int | None = None
        self._run_id: str | None = None

    def load(self, run_id: str | None = None, *, repo_root: Path | None = None) -> None:
        # Runs before 2026-08-14 used 7 raw-price features and an unnormalised
        # target; they are incompatible with the current feature pipeline.
        self._run_id = (
            (run_id or os.environ.get("QUANTILE_MODEL_RUN") or "").strip()
            or "2026-08-14T07-59-57"
        )
        self._store = ArtifactStore(package_models_dir("quantile"))
        self._store.get_run(self._run_id)

    def predict(
        self,
        candles: pd.DataFrame,
        *,
        context: PredictContext,
        drop_last: bool = True,
    ) -> ModelOutput:
        if self._store is None:
            self.load(self._run_id)
        assert self._store is not None
        payload, run = predict_from_candles(
            candles,
            self._store,
            seq_len=self._seq_len,
            run_id=self._run_id,
            drop_last=drop_last,
        )
        return ModelOutput(
            name=self.name,
            payload=payload,
            meta={"run_id": run.run_id, "onnx_path": str(run.onnx_path)},
        )
