from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from technical_analysis_trainer.core.types import ModelOutput, PredictContext


@runtime_checkable
class ModelPlugin(Protocol):
    name: str

    def load(self, run_id: str | None = None, *, repo_root: Path | None = None) -> None: ...

    def predict(self, candles: pd.DataFrame, *, context: PredictContext) -> ModelOutput: ...
