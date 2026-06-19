from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return repo_root() / path


@dataclass
class ModelRun:
    run_id: str
    run_dir: Path
    onnx_path: Path
    scaler_path: Path


class ArtifactStore:
    """Resolve ONNX/scaler paths under models/{package}/models/{run_id}/."""

    def __init__(self, models_root: str | Path):
        self.models_root = Path(models_root)

    def latest_run_id(self) -> str:
        best_t: float = -1.0
        best_name: str | None = None
        if not self.models_root.is_dir():
            raise FileNotFoundError(f"Models directory not found: {self.models_root}")
        for name in os.listdir(self.models_root):
            sub = self.models_root / name
            if not sub.is_dir():
                continue
            scaler = self.find_scaler_path(name)
            if scaler is None:
                continue
            mtime = scaler.stat().st_mtime
            if mtime > best_t:
                best_t = mtime
                best_name = name
        if best_name is None:
            raise FileNotFoundError(f"No scaler_*.pkl or scaler-*.pkl in {self.models_root}")
        return best_name

    def find_scaler_path(self, run_id: str) -> Path | None:
        run_dir = self.models_root / run_id
        for name in (f"scaler_{run_id}.pkl", f"scaler-{run_id}.pkl"):
            p = run_dir / name
            if p.is_file():
                return p
        return None

    def get_run(self, run_id: str | None = None) -> ModelRun:
        rid = run_id or self.latest_run_id()
        run_dir = self.models_root / rid
        scaler_path = self.find_scaler_path(rid)
        if scaler_path is None:
            raise FileNotFoundError(f"Scaler not found for run {rid} in {run_dir}")
        onnx_path = run_dir / f"model_{rid}.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        return ModelRun(run_id=rid, run_dir=run_dir, onnx_path=onnx_path, scaler_path=scaler_path)

    def load_scaler_raw(self, run_id: str | None = None):
        run = self.get_run(run_id)
        with open(run.scaler_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "scaler" in data:
            return data["scaler"], run
        return data, run

    def load_trend_scaler_bundle(
        self, run_id: str | None = None
    ) -> tuple[object, list[str], float, float, ModelRun]:
        run = self.get_run(run_id)
        with open(run.scaler_path, "rb") as f:
            scaler_data = pickle.load(f)
        if not isinstance(scaler_data, dict):
            raise ValueError(
                "Trend scaler must be a dict from train.py: "
                "{scaler, features, scale_clip_min, scale_clip_max}"
            )
        scaler = scaler_data["scaler"]
        features = list(scaler_data["features"])
        lo = float(scaler_data.get("scale_clip_min", -5.0))
        hi = float(scaler_data.get("scale_clip_max", 5.0))
        return scaler, features, lo, hi, run
