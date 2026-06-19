"""Resolve model artifact directories."""

from pathlib import Path

from technical_analysis_trainer.core.artifacts import repo_root


def package_models_dir(package: str) -> Path:
    return repo_root() / "technical_analysis_trainer" / "models" / package / "models"
