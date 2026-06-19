import numpy as np


class VolumeModel:
    def __init__(self):
        pass


def train_volume_model(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict | None = None,
) -> tuple[None]:
    _ = VolumeModel(), X, y, cfg
    return None
