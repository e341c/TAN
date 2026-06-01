import torch
import numpy as np


def pinball_loss(y_true, y_pred, q: float) -> float:
    """Mean pinball loss for one quantile (matches QuantileLoss per-head term)."""
    err = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.maximum((q - 1.0) * err, q * err)))


def evaluate_model(model, X, y, quantiles=(0.1, 0.5, 0.9)):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        y_pred_tensor = model(X_tensor)

    y_pred = y_pred_tensor.cpu().numpy()

    print("y shape:", y.shape)
    print("y_pred shape:", y_pred.shape)

    results = {}
    for i, q in enumerate(quantiles):
        results[f"pinball_q{q}"] = pinball_loss(y, y_pred[:, i], q)

    return results
