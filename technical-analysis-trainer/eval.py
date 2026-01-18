import torch
import numpy as np
from sklearn.metrics import mean_squared_error

# def evaluate_model(model, X, y):
#     model.eval()  # переводим в eval режим
#     X_tensor = torch.tensor(X, dtype=torch.float32)

#     with torch.no_grad():
#         y_pred_tensor = model(X_tensor)

#     y_pred = y_pred_tensor.numpy()

#     # вычисляем MSE для примера
#     mse = mean_squared_error(y, y_pred)

#     return {"mse": mse}


def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


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
