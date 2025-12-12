import torch
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X, y):
    model.eval()  # переводим в eval режим
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad():
        y_pred_tensor = model(X_tensor)
    
    y_pred = y_pred_tensor.numpy()
    
    # вычисляем MSE для примера
    mse = mean_squared_error(y, y_pred)
    
    return {"mse": mse}