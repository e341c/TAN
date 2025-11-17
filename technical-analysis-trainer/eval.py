# py-trainer/eval.py
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score # type: ignore

def evaluate_model(model, X, y): # type: ignore
    preds = model.predict(X) # type: ignore
    mae = mean_absolute_error(y, preds) # type: ignore
    r2 = r2_score(y, preds) # type: ignore
    ic = np.corrcoef(preds, y)[0,1]  # type: ignore # Information Coefficient
    return {"MAE": mae, "R2": r2, "IC": ic} # type: ignore