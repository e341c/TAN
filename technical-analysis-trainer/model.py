# py-trainer/model.py
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error # type: ignore
from dataset import time_split # type: ignore
import numpy as np
from typing import Dict, Any


def train_model(X, y, cfg: Dict[str, Any]): # type: ignore
    params = dict(
        n_estimators=cfg.get("n_estimators", 500),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", 6),
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
    )

    model = lgb.LGBMRegressor(**params)
    scores = []
    for X_tr, X_te, y_tr, y_te in time_split(X, y, n_splits=cfg.get("cv_splits", 5)): # type: ignore
        model.fit( # type: ignore
            X_tr, # type: ignore
            y_tr, # type: ignore
            eval_set=[(X_te, y_te)],
            eval_metric="l1",
            # early_stopping_rounds=30, # type: ignore
            # verbose=False, # type: ignore
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_te) # type: ignore
        scores.append(mean_absolute_error(y_te, preds)) # type: ignore

    return model, {"MAE": float(np.mean(scores))} # type: ignore
