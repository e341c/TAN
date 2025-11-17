# model.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass

# Можно переиспользовать тот же SCALE, что и в train.py при необходимости
# TARGET_SCALE = 1000.0


@dataclass
class CVResult:
    cv_mae: float
    fold_mae: list
    zero_features: list
    used_features: list


class PurgedTimeSeriesSplit:
    """
    Простая реализация Purged CV:
    - n_splits разрезает по времени
    - между train и val оставляем gap (purge_gap), равный horizon (в барах)
    """
    def __init__(self, n_splits=5, purge_gap=0):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X: pd.DataFrame):
        n_samples = len(X)
        # делим на (n_splits + 1) блок
        fold_size = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_gap
            test_end = test_start + fold_size

            if test_end > n_samples:
                break

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            yield train_idx, test_idx


def make_lgbm(cfg):
    return lgb.LGBMRegressor(
        n_estimators=cfg.get("n_estimators", 800),
        learning_rate=cfg.get("learning_rate", 0.03),
        max_depth=cfg.get("max_depth", -1),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),

        # min_gain_to_split=0.0,  # важно при маленьком таргете
        min_split_gain=1e-6,
        min_child_samples=20,
        min_child_weight=1e-3,

        objective="regression",
        random_state=42,
        n_jobs=-1,
    )


def train_model(X: pd.DataFrame, y: pd.Series, cfg: dict):
    n_splits = cfg.get("cv_splits", 5)
    horizon = int(cfg["horizon"])
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge_gap=horizon)

    fold_mae = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = make_lgbm(cfg)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        fold_mae.append(mae)
        models.append(model)

        print(f"📉 Fold {fold + 1}/{n_splits} MAE: {mae:.6f}")

    # Берём лучший фолд по MAE
    best_idx = int(np.argmin(fold_mae))
    best_model = models[best_idx]
    print(f"🏆 Best fold = {best_idx + 1}, MAE={fold_mae[best_idx]:.6f}")

    # ----- Feature Importance -----
    importances = best_model.feature_importances_
    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\n🔍 Feature importance (top 20):")
    print(fi.head(20))

    zero_features = fi[fi["importance"] == 0]["feature"].tolist()
    if zero_features:
        print(f"\n🧹 Dropping {len(zero_features)} zero-importance features:")
        print(zero_features)
        X_reduced = X.drop(columns=zero_features)
    else:
        X_reduced = X.copy()

    used_features = list(X_reduced.columns)

    # ----- Финальное дообучение на ВСЕХ данных с очищенным набором фичей -----
    final_model = make_lgbm(cfg)
    final_model.fit(X_reduced, y)

    cv_result = CVResult(
        cv_mae=float(np.mean(fold_mae)),
        fold_mae=[float(m) for m in fold_mae],
        zero_features=zero_features,
        used_features=used_features,
    )

    metrics = {
        "CV_MAE": cv_result.cv_mae,
        "fold_MAE": cv_result.fold_mae,
        "dropped_features": cv_result.zero_features,
    }

    return final_model, metrics, used_features

# # py-trainer/model.py
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error # type: ignore
# from dataset import time_split # type: ignore
# import numpy as np
# from typing import Dict, Any


# def train_model(X, y, cfg: Dict[str, Any]): # type: ignore
#     params = dict(
#         n_estimators=cfg.get("n_estimators", 500),
#         learning_rate=cfg.get("learning_rate", 0.05),
#         max_depth=cfg.get("max_depth", 6),
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective="regression",
#         random_state=42,
#     )

#     model = lgb.LGBMRegressor(**params)
#     scores = []
#     for X_tr, X_te, y_tr, y_te in time_split(X, y, n_splits=cfg.get("cv_splits", 5)): # type: ignore
#         model.fit( # type: ignore
#             X_tr, # type: ignore
#             y_tr, # type: ignore
#             eval_set=[(X_te, y_te)],
#             eval_metric="l1",
#             # early_stopping_rounds=30, # type: ignore
#             # verbose=False, # type: ignore
#             callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
#         )
#         preds = model.predict(X_te) # type: ignore
#         scores.append(mean_absolute_error(y_te, preds)) # type: ignore

#     return model, {"MAE": float(np.mean(scores))} # type: ignore
