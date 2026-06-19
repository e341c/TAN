import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Оценка бинарного классификатора TrendLSTM.

    Returns:
        dict с метриками: accuracy, f1, precision, recall, roc_auc,
                          confusion_matrix, classification_report
    """
    if class_names is None:
        class_names = ["DOWN", "UP"]

    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred = probs.argmax(axis=1)
    y_prob_up = probs[:, 1]  # вероятность класса UP для ROC-AUC

    print("y shape:     ", y.shape)
    print("y_pred shape:", y_pred.shape)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="binary")
    precision = precision_score(y, y_pred, average="binary", zero_division=0)
    recall = recall_score(y, y_pred, average="binary", zero_division=0)
    roc_auc = roc_auc_score(y, y_prob_up)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names)

    print("\n── Evaluation Results ──────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")
    print("────────────────────────────────────────────\n")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }