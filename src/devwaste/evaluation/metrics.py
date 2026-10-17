"""Evaluation metrics for anomaly detection."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calculate binary detection metrics (positive class = anomaly).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        scores: Continuous anomaly scores (optional, for ROC-AUC)

    Returns:
        Dictionary with precision, recall, F1, false-positive rate,
        true-positive rate, confusion matrix ``[[tn, fp], [fn, tp]]``,
        sample count, and ROC-AUC (None when scores are missing or only
        one class is present).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "false_positive_rate": 0.0,
            "true_positive_rate": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "n_samples": 0,
            "auc": None,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    auc = None
    if scores is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, scores))

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "n_samples": int(len(y_true)),
        "auc": auc,
    }
