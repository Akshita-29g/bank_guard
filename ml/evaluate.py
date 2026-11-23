"""
BankGuard AI — Model Evaluation

Metrics computation, model comparison, and feature-importance extraction.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute a comprehensive set of evaluation metrics.

    Args:
        model:  A fitted scikit-learn estimator with ``predict`` and
                ``predict_proba`` methods.
        X_test: Feature matrix for the test split.
        y_test: True labels for the test split.

    Returns:
        A dict containing:
          - ``precision``, ``recall``, ``f1``  (float, macro-averaged on the
            minority class / binary)
          - ``roc_auc`` (float; set to 0.0 if only one class is present)
          - ``pr_auc``  (float; Precision–Recall AUC, same caveat)
          - ``confusion_matrix`` (2-D list)
          - ``classification_report`` (formatted string)

    Raises:
        ValueError: If inputs are empty.
    """
    if X_test.size == 0 or y_test.size == 0:
        raise ValueError("Test data must not be empty.")

    y_pred = model.predict(X_test)

    # Binary metrics — class 1 (fraud) is the positive class.
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Probability-based metrics
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    # ROC-AUC and PR-AUC require both classes in y_test
    n_classes = len(np.unique(y_test))
    if n_classes >= 2:
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate every model in *models* and return a comparison dict.

    Args:
        models: Mapping of model name → fitted estimator.
        X_test: Feature matrix for the test split.
        y_test: True labels for the test split.

    Returns:
        ``{model_name: evaluation_dict, ...}`` where each
        ``evaluation_dict`` is as returned by :func:`evaluate_model`.
    """
    return {
        name: evaluate_model(model, X_test, y_test)
        for name, model in models.items()
    }


def get_feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract feature importances (RF) or absolute coefficients (LR).

    Args:
        model:         A fitted scikit-learn estimator.
        feature_names: Ordered list of feature column names matching the
                       training matrix.

    Returns:
        A :class:`~pandas.DataFrame` with columns ``feature`` and
        ``importance``, sorted by descending importance.

    Raises:
        ValueError: If the model exposes neither ``feature_importances_``
                    nor ``coef_``.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For logistic regression, use absolute coefficient values
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError(
            "The model does not expose feature_importances_ or coef_. "
            "Cannot extract feature importance."
        )

    # Ensure lengths match (truncate or pad if necessary)
    n = min(len(feature_names), len(importances))
    df = pd.DataFrame(
        {
            "feature": feature_names[:n],
            "importance": importances[:n],
        }
    )
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df
