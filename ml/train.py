"""
BankGuard AI — Model Training

Functions to train, persist, and compare fraud-detection classifiers.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_SEED


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    seed: int = RANDOM_SEED,
) -> Any:
    """Train a single classifier.

    Args:
        X_train:    Feature matrix (2-D array).
        y_train:    Target vector (1-D array of 0/1).
        model_type: ``"logistic_regression"`` or ``"random_forest"``.
        seed:       Random seed for reproducibility.

    Returns:
        A fitted scikit-learn estimator.

    Raises:
        ValueError: If *model_type* is not recognised, or inputs are empty.
    """
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data must not be empty.")

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training labels must contain at least 2 classes, "
            f"but only found: {unique_classes.tolist()}"
        )

    model_type = model_type.lower().strip()

    if model_type == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Choose 'logistic_regression' or 'random_forest'."
        )

    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Train all supported classifiers and return them in a dict.

    Args:
        X_train: Feature matrix.
        y_train: Target vector.
        seed:    Random seed.

    Returns:
        ``{"logistic_regression": <model>, "random_forest": <model>}``
    """
    model_types = ["logistic_regression", "random_forest"]
    return {
        name: train_model(X_train, y_train, model_type=name, seed=seed)
        for name in model_types
    }


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


def save_model(model: Any, path: str) -> None:
    """Serialise a trained model to disk with :mod:`joblib`.

    Args:
        model: A fitted scikit-learn estimator.
        path:  Destination file path (e.g. ``"models/random_forest.joblib"``).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    """Load a previously saved model.

    Args:
        path: Path to the serialised model file.

    Returns:
        The deserialised scikit-learn estimator.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


# ------------------------------------------------------------------
# Model selection
# ------------------------------------------------------------------


def get_best_model(evaluation_results: Dict[str, Dict[str, Any]]) -> str:
    """Return the name of the model with the highest F1-score.

    Args:
        evaluation_results: Mapping of model names to evaluation dicts,
                            each of which must contain an ``"f1"`` key.

    Returns:
        The name (key) of the best-performing model.

    Raises:
        ValueError: If *evaluation_results* is empty.
    """
    if not evaluation_results:
        raise ValueError("evaluation_results must not be empty.")

    return max(evaluation_results, key=lambda name: evaluation_results[name]["f1"])
