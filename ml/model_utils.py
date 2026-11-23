"""
BankGuard AI — Model Utilities

Discovery of saved models, metadata extraction, and the full end-to-end
training pipeline orchestrator.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_COLUMNS, MODEL_DIR, RANDOM_SEED, TARGET_COLUMN, TEST_SIZE
from ml.preprocessing import TransactionPreprocessor
from ml.train import get_best_model, save_model, train_all_models
from ml.evaluate import compare_models


# ------------------------------------------------------------------
# Model discovery
# ------------------------------------------------------------------


def get_available_models() -> List[str]:
    """List saved model files in the ``models/`` directory.

    Looks for ``.joblib`` and ``.pkl`` files.

    Returns:
        Sorted list of file paths relative to the current working
        directory, or absolute paths if ``MODEL_DIR`` is absolute.
    """
    if not os.path.isdir(MODEL_DIR):
        return []

    extensions = {".joblib", ".pkl", ".pickle"}
    model_files = sorted(
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if os.path.splitext(f)[1].lower() in extensions
    )
    return model_files


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Return metadata about a saved model file.

    Args:
        model_path: Path to the model file.

    Returns:
        A dict with keys ``path``, ``filename``, ``type`` (extension),
        ``file_size_bytes``, ``file_size_mb``, and ``modified_date``.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    stat = os.stat(model_path)
    modified_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return {
        "path": model_path,
        "filename": os.path.basename(model_path),
        "type": os.path.splitext(model_path)[1].lower(),
        "file_size_bytes": stat.st_size,
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified_date": modified_dt.isoformat(),
    }


# ------------------------------------------------------------------
# Full training pipeline
# ------------------------------------------------------------------


def run_training_pipeline(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute the complete training pipeline end-to-end.

    Steps:
      1. Validate that the DataFrame contains the target and features.
      2. Preprocess (fit_transform on training data, transform on test).
      3. Stratified train/test split (the split happens **before** fitting
         the preprocessor on training data to prevent data leakage).
      4. Train all supported models on the **training** split only.
      5. Evaluate all models on the **test** split.
      6. Select the best model (by F1-score).
      7. Save the best model and the preprocessor to ``MODEL_DIR``.

    Args:
        df:        Full labelled transaction DataFrame.
        test_size: Fraction of data to reserve for testing (default from
                   :pydata:`config.TEST_SIZE`).
        seed:      Random seed for reproducibility.

    Returns:
        A dict with keys:
          - ``models``           – dict of model_name → fitted estimator
          - ``evaluations``      – dict of model_name → evaluation metrics
          - ``best_model_name``  – name of the highest-F1 model
          - ``preprocessor``     – the fitted :class:`TransactionPreprocessor`
          - ``test_data``        – dict with ``X_test`` and ``y_test``

    Raises:
        ValueError: If *df* is empty, lacks the target column, lacks
                    feature columns, or has fewer than 2 classes.
    """
    # ----- Input validation -------------------------------------------------
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot train.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in the DataFrame."
        )

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        raise ValueError(
            "No feature columns found in the DataFrame. "
            f"Expected some of: {FEATURE_COLUMNS}"
        )

    unique_classes = df[TARGET_COLUMN].nunique()
    if unique_classes < 2:
        raise ValueError(
            f"Need at least 2 classes in '{TARGET_COLUMN}', "
            f"but found {unique_classes}."
        )

    # ----- Stratified train/test split (BEFORE preprocessing) ---------------
    y = df[TARGET_COLUMN].values
    X_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # ----- Preprocessing (fit on training data ONLY) ------------------------
    preprocessor = TransactionPreprocessor(feature_columns=available_features)
    X_train_processed = preprocessor.fit_transform(X_train_df)
    X_test_processed = preprocessor.transform(X_test_df)

    X_train = preprocessor.get_feature_matrix(X_train_processed)
    X_test = preprocessor.get_feature_matrix(X_test_processed)

    # ----- Training (on training data ONLY) ---------------------------------
    models = train_all_models(X_train, y_train, seed=seed)

    # ----- Evaluation (on test data ONLY) -----------------------------------
    evaluations = compare_models(models, X_test, y_test)

    # ----- Best model selection ---------------------------------------------
    best_model_name = get_best_model(evaluations)

    # ----- Persist best model & preprocessor --------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_model_path = os.path.join(MODEL_DIR, f"{best_model_name}.joblib")
    save_model(models[best_model_name], best_model_path)

    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    return {
        "models": models,
        "evaluations": evaluations,
        "best_model_name": best_model_name,
        "preprocessor": preprocessor,
        "test_data": {
            "X_test": X_test,
            "y_test": y_test,
        },
    }
