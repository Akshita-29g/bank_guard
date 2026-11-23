"""
BankGuard AI — Dataset & Prediction-Input Validation

Functions that sanity-check DataFrames before they are fed into the ML pipeline
or the analytics dashboard.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

import pandas as pd
import numpy as np

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_COLUMNS, TARGET_COLUMN


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate a training / evaluation dataset.

    Checks performed:
      1. DataFrame is not empty.
      2. ``Amount`` column is present.
      3. At least one ``V*`` feature column is present.
      4. ``Class`` target column is present (warning if missing).
      5. No column is entirely null.

    Args:
        df: The DataFrame to validate.

    Returns:
        A dict with keys ``valid``, ``errors``, ``warnings``,
        ``row_count``, ``column_count``, ``has_target``, and
        ``has_features``.
    """
    errors: List[str] = []
    warnings: List[str] = []

    row_count: int = len(df)
    column_count: int = len(df.columns)

    # 1. Empty check
    if row_count == 0:
        errors.append("Dataset is empty (0 rows).")

    # 2. Amount column
    if "Amount" not in df.columns:
        errors.append("Required column 'Amount' is missing.")

    # 3. V-feature columns
    v_columns = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    has_features: bool = len(v_columns) > 0
    if not has_features:
        errors.append(
            "No V-feature columns (V1–V28) found. "
            "At least one is required for model training."
        )
    elif len(v_columns) < 28:
        warnings.append(
            f"Only {len(v_columns)} of 28 expected V-feature columns found."
        )

    # 4. Target column
    has_target: bool = TARGET_COLUMN in df.columns
    if not has_target:
        warnings.append(
            f"Target column '{TARGET_COLUMN}' is missing. "
            "The dataset can still be used for prediction but not for training."
        )

    # 5. All-null columns
    if row_count > 0:
        all_null_cols = [
            col for col in df.columns if df[col].isna().all()
        ]
        if all_null_cols:
            warnings.append(
                f"The following columns are entirely null: {all_null_cols}"
            )

    valid: bool = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "row_count": row_count,
        "column_count": column_count,
        "has_target": has_target,
        "has_features": has_features,
    }


def validate_prediction_input(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate a DataFrame intended for inference (prediction only).

    Similar to :func:`validate_dataset` but does **not** require the
    ``Class`` column.

    Args:
        df: The DataFrame to validate.

    Returns:
        A dict with keys ``valid``, ``errors``, ``warnings``,
        ``row_count``, ``column_count``, ``has_target``, and
        ``has_features``.
    """
    errors: List[str] = []
    warnings: List[str] = []

    row_count: int = len(df)
    column_count: int = len(df.columns)

    if row_count == 0:
        errors.append("Input data is empty (0 rows).")

    if "Amount" not in df.columns:
        errors.append("Required column 'Amount' is missing.")

    v_columns = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    has_features: bool = len(v_columns) > 0
    if not has_features:
        errors.append(
            "No V-feature columns (V1–V28) found. "
            "At least one is required for prediction."
        )

    has_target: bool = TARGET_COLUMN in df.columns

    # All-null columns
    if row_count > 0:
        all_null_cols = [
            col for col in df.columns if df[col].isna().all()
        ]
        if all_null_cols:
            warnings.append(
                f"The following columns are entirely null: {all_null_cols}"
            )

    valid: bool = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "row_count": row_count,
        "column_count": column_count,
        "has_target": has_target,
        "has_features": has_features,
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for a transaction dataset.

    Args:
        df: The DataFrame to summarise.

    Returns:
        A dict containing:
          - ``row_count`` / ``column_count``
          - ``numeric_columns`` / ``non_numeric_columns``
          - ``missing_values`` – total missing cells
          - ``missing_by_column`` – per-column missing counts (non-zero only)
          - ``fraud_count`` / ``legitimate_count`` / ``fraud_ratio`` (if Class exists)
          - ``amount_stats`` – dict with min, max, mean, median, std (if Amount exists)
          - ``feature_stats`` – per-V-column mean & std (if V-columns exist)
          - ``dtypes`` – column-name → dtype string mapping
    """
    summary: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    # Column type breakdown
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary["numeric_columns"] = numeric_cols
    summary["non_numeric_columns"] = [
        c for c in df.columns if c not in numeric_cols
    ]

    # Missing values
    missing_series = df.isna().sum()
    summary["missing_values"] = int(missing_series.sum())
    summary["missing_by_column"] = {
        col: int(cnt)
        for col, cnt in missing_series.items()
        if cnt > 0
    }

    # Fraud / legitimate breakdown
    if TARGET_COLUMN in df.columns:
        fraud_count = int((df[TARGET_COLUMN] == 1).sum())
        legit_count = int((df[TARGET_COLUMN] == 0).sum())
        total = fraud_count + legit_count
        summary["fraud_count"] = fraud_count
        summary["legitimate_count"] = legit_count
        summary["fraud_ratio"] = fraud_count / total if total > 0 else 0.0
    else:
        summary["fraud_count"] = None
        summary["legitimate_count"] = None
        summary["fraud_ratio"] = None

    # Amount statistics
    if "Amount" in df.columns:
        amt = df["Amount"].dropna()
        summary["amount_stats"] = {
            "min": float(amt.min()) if len(amt) else 0.0,
            "max": float(amt.max()) if len(amt) else 0.0,
            "mean": float(amt.mean()) if len(amt) else 0.0,
            "median": float(amt.median()) if len(amt) else 0.0,
            "std": float(amt.std()) if len(amt) else 0.0,
        }
    else:
        summary["amount_stats"] = None

    # V-feature statistics
    v_cols = sorted(
        [c for c in df.columns if c.startswith("V") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    if v_cols:
        feature_stats: Dict[str, Dict[str, float]] = {}
        for vc in v_cols:
            col_data = df[vc].dropna()
            feature_stats[vc] = {
                "mean": float(col_data.mean()) if len(col_data) else 0.0,
                "std": float(col_data.std()) if len(col_data) else 0.0,
            }
        summary["feature_stats"] = feature_stats
    else:
        summary["feature_stats"] = None

    # Data types
    summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return summary
