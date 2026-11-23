"""
BankGuard AI — Data Loading & Synthetic Data Generation

Provides helpers to:
  * Load transaction CSVs from disk or in-memory buffers.
  * Generate realistic synthetic credit-card transaction data for demos.
  * Merge model predictions into a display-ready DataFrame.
"""

from __future__ import annotations

import io
import sys
import os
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_COLUMNS, RANDOM_SEED, RISK_THRESHOLDS, TARGET_COLUMN
from utils.helpers import generate_transaction_id, get_risk_category


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(file_path_or_buffer: Union[str, io.IOBase]) -> pd.DataFrame:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Accepts either a filesystem path (string) or a file-like buffer (e.g. a
    :class:`~streamlit.runtime.uploaded_file_manager.UploadedFile`).

    Args:
        file_path_or_buffer: Path string or readable buffer containing CSV data.

    Returns:
        A DataFrame with the parsed CSV contents.

    Raises:
        FileNotFoundError: If a path string is given and the file does not exist.
        ValueError: If the file is empty or cannot be parsed as CSV.
    """
    try:
        if isinstance(file_path_or_buffer, str):
            if not os.path.isfile(file_path_or_buffer):
                raise FileNotFoundError(
                    f"CSV file not found: {file_path_or_buffer}"
                )

        df = pd.read_csv(file_path_or_buffer)

        if df.empty:
            raise ValueError("The CSV file is empty (no rows were parsed).")

        return df

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_samples: int = 1000,
    fraud_ratio: float = 0.02,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a realistic synthetic credit-card transaction dataset.

    The generated data mimics the structure of the well-known Kaggle
    *Credit Card Fraud Detection* dataset:
      * ``Time`` – seconds elapsed from the first transaction (uniform 0–172 800,
        i.e. ≈ 2 days).
      * ``V1``–``V28`` – PCA-transformed features.  Legitimate transactions are
        drawn from a standard-normal distribution.  Fraudulent transactions have
        shifted means on a subset of V-columns (V1, V3, V4, V7, V10, V12, V14,
        V16) so that a classifier can learn to separate the two classes.
      * ``Amount`` – log-normal amounts.  Fraudulent transactions tend toward
        higher amounts.
      * ``Class`` – 0 for legitimate, 1 for fraudulent.
      * ``transaction_id`` – unique human-readable ID (``TXN-…``).

    Args:
        n_samples:   Total number of transactions to generate.
        fraud_ratio: Fraction of transactions that are fraudulent (0–1).
        seed:        Random seed for reproducibility.

    Returns:
        A DataFrame with columns
        ``transaction_id, Time, V1–V28, Amount, Class``.
    """
    rng = np.random.RandomState(seed)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions -------------------------------------------
    legit_v = rng.randn(n_legit, 28)                       # standard-normal
    legit_amounts = np.round(rng.lognormal(mean=3.0, sigma=1.5, size=n_legit), 2)
    # Clip extreme amounts to a realistic range
    legit_amounts = np.clip(legit_amounts, 0.01, 25_000.0)
    legit_time = np.sort(rng.uniform(0, 172_800, size=n_legit))

    # --- Fraudulent transactions -------------------------------------------
    fraud_v = rng.randn(n_fraud, 28)

    # Shift specific V-columns for fraud to create separability.
    # In real PCA-transformed data, fraud tends to show extreme values on
    # certain principal components.
    shift_map = {
        0: -2.5,   # V1
        2: -1.8,   # V3
        3:  2.5,   # V4
        6: -2.0,   # V7
        9: -3.0,   # V10
        11: -2.0,  # V12
        13: -3.5,  # V14
        15: -2.2,  # V16
    }
    for col_idx, shift in shift_map.items():
        fraud_v[:, col_idx] += shift
    # Increase variance slightly on shifted columns for realism
    for col_idx in shift_map:
        fraud_v[:, col_idx] *= 1.3

    # Fraud amounts skew higher
    fraud_amounts = np.round(rng.lognormal(mean=4.5, sigma=1.8, size=n_fraud), 2)
    fraud_amounts = np.clip(fraud_amounts, 1.0, 25_000.0)
    fraud_time = np.sort(rng.uniform(0, 172_800, size=n_fraud))

    # --- Combine ------------------------------------------------------------
    v_all = np.vstack([legit_v, fraud_v])
    amounts_all = np.concatenate([legit_amounts, fraud_amounts])
    time_all = np.concatenate([legit_time, fraud_time])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle everything together
    shuffle_idx = rng.permutation(n_samples)
    v_all = v_all[shuffle_idx]
    amounts_all = amounts_all[shuffle_idx]
    time_all = time_all[shuffle_idx]
    labels = labels[shuffle_idx]

    # Round V-columns for tidiness
    v_all = np.round(v_all, 6)
    time_all = np.round(time_all, 2)

    # Build DataFrame
    v_col_names = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(v_all, columns=v_col_names)
    df.insert(0, "Time", time_all)
    df["Amount"] = amounts_all
    df["Class"] = labels.astype(int)
    df.insert(0, "transaction_id", [generate_transaction_id() for _ in range(n_samples)])

    # Re-sort by Time for a natural chronological look, then reset index
    df = df.sort_values("Time").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Display data preparation
# ---------------------------------------------------------------------------

def prepare_display_data(
    df: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Merge raw transaction data with model predictions for display.

    Adds or updates the following columns:
      * ``transaction_id`` – generated if not already present.
      * ``predicted_class`` – the binary prediction (0 / 1).
      * ``fraud_probability`` – the model's estimated fraud probability.
      * ``risk_level`` – human-readable risk category derived from the
        probability and the configured thresholds.

    Args:
        df:           The original transaction DataFrame.
        predictions:  Array of binary predictions (0/1), same length as *df*.
        probabilities: Array of fraud probabilities (0.0–1.0).

    Returns:
        A copy of *df* augmented with prediction columns.
    """
    display = df.copy()

    # Ensure transaction_id exists
    if "transaction_id" not in display.columns:
        display.insert(
            0,
            "transaction_id",
            [generate_transaction_id() for _ in range(len(display))],
        )

    if predictions is not None:
        display["predicted_class"] = predictions.astype(int)
    else:
        display["predicted_class"] = 0

    if probabilities is not None:
        display["fraud_probability"] = np.round(probabilities, 4)
    else:
        display["fraud_probability"] = 0.0

    display["risk_level"] = display["fraud_probability"].apply(
        lambda p: get_risk_category(p, RISK_THRESHOLDS)
    )

    return display
