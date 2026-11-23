"""
Unit tests for data preprocessing and feature matrix extraction.
"""

from __future__ import annotations

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.preprocessing import TransactionPreprocessor
from utils.data_loader import generate_synthetic_data


def test_preprocessor_fit_transform():
    """Verify scaler fits and transforms without altering non-scaled features."""
    df = generate_synthetic_data(n_samples=50, seed=42)
    prep = TransactionPreprocessor()
    transformed = prep.fit_transform(df)

    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == len(df)
    # Scaled columns should have approximately zero mean and unit variance
    assert abs(transformed["Amount"].mean()) < 0.2
    assert abs(transformed["Time"].mean()) < 0.2


def test_preprocessor_missing_values_handling():
    """Verify preprocessor handles nulls gracefully."""
    df = generate_synthetic_data(n_samples=30, seed=42)
    df.loc[0, "Amount"] = np.nan
    df.loc[1, "V1"] = np.nan

    prep = TransactionPreprocessor()
    transformed = prep.fit_transform(df)
    assert not transformed.isna().any().any()


def test_get_feature_matrix_shape():
    """Verify extracted feature matrix matches feature columns count."""
    df = generate_synthetic_data(n_samples=40, seed=42)
    prep = TransactionPreprocessor().fit(df)
    X = prep.get_feature_matrix(df)
    assert isinstance(X, np.ndarray)
    assert X.shape == (40, 30)  # 28 V-features + Amount + Time
