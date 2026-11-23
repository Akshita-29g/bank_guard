"""
Unit tests for data loader, synthetic generator, and validation utilities.
"""

from __future__ import annotations

import io
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import generate_synthetic_data, load_csv, prepare_display_data
from utils.validation import validate_dataset, validate_prediction_input


def test_generate_synthetic_data_structure():
    """Verify generated synthetic dataset matches expected banking schema."""
    df = generate_synthetic_data(n_samples=200, fraud_ratio=0.05, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "transaction_id" in df.columns
    assert "Time" in df.columns
    assert "Amount" in df.columns
    assert "Class" in df.columns
    for i in range(1, 29):
        assert f"V{i}" in df.columns


def test_generate_synthetic_data_fraud_distribution():
    """Verify generated dataset contains both classes with approx expected fraud ratio."""
    df = generate_synthetic_data(n_samples=500, fraud_ratio=0.04, seed=42)
    fraud_count = (df["Class"] == 1).sum()
    assert fraud_count > 0
    assert fraud_count < len(df)


def test_load_csv_from_buffer():
    """Verify loading from an in-memory CSV buffer."""
    raw_csv = "Time,Amount,Class,V1,V2\n10,50.25,0,0.1,-0.5\n20,999.0,1,-1.2,2.0\n"
    buffer = io.StringIO(raw_csv)
    df = load_csv(buffer)
    assert len(df) == 2
    assert df["Amount"].iloc[1] == 999.0


def test_dataset_validation():
    """Verify validation detects valid and invalid schemas."""
    good_df = generate_synthetic_data(n_samples=50, seed=42)
    res = validate_dataset(good_df)
    assert res["valid"] is True
    assert res["has_target"] is True

    bad_df = pd.DataFrame({"random_col": [1, 2, 3]})
    res_bad = validate_dataset(bad_df)
    assert res_bad["valid"] is False
