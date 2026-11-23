"""
Unit tests for SQLite database operations and repository persistence.
"""

from __future__ import annotations

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.repository import TransactionRepository
from utils.data_loader import generate_synthetic_data


def test_database_initialization(temp_repo: TransactionRepository):
    """Verify repository connects and table initializes cleanly."""
    count = temp_repo.get_transaction_count()
    assert count == 0


def test_save_and_retrieve_transactions(temp_repo: TransactionRepository):
    """Verify transactions persist and query back correctly."""
    df = generate_synthetic_data(n_samples=25, seed=42)
    df["predicted_class"] = 0
    df["fraud_probability"] = 0.05
    df["risk_level"] = "Low Risk"

    saved = temp_repo.save_transactions(df)
    assert saved == 25

    records = temp_repo.get_transactions(limit=50)
    assert len(records) == 25
    assert "transaction_id" in records.columns


def test_save_and_retrieve_analysis_history(temp_repo: TransactionRepository):
    """Verify analysis audit trail logs correctly."""
    temp_repo.save_analysis(
        analysis_type="batch_test",
        dataset_name="test_data.csv",
        total=100,
        fraud_count=3,
        model_used="Random Forest",
        accuracy=0.97,
    )

    history = temp_repo.get_analysis_history(limit=5)
    assert len(history) == 1
    assert history[0]["dataset_name"] == "test_data.csv"
    assert history[0]["fraud_detected"] == 3
