"""
Pytest configuration and shared test fixtures for BankGuard AI test suite.
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import generate_synthetic_data
from ml.preprocessing import TransactionPreprocessor
from ml.train import train_model
from ml.predict import FraudPredictor
from database.db import DatabaseManager
from database.repository import TransactionRepository


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provide a reliable 100-row synthetic dataframe with known columns."""
    return generate_synthetic_data(n_samples=100, fraud_ratio=0.1, seed=42)


@pytest.fixture
def trained_predictor(sample_dataframe: pd.DataFrame) -> FraudPredictor:
    """Provide a fitted FraudPredictor instance."""
    prep = TransactionPreprocessor().fit(sample_dataframe)
    X = prep.get_feature_matrix(sample_dataframe)
    y = sample_dataframe["Class"].values
    model = train_model(X, y, model_type="random_forest", seed=42)
    return FraudPredictor(model=model, preprocessor=prep)


@pytest.fixture
def temp_repo() -> TransactionRepository:
    """Provide a TransactionRepository connected to an isolated temporary SQLite database."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_bankguard.db")
    db_manager = DatabaseManager(db_path=temp_db_path)
    db_manager.initialize()
    return TransactionRepository(db_manager=db_manager)
