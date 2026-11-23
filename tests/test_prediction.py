"""
Unit tests for fraud prediction and risk level classification.
"""

from __future__ import annotations

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.predict import FraudPredictor
from utils.helpers import get_risk_category


def test_get_risk_category():
    """Verify probability thresholding maps to correct risk categories."""
    assert get_risk_category(0.1) == "Low Risk"
    assert get_risk_category(0.5) == "Medium Risk"
    assert get_risk_category(0.9) == "High Risk"


def test_fraud_predictor_batch(trained_predictor: FraudPredictor, sample_dataframe: pd.DataFrame):
    """Verify batch prediction adds risk_level, fraud_probability, predicted_class."""
    predictions = trained_predictor.predict(sample_dataframe)
    assert "fraud_probability" in predictions.columns
    assert "predicted_class" in predictions.columns
    assert "risk_level" in predictions.columns
    assert predictions["fraud_probability"].between(0.0, 1.0).all()
    assert set(predictions["predicted_class"].unique()).issubset({0, 1})


def test_fraud_predictor_single(trained_predictor: FraudPredictor, sample_dataframe: pd.DataFrame):
    """Verify single transaction dictionary scoring."""
    row = sample_dataframe.iloc[0].to_dict()
    res = trained_predictor.predict_single(row)
    assert "fraud_probability" in res
    assert "predicted_class" in res
    assert "risk_level" in res
    assert res["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
