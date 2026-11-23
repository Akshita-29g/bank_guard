"""
BankGuard AI — Fraud Prediction

High-level predictor class that wraps a trained model and a fitted
preprocessor into a single, easy-to-use inference interface.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_COLUMNS
from utils.helpers import get_risk_category
from ml.preprocessing import TransactionPreprocessor
from ml.train import load_model


class FraudPredictor:
    """End-to-end fraud predictor: preprocess → predict → classify risk.

    Can be initialised with live objects **or** with file paths from which
    the model and preprocessor will be loaded lazily.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[TransactionPreprocessor] = None,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            model:            A fitted scikit-learn estimator.
            preprocessor:     A fitted :class:`TransactionPreprocessor`.
            model_path:       Path to a serialised model (loaded on demand).
            preprocessor_path: Path to a serialised preprocessor.
        """
        self.model = model
        self.preprocessor = preprocessor
        self._model_path = model_path
        self._preprocessor_path = preprocessor_path

        # Load from paths if objects were not provided directly
        if self.model is None and self._model_path is not None:
            self.model = load_model(self._model_path)
        if self.preprocessor is None and self._preprocessor_path is not None:
            self.preprocessor = TransactionPreprocessor.load(self._preprocessor_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return ``True`` if both model and preprocessor are loaded."""
        return self.model is not None and self.preprocessor is not None

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a full DataFrame of transactions.

        The input DataFrame is preprocessed, then scored by the model.
        Three new columns are appended to a **copy** of *df*:

        * ``fraud_probability`` – estimated probability of fraud (0–1).
        * ``predicted_class``   – binary label (0 = legit, 1 = fraud).
        * ``risk_level``        – ``"Low Risk"`` / ``"Medium Risk"`` /
          ``"High Risk"`` via :func:`utils.helpers.get_risk_category`.

        Args:
            df: Raw transaction DataFrame (must contain the feature columns).

        Returns:
            A new DataFrame with prediction columns added.

        Raises:
            RuntimeError: If :meth:`is_ready` is ``False``.
            ValueError:   If *df* is empty.
        """
        if not self.is_ready():
            raise RuntimeError(
                "FraudPredictor is not ready. "
                "Ensure both model and preprocessor are loaded."
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Preprocess
        processed = self.preprocessor.transform(df)
        X = self.preprocessor.get_feature_matrix(processed)

        # Predict
        y_pred = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            # Sigmoid of decision function for LR without predict_proba
            scores = self.model.decision_function(X)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_proba = y_pred.astype(float)

        # Build result
        result = df.copy()
        result["fraud_probability"] = np.round(y_proba, 4)
        result["predicted_class"] = y_pred.astype(int)
        result["risk_level"] = result["fraud_probability"].apply(get_risk_category)

        return result

    def predict_single(self, transaction: dict) -> dict:
        """Predict fraud for a single transaction.

        Args:
            transaction: Dictionary with feature column keys (e.g.
                         ``{"Amount": 150.0, "Time": 3600, "V1": -1.2, ...}``).

        Returns:
            A dict with keys: ``fraud_probability``, ``predicted_class``,
            and ``risk_level``.
        """
        df = pd.DataFrame([transaction])
        result_df = self.predict(df)
        row = result_df.iloc[0]
        return {
            "fraud_probability": float(row["fraud_probability"]),
            "predicted_class": int(row["predicted_class"]),
            "risk_level": str(row["risk_level"]),
        }
