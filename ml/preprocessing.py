"""
BankGuard AI — Transaction Preprocessing

Handles feature scaling, missing-value imputation, and feature-matrix
extraction for the credit-card fraud detection pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import FEATURE_COLUMNS


class TransactionPreprocessor:
    """Scale ``Amount`` and ``Time`` while keeping V1–V28 untouched.

    The Kaggle credit-card dataset ships V1–V28 already PCA-transformed, so
    only ``Amount`` and ``Time`` need standardisation.  This class wraps a
    :class:`~sklearn.preprocessing.StandardScaler` for those two columns and
    provides convenience methods for the full pipeline.
    """

    # Columns that need scaling (V1-V28 are already PCA-transformed)
    _SCALE_COLS: List[str] = ["Amount", "Time"]

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        scale_features: bool = True,
    ) -> None:
        """
        Args:
            feature_columns: Ordered list of feature column names to use.
                             Defaults to :pydata:`config.FEATURE_COLUMNS`.
            scale_features:  If ``False``, skip scaling entirely and just
                             pass through the data (useful for debugging).
        """
        self.feature_columns: List[str] = (
            list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
        )
        self.scale_features: bool = scale_features
        self._scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TransactionPreprocessor":
        """Fit the scaler on the ``Amount`` and ``Time`` columns of *df*.

        Args:
            df: Training DataFrame (must contain ``Amount`` and ``Time``).

        Returns:
            ``self``, to allow chaining (e.g. ``preprocessor.fit(df).transform(df)``).

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot fit on an empty DataFrame.")

        if self.scale_features:
            cols_to_scale = [c for c in self._SCALE_COLS if c in df.columns]
            if cols_to_scale:
                self._scaler.fit(df[cols_to_scale].values)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler and handle missing values.

        * Scales ``Amount`` and ``Time`` using the previously fitted
          :class:`~sklearn.preprocessing.StandardScaler`.
        * Leaves ``V1``–``V28`` untouched (already PCA-transformed).
        * Fills remaining NaN values with ``0``.

        Args:
            df: DataFrame to transform.

        Returns:
            A **copy** of *df* with scaled columns and no NaN values.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
            ValueError:   If the DataFrame is empty.
        """
        if not self._fitted:
            raise RuntimeError(
                "TransactionPreprocessor has not been fitted yet. "
                "Call fit() or fit_transform() first."
            )
        if df.empty:
            raise ValueError("Cannot transform an empty DataFrame.")

        result = df.copy()

        # Scale Amount and Time
        if self.scale_features:
            cols_to_scale = [c for c in self._SCALE_COLS if c in result.columns]
            if cols_to_scale:
                result[cols_to_scale] = self._scaler.transform(
                    result[cols_to_scale].values
                )

        # Handle missing values — fill with 0 (neutral for scaled features)
        result = result.fillna(0)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper: ``fit`` then ``transform``.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed copy of *df*.
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Feature-matrix extraction
    # ------------------------------------------------------------------

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the feature columns as a NumPy array.

        Only columns present in both *df* and :attr:`feature_columns` are
        included, preserving the order defined in :attr:`feature_columns`.

        Args:
            df: DataFrame (should already be transformed).

        Returns:
            2-D NumPy array of shape ``(n_samples, n_features)``.

        Raises:
            ValueError: If none of the expected feature columns are found.
        """
        available = [c for c in self.feature_columns if c in df.columns]
        if not available:
            raise ValueError(
                "None of the expected feature columns were found in the DataFrame. "
                f"Expected: {self.feature_columns}, got: {list(df.columns)}"
            )
        return df[available].values

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise this preprocessor to disk with :mod:`joblib`.

        Args:
            path: Destination file path (e.g. ``"models/preprocessor.joblib"``).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TransactionPreprocessor":
        """Deserialise a preprocessor previously saved with :meth:`save`.

        Args:
            path: Path to the saved file.

        Returns:
            A fully-fitted :class:`TransactionPreprocessor` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        return joblib.load(path)
