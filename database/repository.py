"""
BankGuard AI — Transaction & Metrics Repository

High-level CRUD helpers that sit on top of :class:`DatabaseManager` and
speak *pandas* DataFrames.  Every public method guards against SQL
injection via parameterised queries and wraps work in ``try / except``
so callers never see raw database exceptions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from database.db import DatabaseManager

logger = logging.getLogger(__name__)


class TransactionRepository:
    """Repository pattern over the BankGuard SQLite schema."""

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        """
        Parameters
        ----------
        db_manager : DatabaseManager, optional
            An existing manager instance.  When *None* a default one is
            created using ``config.DATABASE_PATH`` and immediately
            initialised (schema applied).
        """
        if db_manager is None:
            db_manager = DatabaseManager()
            db_manager.connect()
            db_manager.initialize()
        self.db = db_manager

    # ------------------------------------------------------------------ #
    #  Transactions
    # ------------------------------------------------------------------ #

    def save_transactions(self, df: pd.DataFrame) -> int:
        """Persist prediction results.  Duplicate ``transaction_id`` rows
        are silently skipped (``INSERT OR IGNORE``).

        Expected DataFrame columns (all optional — missing ones default
        to ``None``):

        * ``transaction_id`` – unique id; auto-generated UUID if absent.
        * ``Time`` or ``time_value``
        * ``Amount`` or ``amount``
        * ``predicted_class``
        * ``fraud_probability``
        * ``risk_level``

        Parameters
        ----------
        df : pd.DataFrame
            Prediction results.

        Returns
        -------
        int
            Number of rows actually inserted.
        """
        if df is None or df.empty:
            logger.info("save_transactions: empty DataFrame — nothing to save.")
            return 0

        saved = 0
        insert_sql = """
            INSERT OR IGNORE INTO transactions
                (transaction_id, time_value, amount,
                 predicted_class, fraud_probability, risk_level, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        try:
            conn = self.db.connect()
            for _, row in df.iterrows():
                try:
                    txn_id = self._col(row, "transaction_id")
                    if txn_id is None:
                        txn_id = str(uuid.uuid4())

                    time_val = self._col(row, "Time", "time_value")
                    amount = self._col(row, "Amount", "amount")
                    pred_class = self._col(row, "predicted_class")
                    fraud_prob = self._col(row, "fraud_probability")
                    risk_lvl = self._col(row, "risk_level")
                    now = datetime.now(timezone.utc).isoformat()

                    cursor = conn.execute(
                        insert_sql,
                        (txn_id, time_val, amount, pred_class, fraud_prob, risk_lvl, now),
                    )
                    if cursor.rowcount > 0:
                        saved += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping row: %s", exc)

            conn.commit()
            logger.info("Saved %d / %d transactions.", saved, len(df))
        except Exception as exc:  # noqa: BLE001
            logger.error("save_transactions failed: %s", exc)

        return saved

    def get_transactions(
        self,
        limit: int = 100,
        offset: int = 0,
        risk_level: Optional[str] = None,
        predicted_class: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve transactions with optional filtering.

        Parameters
        ----------
        limit : int
            Maximum rows to return.
        offset : int
            Number of rows to skip (for pagination).
        risk_level : str, optional
            Filter by risk level (e.g. ``"High Risk"``).
        predicted_class : int, optional
            Filter by predicted class (``0`` = legit, ``1`` = fraud).

        Returns
        -------
        pd.DataFrame
        """
        try:
            clauses: List[str] = []
            params: List[Any] = []

            if risk_level is not None:
                clauses.append("risk_level = ?")
                params.append(risk_level)
            if predicted_class is not None:
                clauses.append("predicted_class = ?")
                params.append(int(predicted_class))

            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            query = (
                f"SELECT * FROM transactions{where} "
                f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
            )
            params.extend([limit, offset])

            rows = self.db.fetchall(query, tuple(params))
            if rows:
                return pd.DataFrame([dict(r) for r in rows])
            return pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            logger.error("get_transactions failed: %s", exc)
            return pd.DataFrame()

    def get_transaction_count(self, risk_level: Optional[str] = None) -> int:
        """Return the total number of stored transactions, optionally
        filtered by *risk_level*.

        Parameters
        ----------
        risk_level : str, optional
            E.g. ``"High Risk"``.

        Returns
        -------
        int
        """
        try:
            if risk_level is not None:
                row = self.db.fetchone(
                    "SELECT COUNT(*) AS cnt FROM transactions WHERE risk_level = ?",
                    (risk_level,),
                )
            else:
                row = self.db.fetchone("SELECT COUNT(*) AS cnt FROM transactions")

            return int(row["cnt"]) if row else 0
        except Exception as exc:  # noqa: BLE001
            logger.error("get_transaction_count failed: %s", exc)
            return 0

    def clear_transactions(self) -> int:
        """Delete **all** rows from the transactions table.

        Returns
        -------
        int
            Number of rows deleted.
        """
        try:
            cursor = self.db.execute("DELETE FROM transactions")
            count = cursor.rowcount
            logger.info("Cleared %d transactions.", count)
            return count
        except Exception as exc:  # noqa: BLE001
            logger.error("clear_transactions failed: %s", exc)
            return 0

    def get_fraud_summary(self) -> Dict[str, Any]:
        """Return aggregate fraud statistics.

        Returns
        -------
        dict
            Keys: ``total``, ``fraud``, ``legitimate``, ``fraud_rate``,
            ``high_risk``, ``medium_risk``, ``low_risk``,
            ``avg_fraud_probability``.
        """
        summary: Dict[str, Any] = {
            "total": 0,
            "fraud": 0,
            "legitimate": 0,
            "fraud_rate": 0.0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "avg_fraud_probability": 0.0,
        }
        try:
            row = self.db.fetchone("SELECT COUNT(*) AS cnt FROM transactions")
            total = int(row["cnt"]) if row else 0
            summary["total"] = total

            row = self.db.fetchone(
                "SELECT COUNT(*) AS cnt FROM transactions WHERE predicted_class = 1"
            )
            fraud = int(row["cnt"]) if row else 0
            summary["fraud"] = fraud
            summary["legitimate"] = total - fraud
            summary["fraud_rate"] = round(fraud / total * 100, 2) if total else 0.0

            for level in ("High Risk", "Medium Risk", "Low Risk"):
                row = self.db.fetchone(
                    "SELECT COUNT(*) AS cnt FROM transactions WHERE risk_level = ?",
                    (level,),
                )
                key = level.lower().replace(" ", "_")
                summary[key] = int(row["cnt"]) if row else 0

            row = self.db.fetchone(
                "SELECT AVG(fraud_probability) AS avg_prob FROM transactions"
            )
            summary["avg_fraud_probability"] = (
                round(float(row["avg_prob"]), 4)
                if row and row["avg_prob"] is not None
                else 0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("get_fraud_summary failed: %s", exc)

        return summary

    # ------------------------------------------------------------------ #
    #  Analysis history
    # ------------------------------------------------------------------ #

    def save_analysis(
        self,
        analysis_type: str,
        dataset_name: Optional[str] = None,
        total: int = 0,
        fraud_count: int = 0,
        model_name: Optional[str] = None,
        accuracy: Optional[float] = None,
        model_used: Optional[str] = None,
    ) -> None:
        """Record a completed analysis run.

        Parameters
        ----------
        analysis_type : str
            E.g. ``"batch_prediction"``, ``"real_time"``.
        dataset_name : str, optional
        total : int
        fraud_count : int
        model_name : str, optional
        accuracy : float, optional
        model_used : str, optional (alias for model_name)
        """
        chosen_model = model_name or model_used
        try:
            self.db.execute(
                """
                INSERT INTO analysis_history
                    (analysis_type, dataset_name, total_transactions,
                     fraud_detected, model_used, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (analysis_type, dataset_name, total, fraud_count, chosen_model, accuracy),
            )
            logger.info("Saved analysis record: %s", analysis_type)
        except Exception as exc:  # noqa: BLE001
            logger.error("save_analysis failed: %s", exc)

    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent analysis records.

        Parameters
        ----------
        limit : int
            Maximum rows.

        Returns
        -------
        list[dict]
        """
        try:
            rows = self.db.fetchall(
                "SELECT * FROM analysis_history ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_analysis_history failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Model metrics
    # ------------------------------------------------------------------ #

    def save_model_metrics(
        self,
        model_name: str,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float,
        pr_auc: float,
    ) -> None:
        """Persist evaluation metrics for a trained model.

        Parameters
        ----------
        model_name : str
        precision, recall, f1, roc_auc, pr_auc : float
        """
        try:
            self.db.execute(
                """
                INSERT INTO model_metrics
                    (model_name, precision_score, recall_score,
                     f1_score, roc_auc, pr_auc)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_name, precision, recall, f1, roc_auc, pr_auc),
            )
            logger.info("Saved model metrics: %s", model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("save_model_metrics failed: %s", exc)

    def get_model_metrics(self) -> List[Dict[str, Any]]:
        """Return all stored model metric records.

        Returns
        -------
        list[dict]
        """
        try:
            rows = self.db.fetchall(
                "SELECT * FROM model_metrics ORDER BY trained_at DESC"
            )
            return [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error("get_model_metrics failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _col(row: pd.Series, *names: str) -> Any:
        """Return the first matching column value from *row*, or ``None``
        if none of the candidate column names exist."""
        for name in names:
            if name in row.index:
                val = row[name]
                # Convert numpy/pandas NA to Python None
                if pd.isna(val):
                    return None
                return val
        return None
