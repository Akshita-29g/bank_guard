"""
BankGuard AI — Table & Data Export Components

Formatted data tables and export utilities for transaction streams.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import THEME_COLORS


def _style_risk_cells(val: Any) -> str:
    """CSS background style generator for risk level values."""
    if not isinstance(val, str):
        return ""
    if val == "High Risk":
        return f"background-color: rgba(255, 75, 75, 0.25); color: #FF6B6B; font-weight: bold;"
    elif val == "Medium Risk":
        return f"background-color: rgba(255, 165, 0, 0.25); color: #FFA500; font-weight: bold;"
    elif val == "Low Risk":
        return f"background-color: rgba(0, 212, 170, 0.20); color: #00D4AA; font-weight: bold;"
    return ""


def _style_fraud_class(val: Any) -> str:
    """CSS background style generator for fraud predicted class."""
    if val == 1:
        return "background-color: rgba(255, 75, 75, 0.3); color: #FF4B4B; font-weight: bold;"
    elif val == 0:
        return "background-color: rgba(0, 212, 170, 0.15); color: #00D4AA;"
    return ""


def display_transaction_table(
    df: pd.DataFrame,
    page_size: int = 25,
    key_prefix: str = "txn_table",
) -> None:
    """Display an interactive paginated transaction table with styled indicators."""
    if df.empty:
        st.info("No transaction records match the criteria.")
        return

    # Select primary presentation columns
    priority_cols = ["transaction_id", "Time", "Amount", "risk_level", "fraud_probability", "predicted_class", "Class"]
    available_cols = [c for c in priority_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in available_cols]
    ordered_cols = available_cols + remaining_cols

    display_df = df[ordered_cols].copy()

    # Format numeric columns for clean viewing
    if "Amount" in display_df.columns:
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "-")
    if "fraud_probability" in display_df.columns:
        display_df["fraud_probability"] = display_df["fraud_probability"].apply(
            lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "-"
        )
    if "Time" in display_df.columns:
        display_df["Time"] = display_df["Time"].apply(lambda x: f"{x:,.0f}s" if pd.notnull(x) else "-")

    # Pagination controls
    total_rows = len(display_df)
    total_pages = max(1, (total_rows - 1) // page_size + 1)

    if total_pages > 1:
        c1, c2, c3 = st.columns([2, 3, 2])
        with c2:
            page_num = st.number_input(
                f"Page (Total records: {total_rows:,})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key=f"{key_prefix}_page",
            )
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_df = display_df.iloc[start_idx:end_idx]
    else:
        page_df = display_df

    st.dataframe(
        page_df,
        use_container_width=True,
        hide_index=True,
    )


def display_prediction_results(df: pd.DataFrame, title: str = "Fraud Scoring & Risk Summary") -> None:
    """Display highlighted prediction results with summary counts."""
    st.subheader(title)

    if df.empty:
        st.info("No prediction data available.")
        return

    # High-level counts
    total = len(df)
    high_risk = len(df[df.get("risk_level") == "High Risk"]) if "risk_level" in df.columns else 0
    flagged = len(df[df.get("predicted_class") == 1]) if "predicted_class" in df.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Evaluated Volume", f"{total:,}")
    c2.metric("Flagged Fraud (Class 1)", f"{flagged:,}", delta=f"{(flagged/total*100):.2f}%" if total else None)
    c3.metric("High Risk Tier", f"{high_risk:,}", delta=f"{(high_risk/total*100):.2f}%" if total else None)

    display_transaction_table(df, page_size=20, key_prefix="pred_results")


def display_download_button(
    df: pd.DataFrame,
    filename: str = "bankguard_transactions.csv",
    label: str = "📥 Download CSV Data",
) -> None:
    """Render a one-click CSV download button."""
    if df.empty:
        return

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
    )
