"""
BankGuard AI — Transaction Explorer Page

Tabular exploration, dynamic multi-facet filtering, sorting, and export of transactions.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.tables import display_transaction_table, display_download_button
from utils.helpers import format_currency


def render_transaction_explorer(data_state: Dict[str, Any]) -> None:
    """Render the Transaction Explorer interface."""
    st.title("🔍 Transaction Explorer & Ledger Search")
    st.markdown(
        "Inspect individual transaction logs, filter by risk criteria, search by ID, and export audit subsets."
    )

    df: pd.DataFrame | None = data_state.get("df")

    if df is None or df.empty:
        st.warning("⚠️ No dataset loaded. Please load demo data or upload a CSV in **Fraud Detection**.")
        return

    working_df = df.copy()

    # Filter controls in sidebar or expandable filter tray
    with st.expander("🛠️ Filter & Query Controls", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            search_query = st.text_input(
                "Search Transaction ID",
                placeholder="e.g. TXN-",
                help="Search by transaction identifier string",
            )

        with c2:
            fraud_filter = st.selectbox(
                "Fraud Status",
                options=["All Transactions", "Fraudulent Only", "Legitimate Only"],
                index=0,
            )

        with c3:
            available_risks = ["All Risks"]
            if "risk_level" in working_df.columns:
                available_risks += list(working_df["risk_level"].dropna().unique())
            risk_filter = st.selectbox("Risk Level", options=available_risks, index=0)

        with c4:
            min_val = float(working_df["Amount"].min()) if "Amount" in working_df.columns else 0.0
            max_val = float(working_df["Amount"].max()) if "Amount" in working_df.columns else 1000.0
            if min_val >= max_val:
                max_val = min_val + 100.0

            amount_range = st.slider(
                "Amount Range ($)",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(min_val), float(max_val)),
            )

    # Apply filters
    if search_query and "transaction_id" in working_df.columns:
        working_df = working_df[
            working_df["transaction_id"].astype(str).str.contains(search_query, case=False, na=False)
        ]

    if fraud_filter == "Fraudulent Only":
        if "Class" in working_df.columns:
            working_df = working_df[working_df["Class"] == 1]
        elif "predicted_class" in working_df.columns:
            working_df = working_df[working_df["predicted_class"] == 1]
    elif fraud_filter == "Legitimate Only":
        if "Class" in working_df.columns:
            working_df = working_df[working_df["Class"] == 0]
        elif "predicted_class" in working_df.columns:
            working_df = working_df[working_df["predicted_class"] == 0]

    if risk_filter != "All Risks" and "risk_level" in working_df.columns:
        working_df = working_df[working_df["risk_level"] == risk_filter]

    if "Amount" in working_df.columns:
        working_df = working_df[
            (working_df["Amount"] >= amount_range[0]) & (working_df["Amount"] <= amount_range[1])
        ]

    # Summary of filtered results
    st.markdown(
        f"**Showing `{len(working_df):,}` of `{len(df):,}` transactions** "
        f"(Filtered volume: `{format_currency(working_df['Amount'].sum() if 'Amount' in working_df.columns else 0)}`)"
    )

    # Action bar
    c_left, c_right = st.columns([4, 1])
    with c_right:
        display_download_button(
            working_df,
            filename="filtered_transactions.csv",
            label="📥 Export Filtered CSV",
        )

    # Render table
    display_transaction_table(working_df, page_size=25, key_prefix="explorer_tbl")
