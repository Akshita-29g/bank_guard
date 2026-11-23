"""
BankGuard AI — Overview Dashboard Page

Executive summary of banking transactions, fraud KPIs, and portfolio risk distribution.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import THEME_COLORS
from utils.helpers import format_currency, format_percentage, safe_divide
from utils.data_loader import generate_synthetic_data
from components.metrics import display_metric_row
from components.charts import (
    create_fraud_distribution_chart,
    create_amount_distribution_chart,
    create_time_series_chart,
    create_risk_category_chart,
)


def render_dashboard(data_state: Dict[str, Any]) -> None:
    """Render the executive Overview Dashboard."""
    st.title("🛡️ Banking Transaction Overview & Fraud Intelligence")
    st.markdown(
        "Real-time monitoring of transaction health, anomaly exposure, and class distributions."
    )

    df: pd.DataFrame | None = data_state.get("df")

    # If no data is currently loaded, display helpful onboarding card with quick demo loader
    if df is None or df.empty:
        st.info("👋 Welcome to **BankGuard AI**! No transaction stream is currently active.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                """
                To begin exploring:
                - Navigate to **Fraud Detection** on the sidebar to upload your own Kaggle/Banking CSV.
                - Or instantly generate a synthetic demo dataset below to test all features.
                """
            )
            if st.button("⚡ Load 1,000 Synthetic Demo Transactions", type="primary"):
                demo_df = generate_synthetic_data(n_samples=1000, fraud_ratio=0.03, seed=42)
                st.session_state.df = demo_df
                st.success("Successfully loaded 1,000 synthetic demo transactions!")
                st.rerun()
        return

    # Compute key metrics
    total_txns = len(df)
    total_amount = float(df["Amount"].sum()) if "Amount" in df.columns else 0.0
    avg_amount = float(df["Amount"].mean()) if "Amount" in df.columns else 0.0

    # Determine fraud count based on Class or predicted_class
    if "Class" in df.columns:
        fraud_txns = int((df["Class"] == 1).sum())
        legit_txns = int((df["Class"] == 0).sum())
    elif "predicted_class" in df.columns:
        fraud_txns = int((df["predicted_class"] == 1).sum())
        legit_txns = total_txns - fraud_txns
    else:
        fraud_txns = 0
        legit_txns = total_txns

    fraud_rate = safe_divide(fraud_txns, total_txns, 0.0) * 100

    high_risk_txns = (
        int((df["risk_level"] == "High Risk").sum())
        if "risk_level" in df.columns
        else (fraud_txns if "Class" in df.columns else 0)
    )

    # Display KPI Cards Row
    kpi_metrics = [
        {
            "label": "Total Transactions",
            "value": f"{total_txns:,}",
            "icon": "💳",
            "color": THEME_COLORS["accent"],
        },
        {
            "label": "Total Amount",
            "value": format_currency(total_amount),
            "icon": "💰",
            "color": "#3B82F6",
        },
        {
            "label": "Average Ticket",
            "value": format_currency(avg_amount),
            "icon": "🏷️",
            "color": "#8B5CF6",
        },
        {
            "label": "Fraud Incidents",
            "value": f"{fraud_txns:,}",
            "delta": f"{fraud_rate:.2f}% incidence",
            "icon": "🚨",
            "color": THEME_COLORS["danger"] if fraud_txns > 0 else THEME_COLORS["success"],
        },
        {
            "label": "High Risk Exposure",
            "value": f"{high_risk_txns:,}",
            "delta": f"{safe_divide(high_risk_txns, total_txns)*100:.1f}% portfolio",
            "icon": "⚠️",
            "color": THEME_COLORS["warning"],
        },
    ]

    display_metric_row(kpi_metrics)

    st.markdown("---")

    # Interactive charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        fig_donut = create_fraud_distribution_chart(fraud_txns, legit_txns)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        fig_amt = create_amount_distribution_chart(df, column="Amount")
        st.plotly_chart(fig_amt, use_container_width=True)

    # Interactive charts row 2
    col_bottom_left, col_bottom_right = st.columns(2)

    with col_bottom_left:
        fig_time = create_time_series_chart(df, time_col="Time", class_col="Class" if "Class" in df.columns else "predicted_class")
        st.plotly_chart(fig_time, use_container_width=True)

    with col_bottom_right:
        if "risk_level" in df.columns:
            fig_risk = create_risk_category_chart(df)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            # If risk_level isn't computed yet, show informational prompt
            st.info("💡 To view risk tiers ('Low', 'Medium', 'High'), run predictions on the **Fraud Detection** page or train an ML model!")
            fig_amt2 = create_amount_distribution_chart(df[df["Class"] == 1] if "Class" in df.columns else df)
            st.plotly_chart(fig_amt2, use_container_width=True)
