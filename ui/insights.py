"""
BankGuard AI — Business Intelligence & Analytical Insights Page

Data-driven analytical insights derived directly from active transaction streams.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import THEME_COLORS
from utils.helpers import format_currency, safe_divide


def render_insights(data_state: Dict[str, Any]) -> None:
    """Render dynamically computed business and operational fraud insights."""
    st.title("💡 Banking Fraud Analytics & Strategic Insights")
    st.markdown(
        "Quantitative findings computed directly from current transaction patterns to assist risk officers and investigators."
    )

    df: pd.DataFrame | None = st.session_state.get("df")

    if df is None or df.empty:
        st.warning("⚠️ No transaction dataset is currently loaded. Please load demo data or upload a CSV in **Fraud Detection**.")
        return

    # Use ground truth 'Class' or predicted 'predicted_class'
    target_col = "Class" if "Class" in df.columns else ("predicted_class" if "predicted_class" in df.columns else None)

    total_count = len(df)
    total_amount = df["Amount"].sum() if "Amount" in df.columns else 0.0

    st.subheader("1. Executive Summary")

    if target_col is not None:
        fraud_df = df[df[target_col] == 1]
        legit_df = df[df[target_col] == 0]

        fraud_count = len(fraud_df)
        fraud_rate = (fraud_count / total_count) * 100 if total_count else 0.0
        fraud_amount = fraud_df["Amount"].sum() if "Amount" in fraud_df.columns else 0.0
        legit_amount = legit_df["Amount"].sum() if "Amount" in legit_df.columns else 0.0
        avg_fraud_ticket = fraud_df["Amount"].mean() if fraud_count else 0.0
        avg_legit_ticket = legit_df["Amount"].mean() if len(legit_df) else 0.0

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Flagged Fraud Ratio", f"{fraud_rate:.2f}%", f"{fraud_count:,} events")
        i2.metric("Total Fraud Exposure", format_currency(fraud_amount), f"{safe_divide(fraud_amount, total_amount)*100:.1f}% of volume")
        i3.metric("Avg Fraud Ticket", format_currency(avg_fraud_ticket))
        i4.metric("Avg Legit Ticket", format_currency(avg_legit_ticket))
    else:
        st.info("ℹ️ Predictions not yet run on this dataset. Run fraud scoring to unlock full financial exposure insights.")

    st.markdown("---")
    st.subheader("2. Key Empirical Findings")

    c_left, c_right = st.columns(2)

    with c_left:
        # Amount bracket analysis
        st.markdown("#### 📊 Fraud Incidence by Amount Bracket")
        if "Amount" in df.columns and target_col is not None:
            # Create amount buckets
            bins = [0, 50, 200, 500, 1000, 5000, float("inf")]
            labels = ["$0–$50", "$50–$200", "$200–$500", "$500–$1,000", "$1k–$5k", "$5k+"]
            df_bucketed = df.copy()
            df_bucketed["bucket"] = pd.cut(df_bucketed["Amount"], bins=bins, labels=labels, right=False)

            bucket_stats = (
                df_bucketed.groupby("bucket", observed=False)
                .agg(
                    total=("Amount", "count"),
                    fraud_count=(target_col, "sum"),
                )
                .reset_index()
            )
            bucket_stats["fraud_rate_pct"] = (bucket_stats["fraud_count"] / bucket_stats["total"] * 100).fillna(0)

            fig_b = go.Figure(
                data=[
                    go.Bar(
                        x=bucket_stats["bucket"],
                        y=bucket_stats["fraud_rate_pct"],
                        marker_color=THEME_COLORS["accent"],
                        text=[f"{r:.1f}%" for r in bucket_stats["fraud_rate_pct"]],
                        textposition="auto",
                    )
                ]
            )
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 58, 95, 0.2)",
                font=dict(color="#FFFFFF"),
                xaxis_title="Transaction Bracket",
                yaxis_title="Fraud Incidence (%)",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_b, use_container_width=True)

            # Text summary
            max_risk_bucket = bucket_stats.sort_values(by="fraud_rate_pct", ascending=False).iloc[0]
            st.caption(
                f"📌 **Observation**: The `{max_risk_bucket['bucket']}` tier shows the highest proportion of fraudulent activity "
                f"at **{max_risk_bucket['fraud_rate_pct']:.2f}%**."
            )
        else:
            st.info("Amount distribution data requires transaction records with Amount feature.")

    with c_right:
        # Suspicious Time Activity
        st.markdown("#### 🕒 Temporal Fraud Velocity")
        if "Time" in df.columns and target_col is not None:
            # Bin time into 8 segments
            df_time = df.copy()
            df_time["time_bin"] = pd.qcut(df_time["Time"], q=6, labels=[f"T-Segment {i+1}" for i in range(6)], duplicates="drop")

            time_stats = (
                df_time.groupby("time_bin", observed=False)
                .agg(
                    total=("Time", "count"),
                    fraud_count=(target_col, "sum"),
                )
                .reset_index()
            )
            time_stats["fraud_rate_pct"] = (time_stats["fraud_count"] / time_stats["total"] * 100).fillna(0)

            fig_t = go.Figure(
                data=[
                    go.Scatter(
                        x=time_stats["time_bin"],
                        y=time_stats["fraud_rate_pct"],
                        mode="lines+markers",
                        line=dict(color=THEME_COLORS["danger"], width=3),
                        marker=dict(size=8, color="#FFFFFF"),
                    )
                ]
            )
            fig_t.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 58, 95, 0.2)",
                font=dict(color="#FFFFFF"),
                xaxis_title="Time Epoch",
                yaxis_title="Suspicious Rate (%)",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_t, use_container_width=True)

            peak_time = time_stats.sort_values(by="fraud_rate_pct", ascending=False).iloc[0]
            st.caption(
                f"📌 **Observation**: Peak velocity occurred during `{peak_time['time_bin']}` "
                f"with a **{peak_time['fraud_rate_pct']:.2f}%** anomaly frequency."
            )
        else:
            st.info("Time series insights require a 'Time' column.")

    st.markdown("---")
    st.subheader("3. Actionable Operational Recommendations")

    # Risk triage
    if "risk_level" in df.columns:
        high_r = len(df[df["risk_level"] == "High Risk"])
        med_r = len(df[df["risk_level"] == "Medium Risk"])
        low_r = len(df[df["risk_level"] == "Low Risk"])

        r1, r2, r3 = st.columns(3)
        with r1:
            st.error(f"🛑 **Immediate Review**: {high_r:,} transactions ({safe_divide(high_r, total_count)*100:.1f}%) require manual block or 2FA verification.")
        with r2:
            st.warning(f"⚠️ **Step-Up Challenge**: {med_r:,} transactions ({safe_divide(med_r, total_count)*100:.1f}%) warrant biometric or SMS OTP confirmation.")
        with r3:
            st.success(f"✅ **Straight-Through Processing**: {low_r:,} transactions ({safe_divide(low_r, total_count)*100:.1f}%) approved frictionless.")
    else:
        st.info("Run ML scoring in **Fraud Detection** to view automatic risk triage recommendations.")
