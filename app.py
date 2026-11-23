"""
BankGuard AI — Main Streamlit Application Entrypoint

Fintech-grade Fraud Detection & Transaction Analytics Platform.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import APP_NAME, APP_VERSION, PAGE_CONFIG, THEME_COLORS

# Configure Streamlit page parameters
st.set_page_config(**PAGE_CONFIG)

# Apply fintech dark theme CSS
st.markdown(
    f"""
    <style>
        /* Base typography & palette */
        .stApp {{
            background-color: #0E1726;
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #0B1320;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        /* Card and container elements */
        .css-1r6slb0, .css-12w0qpk {{
            background-color: {THEME_COLORS['card_bg']};
            border-radius: 8px;
            padding: 16px;
        }}

        /* Buttons */
        div.stButton > button:first-child {{
            background-color: {THEME_COLORS['accent']};
            color: #0E1726;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.2rem;
            transition: all 0.2s ease;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #00BF9A;
            box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
        }}

        /* Headers */
        h1, h2, h3, h4 {{
            color: #FFFFFF !important;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}

        /* Metric widget styling */
        div[data-testid="stMetricValue"] {{
            color: {THEME_COLORS['accent']} !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stMetricLabel"] {{
            color: #94A3B8 !important;
            font-weight: 500 !important;
        }}

        /* Tables & dataframes */
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "predictor" not in st.session_state:
    st.session_state.predictor = None

# ─────────────────────────────────────────────────────────────
# Sidebar Navigation & System Telemetry
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown(f"## 🛡️ **{APP_NAME}**")
st.sidebar.caption(f"Enterprise Fraud Intelligence • v{APP_VERSION}")
st.sidebar.markdown("---")

navigation_page = st.sidebar.radio(
    "Navigation Console",
    options=[
        "Overview Dashboard",
        "Fraud Detection & Scoring",
        "Transaction Explorer",
        "ML Model Performance",
        "Analytical Insights",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 System Telemetry")

# Telemetry 1: Dataset Status
current_df = st.session_state.df
if current_df is not None and not current_df.empty:
    st.sidebar.success(f"📊 Dataset: **{len(current_df):,}** records active")
else:
    st.sidebar.info("📊 Dataset: None loaded")

# Telemetry 2: ML Model Status
current_predictor = st.session_state.predictor
if current_predictor is not None and current_predictor.is_ready():
    st.sidebar.success("🤖 Predictor: Armed & Loaded")
else:
    st.sidebar.warning("🤖 Predictor: Untrained")

# Telemetry 3: Persistence Database
try:
    from database.db import DatabaseManager

    db = DatabaseManager()
    db.initialize()
    st.sidebar.caption("💾 SQLite Store: Connected (`data/bankguard.db`)")
except Exception:
    st.sidebar.caption("💾 SQLite Store: Offline (fallback active)")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Disclaimer**: Designed for research, evaluation, and portfolio demonstration. "
    "Does not contain live IndusInd Bank data."
)

# ─────────────────────────────────────────────────────────────
# Page Routing
# ─────────────────────────────────────────────────────────────
data_state = {
    "df": st.session_state.df,
    "predictions": st.session_state.predictions,
    "model_trained": st.session_state.model_trained,
    "model_results": st.session_state.model_results,
    "predictor": st.session_state.predictor,
}

if navigation_page == "Overview Dashboard":
    from ui.dashboard import render_dashboard

    render_dashboard(data_state)

elif navigation_page == "Fraud Detection & Scoring":
    from ui.fraud_detection import render_fraud_detection

    render_fraud_detection(data_state)

elif navigation_page == "Transaction Explorer":
    from ui.transaction_explorer import render_transaction_explorer

    render_transaction_explorer(data_state)

elif navigation_page == "ML Model Performance":
    from ui.model_performance import render_model_performance

    render_model_performance(data_state)

elif navigation_page == "Analytical Insights":
    from ui.insights import render_insights

    render_insights(data_state)
