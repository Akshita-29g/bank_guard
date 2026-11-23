"""
BankGuard AI — Fraud Detection & Inference Page

Upload transaction batches, execute ML scoring, flag anomalies, and log to SQLite.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODEL_DIR
from utils.validation import validate_dataset, validate_prediction_input
from utils.data_loader import load_csv, generate_synthetic_data, prepare_display_data
from ml.predict import FraudPredictor
from ml.model_utils import get_available_models
from database.repository import TransactionRepository
from components.tables import display_prediction_results, display_download_button


def render_fraud_detection(data_state: Dict[str, Any]) -> None:
    """Render the Fraud Detection & Scoring interface."""
    st.title("🎯 Transaction Fraud Scoring & Anomaly Flagging")
    st.markdown(
        "Upload raw or pre-processed transaction streams to detect fraudulent patterns in real time."
    )

    # Section 1: Ingestion
    st.subheader("1. Ingest Transaction Stream")
    tab1, tab2 = st.tabs(["📁 Upload CSV Dataset", "⚡ Generate Synthetic Demo Stream"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload Credit Card Transactions CSV (Kaggle or compatible schema)",
            type=["csv"],
            help="File must contain 'Amount' and preferably 'Time' and 'V1'-'V28' features.",
        )
        if uploaded_file is not None:
            try:
                df_loaded = load_csv(uploaded_file)
                st.session_state.df = df_loaded
                st.success(f"Successfully loaded {len(df_loaded):,} transactions from `{uploaded_file.name}`.")
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")

    with tab2:
        st.write("Need sample data to evaluate BankGuard AI? Generate realistic synthetic banking transactions:")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_samples = st.number_input("Transaction Volume", min_value=100, max_value=20000, value=1000, step=100)
        with c2:
            fraud_rate = st.slider("Target Fraud Ratio (%)", min_value=0.5, max_value=15.0, value=2.5, step=0.5)
        with c3:
            st.write("")
            st.write("")
            if st.button("Generate Synthetic Data", type="primary"):
                synthetic_df = generate_synthetic_data(
                    n_samples=int(n_samples),
                    fraud_ratio=fraud_rate / 100.0,
                    seed=42,
                )
                st.session_state.df = synthetic_df
                st.success(f"Generated {len(synthetic_df):,} synthetic transactions with ~{fraud_rate}% fraud!")

    # Current working dataframe
    df: pd.DataFrame | None = st.session_state.get("df")

    if df is None or df.empty:
        st.info("👈 Please upload a CSV file or generate synthetic demo transactions above to proceed.")
        return

    # Section 2: Validation
    st.markdown("---")
    st.subheader("2. Dataset Schema Validation")
    val_res = validate_prediction_input(df)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Row Count", f"{val_res.get('row_count', len(df)):,}")
    v2.metric("Column Count", f"{val_res.get('column_count', len(df.columns)):,}")
    v3.metric("Schema Valid", "✅ Passed" if val_res.get("valid") else "❌ Issues Found")
    v4.metric("Ground Truth 'Class'", "Present" if "Class" in df.columns else "Absent (Unlabeled)")

    if val_res.get("warnings"):
        for w in val_res["warnings"]:
            st.warning(f"⚠️ {w}")
    if val_res.get("errors"):
        for err in val_res["errors"]:
            st.error(f"⛔ {err}")
        return

    # Section 3: Run Prediction
    st.markdown("---")
    st.subheader("3. Model Scoring & Inference")

    predictor: FraudPredictor | None = st.session_state.get("predictor")

    # Try auto-loading if model exists in models/
    if predictor is None:
        saved_models = get_available_models()
        rf_model = os.path.join(MODEL_DIR, "best_model.joblib")
        rf_prep = os.path.join(MODEL_DIR, "preprocessor.joblib")
        if os.path.exists(rf_model) and os.path.exists(rf_prep):
            try:
                predictor = FraudPredictor(model_path=rf_model, preprocessor_path=rf_prep)
                st.session_state.predictor = predictor
            except Exception:
                pass

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        run_pred_btn = st.button("🚀 Run Fraud Prediction", type="primary", use_container_width=True)

    with col_info:
        if predictor is not None and predictor.is_ready():
            st.success("🟢 ML Predictor is armed and ready.")
        else:
            st.warning("⚠️ No active model found. Visit **Model Performance** to train a model first, or click below to train one quickly!")

    if run_pred_btn:
        if predictor is None or not predictor.is_ready():
            # Automatically train a quick baseline model so the user never faces a blocker
            with st.spinner("Training quick baseline model to score transactions..."):
                from ml.model_utils import run_training_pipeline
                train_df = df if "Class" in df.columns else generate_synthetic_data(1000, 0.03, seed=42)
                train_res = run_training_pipeline(train_df, seed=42)
                best_name = train_res["best_model_name"]
                predictor = FraudPredictor(
                    model=train_res["models"][best_name],
                    preprocessor=train_res["preprocessor"],
                )
                st.session_state.predictor = predictor
                st.session_state.model_results = train_res

        with st.spinner("Executing fraud inference pipeline..."):
            try:
                scored_df = predictor.predict(df)
                st.session_state.df = scored_df
                st.session_state.predictions = scored_df
                st.success("Inference completed successfully!")

                # Attempt SQLite logging
                try:
                    repo = TransactionRepository()
                    saved_count = repo.save_transactions(scored_df)
                    fraud_count = int((scored_df["predicted_class"] == 1).sum()) if "predicted_class" in scored_df.columns else 0
                    repo.save_analysis(
                        analysis_type="batch_prediction",
                        dataset_name="user_batch",
                        total=len(scored_df),
                        fraud_count=fraud_count,
                        model_used="Active Classifier",
                    )
                except Exception as db_err:
                    # SQLite failure should never crash the app
                    st.caption(f"(Database logging skipped: {db_err})")

            except Exception as e:
                st.error(f"Inference error: {e}")

    # Section 4: Display Results
    if "predicted_class" in df.columns or "fraud_probability" in df.columns:
        st.markdown("---")
        display_prediction_results(df, title="4. Evaluated Fraud Results")

        # Export scored dataset
        display_download_button(
            df,
            filename="bankguard_scored_transactions.csv",
            label="📥 Download Full Scored Dataset (CSV)",
        )
