"""
BankGuard AI — Model Performance & Evaluation Page

Comparative analytics for Logistic Regression, Random Forest, confusion matrices, and ROC-AUC.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MODEL_DIR
from ml.model_utils import run_training_pipeline, get_available_models, get_model_info
from ml.predict import FraudPredictor
from ml.evaluate import get_feature_importance
from components.charts import (
    create_model_comparison_chart,
    create_confusion_matrix_chart,
    create_roc_curve_chart,
    create_feature_importance_chart,
)
from utils.data_loader import generate_synthetic_data


def render_model_performance(data_state: Dict[str, Any]) -> None:
    """Render the Machine Learning evaluation and benchmarking cockpit."""
    st.title("🤖 ML Model Performance & Benchmark Suite")
    st.markdown(
        "Compare classification models on rare fraud events using precision, recall, F1-score, and ROC-AUC curves."
    )

    df: pd.DataFrame | None = st.session_state.get("df")

    # Training controls
    with st.expander("⚙️ Model Training Controls", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            st.markdown(
                """
                **Training Methodology**:
                - Stratified train-test split (80/20) preserving rare fraud ratios.
                - Standardized scaling on `Amount` and `Time`; pre-transformed `V1`-`V28`.
                - Balanced class weights to counteract heavy class imbalance.
                """
            )

        with c2:
            model_selection = st.selectbox(
                "Model Pipeline",
                options=["Train & Compare All (RF + LR)", "Random Forest Only", "Logistic Regression Only"],
                index=0,
            )

        with c3:
            st.write("")
            st.write("")
            train_btn = st.button("🏋️ Train New Model", type="primary", use_container_width=True)

    if train_btn:
        # Require a dataset with Class labels or synthesize one
        training_df = df
        if training_df is None or "Class" not in training_df.columns:
            st.info("ℹ️ No labeled dataset active. Generating a standard 2,000-sample balanced training set...")
            training_df = generate_synthetic_data(n_samples=2000, fraud_ratio=0.03, seed=42)
            st.session_state.df = training_df

        with st.spinner("Training models, performing stratified split, and computing metrics..."):
            try:
                results = run_training_pipeline(training_df, test_size=0.2, seed=42)
                st.session_state.model_results = results
                st.session_state.model_trained = True

                best_name = results["best_model_name"]
                st.session_state.predictor = FraudPredictor(
                    model=results["models"][best_name],
                    preprocessor=results["preprocessor"],
                )
                st.success(f"Training completed! Best performing model: **{best_name.replace('_', ' ').title()}**")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Inspect current model results
    model_results: Dict[str, Any] | None = st.session_state.get("model_results")

    if not model_results:
        # Check if saved models exist on disk
        available = get_available_models()
        if available:
            st.info(f"💡 Found {len(available)} saved model artifacts in `{MODEL_DIR}`. Train a new model above or explore saved weights.")
        else:
            st.info("👉 Click **Train New Model** above to train the fraud detection algorithms and view performance benchmarks.")
        return

    evaluations = model_results.get("evaluations", {})
    best_name = model_results.get("best_model_name", "random_forest")

    # Overview comparison chart
    st.subheader("1. Comparative Benchmark: Logistic Regression vs Random Forest")
    comp_fig = create_model_comparison_chart(evaluations)
    st.plotly_chart(comp_fig, use_container_width=True)

    # Detailed metrics for best model
    st.markdown("---")
    st.subheader(f"2. Best Model Diagnostics: {best_name.replace('_', ' ').title()}")

    best_eval = evaluations.get(best_name, {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{best_eval.get('precision', 0):.3f}", help="Of transactions flagged as fraud, how many were true fraud?")
    m2.metric("Recall", f"{best_eval.get('recall', 0):.3f}", help="Of all actual fraud, how many did the model catch?")
    m3.metric("F1-Score", f"{best_eval.get('f1', 0):.3f}", help="Harmonic mean of Precision and Recall.")
    m4.metric("ROC-AUC", f"{best_eval.get('roc_auc', 0):.3f}", help="Area under the Receiver Operating Characteristic curve.")
    m5.metric("PR-AUC", f"{best_eval.get('pr_auc', 0):.3f}", help="Precision-Recall Area under curve (crucial for imbalanced fraud data).")

    # Confusion Matrix & Feature Importance
    c_cm, c_fi = st.columns(2)

    with c_cm:
        cm = best_eval.get("confusion_matrix")
        if cm is not None:
            fig_cm = create_confusion_matrix_chart(cm)
            st.plotly_chart(fig_cm, use_container_width=True)

    with c_fi:
        # Get feature importance if model supports it
        best_model_obj = model_results.get("models", {}).get(best_name)
        if best_model_obj is not None:
            from config import FEATURE_COLUMNS
            fi_df = get_feature_importance(best_model_obj, FEATURE_COLUMNS)
            fig_fi = create_feature_importance_chart(fi_df, top_n=12)
            st.plotly_chart(fig_fi, use_container_width=True)

    # ROC curve chart if probabilities exist
    roc_data = best_eval.get("roc_curve")
    if roc_data:
        fpr, tpr, auc_val = roc_data
        fig_roc = create_roc_curve_chart(fpr, tpr, auc_val)
        st.plotly_chart(fig_roc, use_container_width=True)
