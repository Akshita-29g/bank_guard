"""
BankGuard AI — Plotly Interactive Charts Component

Reusable charting functions configured with dark fintech styling.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import THEME_COLORS


def _get_dark_layout(title: str = "") -> dict:
    """Helper to return consistent dark theme layout dictionary."""
    return dict(
        title=dict(text=title, font=dict(color=THEME_COLORS["text"], size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 58, 95, 0.2)",
        font=dict(color=THEME_COLORS["text"], family="Inter, system-ui, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color=THEME_COLORS["text"]),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color=THEME_COLORS["text"]),
        ),
        legend=dict(
            bgcolor="rgba(30, 58, 95, 0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color=THEME_COLORS["text"]),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
    )


def create_fraud_distribution_chart(fraud_count: int, legit_count: int) -> go.Figure:
    """Create an interactive donut chart of legitimate vs fraudulent transactions."""
    labels = ["Legitimate", "Fraudulent"]
    values = [legit_count, fraud_count]
    colors = [THEME_COLORS["success"], THEME_COLORS["danger"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#0E192D", width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(color="#FFFFFF", size=13),
            )
        ]
    )

    layout = _get_dark_layout("Transaction Class Distribution")
    fig.update_layout(**layout)
    return fig


def create_amount_distribution_chart(df: pd.DataFrame, column: str = "Amount") -> go.Figure:
    """Create transaction amount distribution histogram comparing fraud and legitimate."""
    fig = go.Figure()

    if df.empty or column not in df.columns:
        fig.update_layout(**_get_dark_layout("Transaction Amount Distribution (No Data)"))
        return fig

    # Filter out extreme outliers for visualization clarity if needed
    p99 = df[column].quantile(0.99) if len(df) > 10 else df[column].max()
    sub_df = df[df[column] <= (p99 * 1.2)] if p99 > 0 else df

    if "Class" in sub_df.columns:
        legit = sub_df[sub_df["Class"] == 0][column]
        fraud = sub_df[sub_df["Class"] == 1][column]

        fig.add_trace(
            go.Histogram(
                x=legit,
                name="Legitimate",
                marker_color=THEME_COLORS["success"],
                opacity=0.6,
                nbinsx=40,
            )
        )
        if not fraud.empty:
            fig.add_trace(
                go.Histogram(
                    x=fraud,
                    name="Fraudulent",
                    marker_color=THEME_COLORS["danger"],
                    opacity=0.8,
                    nbinsx=40,
                )
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=sub_df[column],
                name="Amount",
                marker_color=THEME_COLORS["accent"],
                opacity=0.7,
                nbinsx=40,
            )
        )

    layout = _get_dark_layout("Transaction Amount Distribution (≤ 99th Percentile)")
    fig.update_layout(
        **layout,
        barmode="overlay",
        xaxis_title="Amount ($)",
        yaxis_title="Count",
    )
    return fig


def create_time_series_chart(
    df: pd.DataFrame, time_col: str = "Time", class_col: str = "Class"
) -> go.Figure:
    """Create a time-series or binned distribution showing transactions over time."""
    fig = go.Figure()

    if df.empty or time_col not in df.columns:
        fig.update_layout(**_get_dark_layout("Transaction Activity Over Time (No Data)"))
        return fig

    # Bin time into intervals (e.g. 30 bins)
    bins = min(50, max(10, len(df) // 20))
    counts, edges = np.histogram(df[time_col], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    fig.add_trace(
        go.Scatter(
            x=centers,
            y=counts,
            mode="lines+markers",
            name="Total Volume",
            line=dict(color=THEME_COLORS["accent"], width=2.5),
            marker=dict(size=4),
        )
    )

    if class_col in df.columns:
        fraud_df = df[df[class_col] == 1]
        if not fraud_df.empty:
            f_counts, _ = np.histogram(fraud_df[time_col], bins=edges)
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=f_counts,
                    name="Fraud Cases",
                    marker_color=THEME_COLORS["danger"],
                    opacity=0.8,
                )
            )

    layout = _get_dark_layout("Transaction Volume & Fraud Patterns Over Time")
    fig.update_layout(
        **layout,
        xaxis_title="Elapsed Time (Seconds)",
        yaxis_title="Number of Transactions",
    )
    return fig


def create_risk_category_chart(df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing transaction count across risk levels."""
    fig = go.Figure()

    if df.empty or "risk_level" not in df.columns:
        fig.update_layout(**_get_dark_layout("Risk Classification Breakdown (No Data)"))
        return fig

    risk_counts = df["risk_level"].value_counts().to_dict()
    categories = ["Low Risk", "Medium Risk", "High Risk"]
    counts = [risk_counts.get(cat, 0) for cat in categories]
    colors = [THEME_COLORS["success"], THEME_COLORS["warning"], THEME_COLORS["danger"]]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=13),
        )
    )

    layout = _get_dark_layout("Risk Classification Breakdown")
    fig.update_layout(
        **layout,
        xaxis_title="Assessed Risk Tier",
        yaxis_title="Count",
    )
    return fig


def create_confusion_matrix_chart(cm: np.ndarray, labels: Optional[List[str]] = None) -> go.Figure:
    """Create an interactive heatmap for confusion matrix."""
    if labels is None:
        labels = ["Legitimate (0)", "Fraudulent (1)"]

    z_text = [[str(val) for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=14, color="#FFFFFF"),
            colorscale=[
                [0.0, "#1E3A5F"],
                [0.5, "#2E4057"],
                [1.0, THEME_COLORS["accent"]],
            ],
            showscale=True,
        )
    )

    layout = _get_dark_layout("Confusion Matrix")
    layout["yaxis"].update(autorange="reversed")
    fig.update_layout(
        **layout,
        xaxis_title="Predicted Label",
        yaxis_title="Actual True Label",
    )
    return fig


def create_roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> go.Figure:
    """Create an interactive ROC curve chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {auc_score:.4f})",
            line=dict(color=THEME_COLORS["accent"], width=3),
        )
    )

    # Reference random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess",
            line=dict(color="rgba(255,255,255,0.4)", width=2, dash="dash"),
        )
    )

    layout = _get_dark_layout("Receiver Operating Characteristic (ROC)")
    fig.update_layout(
        **layout,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
    )
    return fig


def create_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create a horizontal bar chart showing top features by importance."""
    fig = go.Figure()

    if importance_df.empty or "feature" not in importance_df.columns:
        fig.update_layout(**_get_dark_layout("Feature Importance (No Data)"))
        return fig

    top_df = importance_df.head(top_n).iloc[::-1]  # Reverse for top-down display

    fig.add_trace(
        go.Bar(
            x=top_df["importance"],
            y=top_df["feature"],
            orientation="h",
            marker=dict(
                color=top_df["importance"],
                colorscale=[[0, THEME_COLORS["secondary"]], [1, THEME_COLORS["accent"]]],
            ),
            text=[f"{v:.4f}" for v in top_df["importance"]],
            textposition="auto",
            textfont=dict(color="#FFFFFF", size=11),
        )
    )

    layout = _get_dark_layout(f"Top {top_n} Predictive Features")
    fig.update_layout(
        **layout,
        xaxis_title="Importance Score / Absolute Coefficient",
        yaxis_title="Feature",
    )
    return fig


def create_model_comparison_chart(comparison_data: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create grouped bar chart comparing performance metrics across models."""
    fig = go.Figure()

    if not comparison_data:
        fig.update_layout(**_get_dark_layout("Model Evaluation Benchmark (No Data)"))
        return fig

    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]

    colors = [THEME_COLORS["accent"], THEME_COLORS["warning"], "#9C27B0", "#00BCD4"]

    for idx, (model_name, scores) in enumerate(comparison_data.items()):
        values = [scores.get(m, 0.0) for m in metrics]
        color = colors[idx % len(colors)]
        display_name = model_name.replace("_", " ").title()

        fig.add_trace(
            go.Bar(
                name=display_name,
                x=metric_labels,
                y=values,
                marker_color=color,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
                textfont=dict(color="#FFFFFF", size=11),
            )
        )

    layout = _get_dark_layout("Model Performance Comparison")
    layout["yaxis"].update(range=[0, 1.1], title="Score (0.0 – 1.0)")
    fig.update_layout(
        **layout,
        barmode="group",
        xaxis_title="Evaluation Metric",
    )
    return fig
