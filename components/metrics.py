"""
BankGuard AI — Metric Display Component

Renders modern fintech KPI cards with dark theme aesthetics.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import THEME_COLORS


def display_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    icon: Optional[str] = "📊",
    color: Optional[str] = None,
) -> None:
    """Render a single styled fintech KPI metric card."""
    card_color = color or THEME_COLORS["accent"]
    delta_html = ""
    if delta:
        delta_html = f'<div style="font-size: 0.85rem; color: #A0AEC0; margin-top: 4px;">{delta}</div>'

    html = f"""
    <div style="
        background: linear-gradient(135deg, {THEME_COLORS['card_bg']} 0%, #152642 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid {card_color};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 12px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 500; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">{label}</span>
            <span style="font-size: 1.4rem;">{icon}</span>
        </div>
        <div style="font-size: 1.75rem; font-weight: 700; color: #FFFFFF; margin-top: 8px;">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def display_metric_row(metrics: List[Dict[str, Any]]) -> None:
    """Display an array of metric cards in responsive horizontal columns.

    Each dict in metrics should have:
    - label: str
    - value: Any
    - delta: Optional[str]
    - icon: Optional[str]
    - color: Optional[str]
    """
    if not metrics:
        return

    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            display_metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", "-"),
                delta=metric.get("delta"),
                icon=metric.get("icon", "📌"),
                color=metric.get("color"),
            )
