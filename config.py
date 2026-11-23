"""
BankGuard AI — Central Configuration Module

All application-wide constants, thresholds, feature definitions, page settings,
and theme colours are defined here so every other module imports from a single
source of truth.
"""

from __future__ import annotations

import os
from typing import Dict, List

# ──────────────────────────────────────────────
# Application metadata
# ──────────────────────────────────────────────
APP_NAME: str = "BankGuard AI"
APP_VERSION: str = "1.0.0"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42

# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────
TEST_SIZE: float = 0.2

# ──────────────────────────────────────────────
# Risk‑classification thresholds (probability)
#   p < low        → "Low Risk"
#   low ≤ p < med  → "Medium Risk"
#   p ≥ med        → "High Risk"
# ──────────────────────────────────────────────
RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.3,
    "medium": 0.7,
}

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATABASE_PATH: str = "data/bankguard.db"
MODEL_DIR: str = "models/"
DATA_DIR: str = "data/"

# ──────────────────────────────────────────────
# Feature / target schema
# ──────────────────────────────────────────────
FEATURE_COLUMNS: List[str] = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET_COLUMN: str = "Class"

# ──────────────────────────────────────────────
# Streamlit page configuration
# ──────────────────────────────────────────────
PAGE_CONFIG: Dict[str, str] = {
    "page_title": APP_NAME,
    "page_icon": "🛡️",
    "layout": "wide",
}

# ──────────────────────────────────────────────
# Theme / colour palette
# ──────────────────────────────────────────────
THEME_COLORS: Dict[str, str] = {
    "primary": "#1B2A4A",
    "secondary": "#2E4057",
    "accent": "#00D4AA",
    "danger": "#FF4B4B",
    "warning": "#FFA500",
    "success": "#00D4AA",
    "text": "#FFFFFF",
    "card_bg": "#1E3A5F",
}
