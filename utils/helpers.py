"""
BankGuard AI — General-Purpose Utility Functions

Reusable helpers for formatting, ID generation, risk classification, and time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict

import sys
import os

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RISK_THRESHOLDS


def format_currency(amount: float) -> str:
    """Format a numeric amount as a US-dollar string.

    Args:
        amount: The monetary value to format.

    Returns:
        A string such as ``"$1,234.56"`` or ``"-$50.00"`` for negatives.

    Examples:
        >>> format_currency(1234.5)
        '$1,234.50'
        >>> format_currency(-50)
        '-$50.00'
        >>> format_currency(0)
        '$0.00'
    """
    if amount < 0:
        return f"-${abs(amount):,.2f}"
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format a decimal fraction (0–1 scale **or** 0–100 scale) as a percentage.

    If *value* is in the range [0, 1] it is treated as a fraction and
    multiplied by 100.  Values > 1 are assumed to already be percentages.

    Args:
        value: The numeric value to format.

    Returns:
        A string such as ``"85.50%"``.

    Examples:
        >>> format_percentage(0.8549)
        '85.49%'
        >>> format_percentage(85.49)
        '85.49%'
    """
    if abs(value) <= 1.0:
        value *= 100.0
    return f"{value:.2f}%"


def get_risk_category(
    probability: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Classify a fraud probability into a human-readable risk category.

    Args:
        probability: Predicted fraud probability (0.0 – 1.0).
        thresholds:  Optional dict with ``"low"`` and ``"medium"`` keys.
                     Defaults to :pydata:`config.RISK_THRESHOLDS`.

    Returns:
        One of ``"Low Risk"``, ``"Medium Risk"``, or ``"High Risk"``.

    Examples:
        >>> get_risk_category(0.1)
        'Low Risk'
        >>> get_risk_category(0.5)
        'Medium Risk'
        >>> get_risk_category(0.9)
        'High Risk'
    """
    if thresholds is None:
        thresholds = RISK_THRESHOLDS

    low: float = thresholds.get("low", 0.3)
    medium: float = thresholds.get("medium", 0.7)

    if probability >= medium:
        return "High Risk"
    if probability >= low:
        return "Medium Risk"
    return "Low Risk"


def generate_transaction_id() -> str:
    """Generate a unique, human-friendly transaction identifier.

    The ID has the form ``TXN-XXXXXXXX`` where *X* is a hex character derived
    from a UUID-4.

    Returns:
        A string like ``"TXN-3A7F12BC"``.
    """
    short_id = uuid.uuid4().hex[:8].upper()
    return f"TXN-{short_id}"


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide *a* by *b*, returning *default* when *b* is zero.

    Args:
        a: Numerator.
        b: Denominator.
        default: Value returned when *b* is zero.

    Returns:
        ``a / b`` or *default*.

    Examples:
        >>> safe_divide(10, 3)
        3.3333333333333335
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=-1)
        -1
    """
    if b == 0:
        return default
    return a / b


def timestamp_now() -> str:
    """Return the current UTC time as an ISO-8601 formatted string.

    Returns:
        A string like ``"2024-06-15T14:30:00+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat()
