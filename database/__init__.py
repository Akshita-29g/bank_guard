"""
BankGuard AI — Database Package

Provides the database manager, schema initialisation, and repository layer
for persisting transactions, analysis history, and model metrics.
"""

from database.db import DatabaseManager
from database.repository import TransactionRepository

__all__ = ["DatabaseManager", "TransactionRepository"]
