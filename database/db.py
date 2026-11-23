"""
BankGuard AI — Database Manager

Low-level SQLite connection and query helpers.  Reads ``schema.sql``
on first initialisation and creates all tables / indices automatically.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

from config import DATABASE_PATH

logger = logging.getLogger(__name__)

# Resolve the directory that contains *this* file so we can find schema.sql
_THIS_DIR = Path(__file__).resolve().parent
_SCHEMA_FILE = _THIS_DIR / "schema.sql"


class DatabaseManager:
    """Thread-aware SQLite helper with context-manager support."""

    # ------------------------------------------------------------------ #
    #  Construction / destruction
    # ------------------------------------------------------------------ #

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        db_path : str, optional
            Absolute or relative path to the SQLite database file.
            Falls back to ``config.DATABASE_PATH`` when not supplied.
        """
        self.db_path: str = db_path or DATABASE_PATH
        self._conn: Optional[sqlite3.Connection] = None

        # Make sure the parent directory exists (e.g. ``data/``).
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create data directory: %s", exc)

    # ------------------------------------------------------------------ #
    #  Connection handling
    # ------------------------------------------------------------------ #

    def connect(self) -> sqlite3.Connection:
        """Return the current connection or open a new one."""
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA foreign_keys=ON;")
                logger.info("Connected to database: %s", self.db_path)
            except sqlite3.Error as exc:
                logger.error("Failed to connect to database: %s", exc)
                raise
        return self._conn

    def close(self) -> None:
        """Close the database connection if it is open."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("Database connection closed: %s", self.db_path)
            except sqlite3.Error as exc:
                logger.warning("Error closing database: %s", exc)
            finally:
                self._conn = None

    # ------------------------------------------------------------------ #
    #  Schema bootstrap
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Read *schema.sql* and execute it.  Logs a warning on failure."""
        try:
            if not _SCHEMA_FILE.exists():
                logger.warning("Schema file not found: %s", _SCHEMA_FILE)
                return

            schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")
            conn = self.connect()
            conn.executescript(schema_sql)
            conn.commit()
            logger.info("Database schema initialised successfully.")
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Database initialisation failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------ #
    #  Query helpers
    # ------------------------------------------------------------------ #

    def execute(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> sqlite3.Cursor:
        """Execute a single parameterised statement and commit.

        Parameters
        ----------
        query : str
            SQL statement with ``?`` placeholders.
        params : tuple, optional
            Values bound to the placeholders.

        Returns
        -------
        sqlite3.Cursor
        """
        try:
            conn = self.connect()
            cursor = conn.execute(query, params or ())
            conn.commit()
            return cursor
        except sqlite3.Error as exc:
            logger.error("Query execution error: %s | query=%s", exc, query)
            raise

    def fetchall(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[sqlite3.Row]:
        """Return all rows for a SELECT statement.

        Parameters
        ----------
        query : str
            SQL SELECT with ``?`` placeholders.
        params : tuple, optional
            Bind values.

        Returns
        -------
        list[sqlite3.Row]
        """
        try:
            conn = self.connect()
            cursor = conn.execute(query, params or ())
            return cursor.fetchall()
        except sqlite3.Error as exc:
            logger.error("Fetchall error: %s | query=%s", exc, query)
            return []

    def fetchone(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> Optional[sqlite3.Row]:
        """Return the first row for a SELECT, or ``None``.

        Parameters
        ----------
        query : str
            SQL SELECT with ``?`` placeholders.
        params : tuple, optional
            Bind values.

        Returns
        -------
        sqlite3.Row or None
        """
        try:
            conn = self.connect()
            cursor = conn.execute(query, params or ())
            return cursor.fetchone()
        except sqlite3.Error as exc:
            logger.error("Fetchone error: %s | query=%s", exc, query)
            return None

    # ------------------------------------------------------------------ #
    #  Context-manager protocol
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "DatabaseManager":
        self.connect()
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    #  Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        status = "connected" if self._conn else "disconnected"
        return f"<DatabaseManager path={self.db_path!r} {status}>"
