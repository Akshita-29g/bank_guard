-- ============================================================
-- BankGuard AI — Database Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT UNIQUE,
    time_value REAL,
    amount REAL,
    predicted_class INTEGER,
    fraud_probability REAL,
    risk_level TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_type TEXT NOT NULL,
    dataset_name TEXT,
    total_transactions INTEGER,
    fraud_detected INTEGER,
    model_used TEXT,
    accuracy_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    roc_auc REAL,
    pr_auc REAL,
    trained_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_risk ON transactions(risk_level);
CREATE INDEX IF NOT EXISTS idx_transactions_predicted ON transactions(predicted_class);
