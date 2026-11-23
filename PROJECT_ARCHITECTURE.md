# 🏛️ BankGuard AI — System Architecture & Data Contract Specification

This document details the architectural design, component interactions, and data contracts powering **BankGuard AI**.

---

## 1. Architectural Principles

1. **Separation of Concerns**: UI rendering (`ui/`, `components/`), machine learning (`ml/`), database persistence (`database/`), and utilities (`utils/`) are strictly isolated.
2. **Defensive Programming**: Failure in one subsystem (e.g., SQLite connection timeout or invalid CSV schema) never crashes the entire application.
3. **Reproducibility**: Random seeds (`RANDOM_SEED = 42`) are centrally enforced across splitting, model training, and synthetic generation.
4. **Data Integrity**: Preprocessing transformations are fit solely on training partitions to prevent data leakage.

---

## 2. Shared Data Contract (Schema Specification)

All application modules adhere to the following schema definition:

| Column Name | Type | Required | Description |
|---|---|---|---|
| `transaction_id` | `string` | Optional (Auto-generated if absent) | Unique transaction identifier (e.g. `TXN-A1B2C3D4`) |
| `Time` | `float` | Optional (Default 0.0) | Seconds elapsed since the first transaction in dataset |
| `Amount` | `float` | **Required** | Transaction monetary value in base currency |
| `V1` – `V28` | `float` | Optional (Generated if absent) | PCA-transformed anonymized behavioral feature vectors |
| `Class` | `integer` (0 or 1) | Optional for inference, **Required for training** | Ground truth label (`0` = Legitimate, `1` = Fraudulent) |
| `fraud_probability` | `float` [0.0 – 1.0] | Generated during prediction | Model predicted probability of fraud |
| `predicted_class` | `integer` (0 or 1) | Generated during prediction | Discrete binary classification decision |
| `risk_level` | `string` | Generated during prediction | Categorical risk tier (`Low Risk`, `Medium Risk`, `High Risk`) |

### Risk Level Stratification Rules:
- **Low Risk**: $\text{fraud\_probability} < 0.30$
- **Medium Risk**: $0.30 \le \text{fraud\_probability} < 0.70$
- **High Risk**: $\text{fraud\_probability} \ge 0.70$

*(Thresholds configurable via `config.py:RISK_THRESHOLDS`)*

---

## 3. Database Schema (SQLite DDL)

Transactions and operational audit events are persisted in `data/bankguard.db`:

```sql
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
```

---

## 4. End-to-End Information Flow

```
1. Ingestion:
   [CSV File / Synthetic Engine] ──> [utils.validation.validate_dataset]
                                                   │
2. Preprocessing:                                  ▼
   [Raw DataFrame] ───────────────> [ml.preprocessing.TransactionPreprocessor]
                                    - Imputes missing values with 0
                                    - StandardScaler on (Amount, Time)
                                                   │
3. Model Training / Scoring:                       ▼
   [Clean Feature Matrix X] ──────> [ml.train.train_model] OR [ml.predict.FraudPredictor]
                                    - Stratified split (80/20)
                                    - Balanced class weights
                                                   │
4. Enrichment & Triage:                            ▼
   [Predictions DataFrame] ───────> [utils.helpers.get_risk_category]
                                    - Generates risk_level & probabilities
                                                   │
5. Presentation & Persistence:                     ▼
   [Enriched Stream] ─────────────> [Streamlit Components] (Charts, Tables)
                                  └─> [database.repository.TransactionRepository]
```
