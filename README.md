#  BankGuard AI: AI-Powered Banking Fraud Detection & Transaction Analytics Platform

![BankGuard AI Banner](https://img.shields.io/badge/BankGuard%20AI-v1.0.0-00D4AA?style=for-the-badge&logo=shield&logoColor=white)
![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-1B2A4A?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

> **Portfolio Showcase**: Built for technology, data engineering, artificial intelligence, and business intelligence internship evaluation at **IndusInd Bank**.

---

##  Executive Summary & Problem Statement

Financial institutions handle millions of digital transactions per day. Fraudulent transactions constitute a very small percentage (typically < 0.2%) of overall volume, but represent billions of dollars in enterprise losses, regulatory exposure, and customer distress.

Traditional rule-based fraud detection engines struggle with:
1. **Extreme Class Imbalance**: Legitimate transactions overwhelmingly outnumber fraud cases.
2. **Evolving Attack Vectors**: Static if-else rules cannot catch nuanced, multivariate behavioral shifts.
3. **High False Positive Rates**: Inconvenient card declines alienate high-value cardholders.

**BankGuard AI** is an enterprise-grade, end-to-end fraud intelligence application that leverages machine learning classification, real-time risk stratification (Low / Medium / High), business intelligence diagnostics, and lightweight transaction persistence in SQLite.

---

##  System Architecture

```
                                  USER (Browser)
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │   Streamlit Dashboard   │
                           │       (app.py)          │
                           └────────────┬────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  Data Ingestion  │           │   ML Inference   │           │   Analytics &    │
│  & Validation    │           │     Pipeline     │           │   BI Engine      │
│(utils/data_loader│           │  (ml/predict.py) │           │ (ui/insights.py) │
│ utils/validation)│           └────────┬─────────┘           └──────────────────┘
└────────┬─────────┘                    │
         │                              ▼
         │                     ┌──────────────────┐
         │                     │  Trained Models  │
         │                     │ (models/*.joblib)│
         │                     └────────┬─────────┘
         │                              │
         ▼                              ▼
┌────────────────────────────────────────────────────────┐
│             SQLite Persistent Storage                  │
│       (database/db.py & database/repository.py)        │
└────────────────────────────────────────────────────────┘
```

---

##  Key Features

- **Executive Analytics Dashboard**: Real-time KPI metric cards (Total Volume, Fraud Incidence %, High Risk Exposure) alongside Plotly charts (Class distribution donut, Amount density histogram, Temporal velocity curves).
- **Interactive Transaction Explorer**: Live transaction ledger with instant multi-facet filtering by transaction ID, fraud state, risk level, and ticket amount range with 1-click CSV export.
- **Robust Ingestion & Schema Validation**: Automated schema inspection against Kaggle credit card schemas (`Time`, `Amount`, `V1`–`V28`), with duplicate handling, missing value imputation, and a zero-dependency synthetic transaction generator.
- **Machine Learning Benchmark Suite**: Comparative evaluation of **Random Forest** and **Logistic Regression** using metrics designed for rare events: **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC** (no misleading accuracy metrics).
- **Configurable Risk Tiering**: Automatic stratification into **Low Risk** (< 30%), **Medium Risk** (30%–70%), and **High Risk** (> 70%) to prioritize fraud operations workflows.
- **SQLite Audit Trail**: Automatic database logging of scored transactions and model evaluation history with SQL injection prevention.

---

##  Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | Streamlit | Responsive dark-theme fintech dashboard |
| **Data Processing** | Pandas & NumPy | High-performance dataframe operations and numerical transformations |
| **Machine Learning** | Scikit-Learn | Balanced Random Forest, Logistic Regression, and standard scalers |
| **Visualizations** | Plotly Express & Graph Objects | Interactive, responsive financial charts |
| **Database** | SQLite 3 | Lightweight ACID-compliant transaction persistence |
| **Model Persistence** | Joblib | Efficient serialization of estimators and transformers |
| **Testing** | Pytest | Unit testing for pipeline, preprocessing, and DB operations |

---

##  Project Structure

```
BankGuard-AI/
│
├── app.py                     # Main Streamlit application entrypoint
├── config.py                  # Central configuration (thresholds, paths, colors)
├── requirements.txt           # Python dependency specification
├── README.md                  # Comprehensive project documentation
├── PROJECT_ARCHITECTURE.md    # Detailed technical architecture & contracts
├── MODEL_CARD.md              # Machine learning model metadata & ethics
├── .gitignore                 # Version control exclusion rules
│
├── data/
│   ├── README.md              # Dataset download & synthetic data guidance
│   └── bankguard.db           # SQLite database (auto-created at runtime)
│
├── ml/
│   ├── __init__.py
│   ├── preprocessing.py       # Scaler, imputer, and feature matrix generator
│   ├── train.py               # Balanced Random Forest & Logistic Regression trainers
│   ├── predict.py             # High-level inference engine with risk tiering
│   ├── evaluate.py            # Precision, Recall, F1, ROC-AUC, PR-AUC diagnostics
│   └── model_utils.py         # End-to-end training pipeline orchestrator
│
├── ui/
│   ├── __init__.py
│   ├── dashboard.py           # Executive KPI overview & macro trends
│   ├── transaction_explorer.py# Ledger search, dynamic filtering & CSV download
│   ├── fraud_detection.py     # CSV upload, validation & batch scoring
│   ├── model_performance.py   # Confusion matrices, ROC curves & feature importance
│   └── insights.py            # Data-driven financial analytics & recommendations
│
├── components/
│   ├── __init__.py
│   ├── charts.py              # Reusable dark-themed Plotly charts
│   ├── metrics.py             # Styled fintech KPI metric cards
│   └── tables.py              # Paginated transaction data tables
│
├── database/
│   ├── __init__.py
│   ├── db.py                  # Thread-safe SQLite connection & context manager
│   ├── schema.sql             # DDL table schemas and index definitions
│   └── repository.py          # Data Access Object (DAO) for transactions & metrics
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # Currency formatters, risk categorizer, UUID generator
│   ├── validation.py          # Schema validation & dataset integrity checks
│   └── data_loader.py         # CSV file loader and realistic synthetic data generator
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures and mock objects
│   ├── test_data_loader.py    # Ingestion & synthetic data tests
│   ├── test_preprocessing.py  # Scaling & imputation tests
│   ├── test_prediction.py     # Inference & thresholding tests
│   └── test_database.py       # SQLite CRUD & schema tests
│
└── models/
    └── .gitkeep               # Directory for serialised .joblib models
```

---

##  Quick Start Guide (Windows Laptop)

### 1. Clone or Open Workspace
Open your terminal (PowerShell or Command Prompt) and navigate to the project directory:
```powershell
cd c:\Users\AKSHITA\OneDrive\Attachments\Bank_Guard
```

### 2. Verify or Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit application locally:
```powershell
python -m streamlit run app.py
```
*The application will automatically launch in your default web browser at `http://localhost:8501`.*

---

##  Dataset Instructions

### Option A: Public Kaggle Dataset (Recommended for Full Evaluation)
1. Download the **Credit Card Fraud Detection Dataset** from Kaggle:
   - URL: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` inside the `data/` directory, or upload it directly through the **Fraud Detection & Scoring** page in the UI.

### Option B: Built-in Realistic Synthetic Generator (Zero Downloads Required)
If you don't have the Kaggle dataset downloaded yet, BankGuard AI features an integrated synthetic data generator:
- Accurately replicates log-normal transaction amount distributions.
- Simulates PCA-transformed Gaussian feature vectors (`V1`–`V28`).
- Generates realistic class imbalance (~2% to 5% fraud).
- Generates sample datasets from 100 to 20,000 transactions with a single click.

---

##  Machine Learning Methodology

### 1. Data Leakage Prevention & Stratification
- Fraud events represent a small fraction of transactions. A naive random split would leave the test fold with too few or zero fraud cases.
- BankGuard AI enforces **Stratified Train-Test Splitting** (`test_size=0.2, stratify=y`), ensuring identical class distributions between training and testing sets.
- Feature scaling parameters are fitted **strictly on the training partition** and applied downstream to the test set to eliminate data leakage.

### 2. Handling Class Imbalance
- Standard accuracy metric is deceptive (a dummy model predicting all legitimate transactions achieves 99.8% accuracy while missing 100% of fraud).
- We configure `class_weight='balanced'` in both **Random Forest** and **Logistic Regression**, which automatically adjusts penalties inversely proportional to class frequencies.

### 3. Comprehensive Diagnostics
Evaluated on:
- **Precision**: Minimizes false alarms that annoy legitimate customers.
- **Recall (Sensitivity)**: Maximizes detection of actual fraud dollars.
- **F1-Score**: Balanced harmonic mean of precision and recall.
- **ROC-AUC & PR-AUC**: Measures separability under varying decision thresholds.

---

##  Running Unit Tests

Run the full automated test suite:
```powershell
python -m pytest tests/ -v
```

All tests execute in milliseconds using temporary in-memory fixtures to prevent test artifacts from polluting production databases.

---

## Security & Data Privacy Notice

> [!IMPORTANT]
> **Disclaimer**: This application is built for research, academic demonstration, and interview evaluation. The public dataset consists of anonymized PCA-transformed numerical features (`V1`–`V28`) from European cardholders. **No live, private, or actual customer data from IndusInd Bank is used, stored, or accessed.**

---

##  Future Enhancements

1. **Streaming Anomaly Detection**: Integration with Apache Kafka or RabbitMQ for real-time WebSocket transaction scoring.
2. **Explainable AI (XAI)**: Integration of TreeSHAP to provide localized feature explanation waterfalls per flagged transaction.
3. **Graph Neural Networks (GNN)**: Entity resolution and transaction graph analysis to detect organized fraud rings and synthetic identities.

---

## 📄 License
This project is open-source under the **MIT License**.
