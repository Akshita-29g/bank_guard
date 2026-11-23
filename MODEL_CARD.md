# 📋 BankGuard AI — Model Card

## Model Overview
- **Model Name**: BankGuard AI Fraud Ensemble (Random Forest & Logistic Regression)
- **Version**: 1.0.0
- **Model Type**: Supervised Binary Classification
- **Primary Estimators**:
  1. `sklearn.ensemble.RandomForestClassifier` (100 estimators, balanced class weights)
  2. `sklearn.linear_model.LogisticRegression` (L-BFGS solver, balanced class weights, max 1,000 iterations)
- **Developer**: B.Tech ECE Intern Candidate for IndusInd Bank

---

## Intended Use
- **Primary Use**: Automated fraud scoring, anomaly triaging, and risk categorization of electronic credit card transactions.
- **Target Users**: Fraud operations specialists, risk compliance officers, financial intelligence analysts.
- **Out-of-Scope Uses**: Automated unappealable account termination or credit line denial without secondary human review.

---

## Training & Evaluation Data
- **Dataset Source**: Anonymized credit card transactions dataset by ULB Machine Learning Group / Kaggle.
- **Total Dimensions**: 30 input features (28 PCA components `V1`–`V28`, plus numerical `Amount` and elapsed `Time`).
- **Class Imbalance Ratio**: Approximately 0.17% positive (fraud) cases versus 99.83% negative (legitimate) cases.
- **Splitting Strategy**: 80% Training / 20% Testing with strict stratified sampling.

---

## Performance Metrics & Benchmark

Because fraud represents less than 1% of transactions, accuracy is rejected as a valid performance metric. Models are evaluated using:

| Metric | Target Priority | Strategic Banking Rationale |
|---|---|---|
| **PR-AUC** | Primary | Evaluates precision and recall dynamics without bias from the massive true negative class. |
| **Recall** | High | Minimizes false negatives (missed fraud), protecting bank and cardholder capital. |
| **Precision** | High | Minimizes false positives, preventing unwarranted transaction declines. |
| **ROC-AUC** | Secondary | Measures global discriminative capability across all potential decision thresholds. |

---

## Ethical Considerations & Bias Mitigation
1. **Customer Privacy**: Raw personal identifying information (PII) such as account numbers, social security IDs, customer names, and merchant names are excluded.
2. **Non-Discriminatory Features**: Features are mathematical principal components representing transaction dynamics, not socio-demographic indicators.
3. **Human-in-the-Loop Design**: Moderate fraud probabilities (30% to 70%) trigger step-up authentication (e.g., OTP or biometric challenge) rather than automatic rejection.
