# BankGuard AI — Data Directory

## Overview

This directory stores datasets used by the BankGuard AI fraud-detection platform.

## Expected Dataset Format

BankGuard AI expects a **CSV file** with the following schema (modelled after the
[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)):

| Column   | Type    | Description |
|----------|---------|-------------|
| `Time`   | float   | Seconds elapsed since the first transaction in the dataset. |
| `V1`–`V28` | float | Principal components obtained via PCA (anonymised features). |
| `Amount` | float   | Transaction amount in the original currency. |
| `Class`  | int (0/1) | Target label — **0 = legitimate**, **1 = fraudulent**. |

## How to Add Your Own Data

1. Place your CSV file in this directory (e.g. `data/creditcard.csv`).
2. Ensure the column names match the schema above.
3. The application will auto-detect and load it.

> **Note:** CSV files in this directory are excluded from version control via
> `.gitignore` to prevent accidental commits of sensitive financial data.

## Synthetic / Demo Data

If no CSV file is present the application can generate realistic synthetic
transaction data on the fly via `utils/data_loader.generate_synthetic_data()`.
This is useful for demos, testing, and development.
