"""
Central configuration for the E-Commerce Customer Churn project.

Schema matches the Kaggle dataset
"E-Commerce Customer Churn 2025 - 10K"
https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k

Keeping all paths, column names and parameters in one place makes the
PySpark pipeline reproducible (Objective 2) and easy to adapt if the
confirmed Kaggle column list differs slightly from the assumptions here.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")

# Default location where the Kaggle CSV should be placed by the user.
RAW_CSV = os.path.join(RAW_DIR, "ecommerce_customer_churn_10k.csv")

# Cleaned data is written here by pipeline_01_data_cleaning.py (Parquet).
CLEAN_PARQUET = os.path.join(PROCESSED_DIR, "clean")

# ---------------------------------------------------------------------------
# Schema  (real columns from the Kaggle dataset)
# ---------------------------------------------------------------------------
# Target column (1 = churned, 0 = retained).
TARGET = "churned"

# Identifier column dropped before modelling.
ID_COL = "customer_id"

# Date columns -> parsed to timestamps during cleaning.
DATE_COLS = ["signup_date", "last_purchase_date"]

# Numeric features that may contain missing values and are median-imputed.
NUMERIC_IMPUTE_COLS = [
    "age",
    "total_spent_usd",
    "total_orders",
    "days_since_last_purchase",
]

# All numeric feature columns used for EDA / modelling.
NUMERIC_COLS = [
    "age",
    "is_premium_member",
    "total_spent_usd",
    "total_orders",
    "days_since_last_purchase",
]

# Engineered numeric feature(s) added during cleaning (account tenure in days).
ENGINEERED_NUMERIC_COLS = ["account_age_days"]

# Categorical feature columns (string-indexed before modelling).
CATEGORICAL_COLS = [
    "gender",
    "country",
    "device_type",
    "favorite_category",
]

# Known inconsistent category spellings -> canonical form.
# (None known for this dataset; left here so the cleaning step is generic.)
CATEGORY_NORMALISATION: dict[str, dict[str, str]] = {}

# ---------------------------------------------------------------------------
# RFM (Objective 4) -- TRUE RFM, this dataset has real dates & orders/spend
# ---------------------------------------------------------------------------
# Recency   = days_since_last_purchase  (lower = more recent = better)
# Frequency = total_orders              (higher = better)
# Monetary  = total_spent_usd           (higher = better)
RFM_RECENCY = "days_since_last_purchase"
RFM_FREQUENCY = "total_orders"
RFM_MONETARY = "total_spent_usd"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
