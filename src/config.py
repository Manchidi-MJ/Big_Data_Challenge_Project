"""
Central configuration for the E-Commerce Customer Churn project.

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
RAW_CSV = os.path.join(RAW_DIR, "ecommerce_churn.csv")

# Cleaned data is written here by 01_data_cleaning.py (Parquet keeps types).
CLEAN_PARQUET = os.path.join(PROCESSED_DIR, "clean")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Target column (1 = churned, 0 = retained).
TARGET = "Churn"

# Identifier column dropped before modelling.
ID_COL = "CustomerID"

# Numeric features that commonly contain missing values in this dataset and
# are imputed with the median during cleaning.
NUMERIC_IMPUTE_COLS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
]

# All numeric feature columns used for EDA/modelling.
NUMERIC_COLS = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

# Categorical feature columns (string-indexed before modelling).
CATEGORICAL_COLS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Known inconsistent category spellings in this dataset family -> canonical.
CATEGORY_NORMALISATION = {
    "PreferredLoginDevice": {
        "Mobile Phone": "Phone",
        "Mobile": "Phone",
    },
    "PreferredPaymentMode": {
        "CC": "Credit Card",
        "COD": "Cash on Delivery",
    },
    "PreferedOrderCat": {
        "Mobile": "Mobile Phone",
    },
}

# ---------------------------------------------------------------------------
# RFM proxies (Objective 4)
# ---------------------------------------------------------------------------
# This churn dataset has no raw transaction timestamps, so classic RFM maps
# to the closest available proxies. This is stated openly in the report.
RFM_RECENCY = "DaySinceLastOrder"   # lower = more recent = better
RFM_FREQUENCY = "OrderCount"        # higher = better
RFM_MONETARY = "CashbackAmount"     # higher = better (spend proxy)

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
