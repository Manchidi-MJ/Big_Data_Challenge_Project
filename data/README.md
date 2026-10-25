# Data

## Dataset

**E-Commerce Customer Churn 2025 – 10K** (Kaggle)
<https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k>

- ~10,000 customer records, one row per customer.
- Target column: `churned` (1 = churned, 0 = retained).
- Features cover demographics, device/category preferences, membership,
  spend, order history and purchase dates.

## How to get it

1. Download the CSV from the Kaggle link above (sign in required).
2. Rename it to `ecommerce_customer_churn_10k.csv`.
3. Place it at `data/raw/ecommerce_customer_churn_10k.csv`.

The pipeline reads from that path. No code changes are needed.

## Don't have the file yet?

A synthetic sample with the **same schema** is generated automatically when
you run the pipeline, so everything runs end-to-end for development/testing:

```bash
python -m src.generate_sample_data --rows 10000
```

> The synthetic data is for pipeline testing only. Use the real Kaggle CSV
> for any reported results. Note in your write-up whether the final dataset
> is real or synthetic (Objective 1).

## Columns

| Column | Type | Notes |
|---|---|---|
| `customer_id` | int | Identifier (dropped before modelling) |
| `age` | num | Customer age |
| `gender` | cat | Male / Female / Other |
| `country` | cat | Customer country |
| `device_type` | cat | Mobile / Desktop / Tablet |
| `favorite_category` | cat | Most-ordered category |
| `is_premium_member` | int | 1 if premium member |
| `total_spent_usd` | num | Lifetime spend — **RFM Monetary** |
| `total_orders` | num | Lifetime orders — **RFM Frequency** |
| `days_since_last_purchase` | num | Days since last order — **RFM Recency** |
| `signup_date` | date | Account creation date |
| `last_purchase_date` | date | Most recent order date |
| `churned` | int | **Target**: 1 = churned, 0 = retained |

Engineered during cleaning: `account_age_days` (from `signup_date`).

> **Confirm** these columns against the downloaded file and update
> `src/config.py` if the real names differ (Objective 1).

Raw and processed data are git-ignored and are **not** committed.
