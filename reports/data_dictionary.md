# Data dictionary

**Dataset:** E-Commerce Customer Churn 2025 – 10K
**Source:** Kaggle — <https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k>
**Verified:** 10,000 rows × 15 columns, one row per customer, no missing values,
no duplicate rows or customer IDs.

> This dictionary describes the **actual verified columns** in the downloaded
> CSV. It differs from the provisional column list in the original brief (which
> listed fields such as `total_returns`, `coupon_used`, `city_tier`,
> `preferred_payment_method`, `is_active`, `recency`, `frequency`, `tenure_days`
> that are **not present**). The schema below is authoritative.

| # | Column | Type | Meaning | Role | Modelling considerations |
|---|--------|------|---------|------|--------------------------|
| 1 | `customer_id` | string | Unique customer identifier (e.g. `CUST007563`) | Identifier | **Excluded** from features; used for joining/identification only |
| 2 | `signup_date` | date | Date the customer registered | Derived-only | Not used as a raw feature; used to derive tenure |
| 3 | `last_purchase_date` | date | Date of the customer's most recent purchase | **Leakage** | Encodes recency → defines churn; excluded from the primary model |
| 4 | `days_since_last_purchase` | int | Days between last purchase and the snapshot date | **Leakage / target-defining** | `churned == (days_since > 90)` exactly; excluded from the primary (Experiment B) model |
| 5 | `account_age_days` | int | Customer tenure = `last_purchase_date − signup_date` (verified 100% match) | Feature | Tenure proxy; modest negative correlation with churn (−0.12) |
| 6 | `country` | categorical | Customer country (United States, France, Japan, Germany, …) | Feature | One-hot encoded |
| 7 | `age` | int | Customer age in years | Feature | Range 18–69 observed; standardised |
| 8 | `gender` | categorical | Male / Female / Other | Feature | One-hot encoded |
| 9 | `device_type` | categorical | Mobile / Tablet / Desktop | Feature | One-hot encoded |
| 10 | `favorite_category` | categorical | Preferred product category (Groceries, Beauty, Fashion, Home & Garden, …) | Feature | One-hot encoded |
| 11 | `total_orders` | int | Total number of orders placed | Feature (RFM-Frequency) | Used as RFM Frequency |
| 12 | `total_spent_usd` | float | Total amount spent (USD) | Feature (RFM-Monetary) | Used as RFM Monetary; correlated 0.85 with `avg_order_value` |
| 13 | `avg_order_value` | float | Average value per order (USD) | Feature | Redundant with `total_spent_usd` (|r|=0.85) |
| 14 | `is_premium_member` | int (0/1) | Premium membership flag | Feature | Binary; near-zero correlation with churn |
| 15 | `churned` | int (0/1) | **Target.** 1 = churned, 0 = retained | Target | Defined by the recency rule `days_since_last_purchase > 90` |

## Verified class distribution

| Class | Label | Count | Proportion |
|-------|-------|-------|------------|
| 0 | Retained | 6,229 | 62.29% |
| 1 | Churned | 3,771 | 37.71% |

Moderate class imbalance (majority:minority ≈ 1.65:1).

## Key verified relationships

* **Target definition (leakage):** every retained customer has
  `days_since_last_purchase ∈ [1, 90]`; every churned customer has
  `days_since_last_purchase ∈ [91, 719]`. A threshold at 90 reproduces `churned`
  with **100% accuracy**.
* **Tenure identity:** `account_age_days = last_purchase_date − signup_date`
  (100% of rows).
* **Feature redundancy:** `total_spent_usd` and `avg_order_value` correlate at
  0.85.
* **Everything else:** all other numeric features correlate with `churned` at
  |r| < 0.12.
