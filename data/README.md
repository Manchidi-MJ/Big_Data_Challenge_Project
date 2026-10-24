# Data

## Dataset

**E-Commerce Customer Churn 2025 – 10K** (Kaggle)
<https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k>

- ~10,000 customer records, one row per customer.
- Target column: `Churn` (1 = churned, 0 = retained).
- Features cover engagement/behaviour, account details, demographics and
  platform preferences.

## How to get it

1. Download the CSV from the Kaggle link above (sign in required).
2. Rename it to `ecommerce_churn.csv`.
3. Place it at `data/raw/ecommerce_churn.csv`.

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

## Expected columns

| Column | Type | Notes |
|---|---|---|
| `CustomerID` | int | Identifier (dropped before modelling) |
| `Churn` | int | **Target**: 1 = churned, 0 = retained |
| `Tenure` | num | Months with the platform |
| `PreferredLoginDevice` | cat | e.g. Phone, Computer |
| `CityTier` | int | 1 / 2 / 3 |
| `WarehouseToHome` | num | Distance warehouse→home |
| `PreferredPaymentMode` | cat | e.g. Credit Card, UPI, COD |
| `Gender` | cat | Male / Female |
| `HourSpendOnApp` | num | Hours spent on app/website |
| `NumberOfDeviceRegistered` | int | Devices registered |
| `PreferedOrderCat` | cat | Preferred order category |
| `SatisfactionScore` | int | 1–5 |
| `MaritalStatus` | cat | Single / Married / Divorced |
| `NumberOfAddress` | int | Saved addresses |
| `Complain` | int | 1 if complained recently |
| `OrderAmountHikeFromlastYear` | num | % order-value increase YoY |
| `CouponUsed` | num | Coupons used |
| `OrderCount` | num | Orders placed |
| `DaySinceLastOrder` | num | Days since last order |
| `CashbackAmount` | num | Cashback received (spend proxy) |

> **Confirm** these columns against the downloaded file and update
> `src/config.py` if the real names differ (Objective 1 / "points to check").

Raw and processed data are git-ignored and are **not** committed.
