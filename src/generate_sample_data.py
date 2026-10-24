"""
Generate a SYNTHETIC sample dataset matching the schema of the Kaggle
'E-Commerce Customer Churn 2025 - 10K' dataset.

Purpose: let the entire pipeline run end-to-end (and be tested) without the
real CSV. The generated churn signal intentionally depends on tenure,
complaints, satisfaction and recency so that EDA and the model produce
sensible, interpretable results. Replace data/raw/ecommerce_churn.csv with
the real Kaggle file for the final run -- no code changes are needed.

Usage:
    python -m src.generate_sample_data --rows 10000
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import config


def generate(rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.gamma(shape=2.0, scale=6.0, size=rows).round().clip(0, 61)
    complain = rng.binomial(1, 0.28, size=rows)
    satisfaction = rng.integers(1, 6, size=rows)
    day_since_last = rng.gamma(shape=2.0, scale=3.0, size=rows).round().clip(0, 46)
    order_count = rng.poisson(2.5, size=rows).clip(1, 16)
    cashback = (rng.normal(180, 50, size=rows)).clip(0, 400).round(2)
    coupon = rng.poisson(1.5, size=rows).clip(0, 16)
    hike = rng.normal(15, 4, size=rows).clip(10, 26).round()
    hours = rng.normal(3, 1, size=rows).clip(0, 5).round()
    warehouse = rng.normal(15, 8, size=rows).clip(5, 127).round()
    devices = rng.integers(1, 7, size=rows)
    addresses = rng.integers(1, 12, size=rows)
    city_tier = rng.choice([1, 2, 3], size=rows, p=[0.65, 0.1, 0.25])

    # Churn probability driven by interpretable risk factors.
    logit = (
        -1.6
        - 0.07 * tenure
        + 1.1 * complain
        - 0.25 * (satisfaction - 3)
        + 0.06 * day_since_last
        - 0.004 * cashback
        - 0.08 * order_count
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(50001, 50001 + rows),
            "Churn": churn,
            "Tenure": tenure,
            "PreferredLoginDevice": rng.choice(
                ["Mobile Phone", "Phone", "Computer"], size=rows, p=[0.45, 0.2, 0.35]
            ),
            "CityTier": city_tier,
            "WarehouseToHome": warehouse,
            "PreferredPaymentMode": rng.choice(
                ["Debit Card", "Credit Card", "CC", "E wallet", "COD", "UPI"],
                size=rows,
                p=[0.35, 0.2, 0.08, 0.17, 0.1, 0.1],
            ),
            "Gender": rng.choice(["Male", "Female"], size=rows, p=[0.6, 0.4]),
            "HourSpendOnApp": hours,
            "NumberOfDeviceRegistered": devices,
            "PreferedOrderCat": rng.choice(
                ["Laptop & Accessory", "Mobile Phone", "Mobile", "Fashion", "Grocery", "Others"],
                size=rows,
                p=[0.3, 0.2, 0.08, 0.22, 0.1, 0.1],
            ),
            "SatisfactionScore": satisfaction,
            "MaritalStatus": rng.choice(
                ["Single", "Married", "Divorced"], size=rows, p=[0.35, 0.5, 0.15]
            ),
            "NumberOfAddress": addresses,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": hike,
            "CouponUsed": coupon,
            "OrderCount": order_count,
            "DaySinceLastOrder": day_since_last,
            "CashbackAmount": cashback,
        }
    )

    # Inject realistic missingness into the columns Kaggle users know are sparse.
    for col in ["Tenure", "WarehouseToHome", "HourSpendOnApp",
                "OrderAmountHikeFromlastYear", "CouponUsed",
                "OrderCount", "DaySinceLastOrder"]:
        mask = rng.random(rows) < 0.04
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data.")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=config.RAW_CSV)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(rows=args.rows, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} synthetic rows and {df.shape[1]} columns to {args.out}")
    print(f"Churn rate: {df['Churn'].mean():.1%}")


if __name__ == "__main__":
    main()
