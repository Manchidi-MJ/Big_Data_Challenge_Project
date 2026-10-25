"""
Generate a SYNTHETIC sample dataset matching the schema of the Kaggle
'E-Commerce Customer Churn 2025 - 10K' dataset.

Real columns:
    customer_id, age, gender, country, device_type, favorite_category,
    is_premium_member, total_spent_usd, total_orders,
    days_since_last_purchase, signup_date, last_purchase_date, churned

Purpose: let the whole pipeline run end-to-end (and be tested) without the
real CSV. The churn signal intentionally depends on recency, frequency,
spend and membership so EDA and the model produce interpretable results.
Replace data/raw/ecommerce_customer_churn_10k.csv with the real Kaggle file
for the final run -- no code changes needed.

Usage:
    python -m src.generate_sample_data --rows 10000
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config


def generate(rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(38, 12, size=rows).clip(18, 80).round().astype(int)
    gender = rng.choice(["Male", "Female", "Other"], size=rows, p=[0.48, 0.48, 0.04])
    country = rng.choice(
        ["USA", "UK", "India", "Germany", "Canada", "Australia"],
        size=rows, p=[0.3, 0.18, 0.22, 0.1, 0.1, 0.1],
    )
    device = rng.choice(
        ["Mobile", "Desktop", "Tablet"], size=rows, p=[0.55, 0.33, 0.12]
    )
    fav_cat = rng.choice(
        ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books"],
        size=rows, p=[0.26, 0.24, 0.16, 0.14, 0.12, 0.08],
    )
    is_premium = rng.binomial(1, 0.3, size=rows)

    total_orders = rng.poisson(8, size=rows).clip(1, 60)
    # Spend scales with order count plus noise.
    total_spent = (total_orders * rng.normal(55, 20, size=rows)).clip(5, None).round(2)
    days_since_last = rng.gamma(shape=2.0, scale=30.0, size=rows).round().clip(0, 365).astype(int)

    # Dates: signup 0-1000 days ago; last purchase = today - days_since_last.
    today = datetime(2025, 1, 1)
    signup_offset = rng.integers(30, 1000, size=rows)
    signup_date = [today - timedelta(days=int(d)) for d in signup_offset]
    last_purchase_date = [today - timedelta(days=int(d)) for d in days_since_last]

    # Churn probability driven by interpretable risk factors.
    logit = (
        -1.2
        + 0.010 * days_since_last        # long time since purchase -> churn
        - 0.09 * total_orders            # frequent buyers stay
        - 0.0010 * total_spent           # high spenders stay
        - 0.6 * is_premium               # premium members stay
        + 0.006 * (age - 38)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(100001, 100001 + rows),
            "age": age,
            "gender": gender,
            "country": country,
            "device_type": device,
            "favorite_category": fav_cat,
            "is_premium_member": is_premium,
            "total_spent_usd": total_spent,
            "total_orders": total_orders,
            "days_since_last_purchase": days_since_last,
            "signup_date": [d.strftime("%Y-%m-%d") for d in signup_date],
            "last_purchase_date": [d.strftime("%Y-%m-%d") for d in last_purchase_date],
            "churned": churned,
        }
    )

    # Inject light, realistic missingness.
    for col in ["age", "total_spent_usd", "total_orders", "days_since_last_purchase"]:
        mask = rng.random(rows) < 0.03
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
    print(f"Churn rate: {df['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
