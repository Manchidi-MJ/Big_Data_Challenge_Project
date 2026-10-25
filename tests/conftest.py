"""Shared pytest fixtures.

The ``synthetic_df`` fixture is a SMALL, clearly-labelled synthetic dataset used
ONLY to test software behaviour. It is NOT the real Kaggle data and must never be
used to make claims about real customer behaviour. Its schema matches the
verified real schema so the same code paths are exercised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """200-row synthetic dataset matching the verified schema.

    Churn is defined by the same snapshot rule as the real data
    (days_since_last_purchase > 90) so leakage tests are meaningful.
    """
    rng = np.random.default_rng(42)
    n = 200
    days_since = rng.integers(1, 400, size=n)
    signup = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    account_age = rng.integers(10, 500, size=n)
    last_purchase = signup + pd.to_timedelta(account_age, unit="D")
    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:05d}" for i in range(n)],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "last_purchase_date": last_purchase.strftime("%Y-%m-%d"),
            "days_since_last_purchase": days_since,
            "account_age_days": account_age,
            "country": rng.choice(["United States", "France", "Japan", "Germany"], n),
            "age": rng.integers(18, 70, size=n),
            "gender": rng.choice(["Male", "Female", "Other"], n),
            "device_type": rng.choice(["Mobile", "Tablet", "Desktop"], n),
            "favorite_category": rng.choice(["Groceries", "Beauty", "Fashion", "Home & Garden"], n),
            "total_orders": rng.integers(1, 100, size=n),
            "total_spent_usd": rng.uniform(5, 1000, size=n).round(2),
            "avg_order_value": rng.uniform(5, 300, size=n).round(2),
            "is_premium_member": rng.integers(0, 2, size=n),
        }
    )
    df["churned"] = (df["days_since_last_purchase"] > 90).astype(int)
    return df
