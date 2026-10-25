"""Tests for RFM scoring and segmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn_project import rfm_analysis as rfm


def test_rfm_scores_in_range(synthetic_df):
    out, meta = rfm.compute_rfm(synthetic_df)
    for col in ("R", "F", "M"):
        assert out[col].min() >= 1
        assert out[col].max() <= meta["n_bins_requested"]


def test_recency_direction_lower_is_better(synthetic_df):
    """Lower days_since_last_purchase must map to a HIGHER R score."""
    out, _ = rfm.compute_rfm(synthetic_df)
    merged = out.copy()
    # Correlation between raw recency (days) and R score should be negative.
    corr = np.corrcoef(merged["recency"], merged["R"])[0, 1]
    assert corr < 0


def test_frequency_direction_higher_is_better(synthetic_df):
    out, _ = rfm.compute_rfm(synthetic_df)
    corr = np.corrcoef(out["frequency"], out["F"])[0, 1]
    assert corr > 0


def test_tied_quantiles_do_not_crash():
    """A column of (almost) all identical values should not raise."""
    n = 100
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "days_since_last_purchase": [10] * n,       # all tied
            "total_orders": [5] * (n - 1) + [99],       # nearly all tied
            "total_spent_usd": np.linspace(1, 100, n),
            "churned": [0] * n,
        }
    )
    out, meta = rfm.compute_rfm(df)
    assert len(out) == n
    # Fewer than 5 bins expected for the tied columns.
    assert meta["n_bins_used"]["R"] >= 1


def test_segment_summary_covers_all_rows(synthetic_df):
    out, _ = rfm.compute_rfm(synthetic_df)
    summ = rfm.segment_summary(out)
    assert summ["n_customers"].sum() == len(synthetic_df)
    assert abs(summ["pct_customers"].sum() - 100.0) < 0.5
