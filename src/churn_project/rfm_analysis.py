"""RFM segmentation (Phase 4).

This dataset provides customer-level summary features, not transaction logs, so
we use TRUE RFM derived from real fields:

    Recency   = days_since_last_purchase   (LOWER is better -> higher R score)
    Frequency = total_orders               (HIGHER is better)
    Monetary  = total_spent_usd            (HIGHER is better)

Scoring: each dimension is binned into up to 5 groups by quantile. Quantile
boundaries can tie when a value is very common; we therefore use rank-based
binning (``pd.qcut`` on ranks with ``duplicates='drop'``) and gracefully fall
back to fewer bins when the data cannot support 5 distinct groups. The number of
bins actually used per dimension is reported so the scoring is transparent.

Segment labels are assigned from rule-based score combinations. These labels are
an interpretive convenience, NOT a scientifically established taxonomy for this
dataset — the rules are documented below and in the report.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, Config


def _score_higher_better(series: pd.Series, n_bins: int) -> Tuple[pd.Series, int]:
    """Bin so that HIGHER values get HIGHER scores (1..k). Returns (scores, k)."""
    ranks = series.rank(method="first")
    try:
        binned = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        binned = pd.Series(np.zeros(len(series)), index=series.index)
    k = int(pd.Series(binned).nunique())
    scores = pd.Series(binned, index=series.index).astype(int) + 1
    return scores, k


def _score_lower_better(series: pd.Series, n_bins: int) -> Tuple[pd.Series, int]:
    """Bin so that LOWER values get HIGHER scores (1..k). Returns (scores, k)."""
    # Negate so lower raw value -> higher rank -> higher score.
    scores, k = _score_higher_better(-series.astype(float), n_bins)
    return scores, k


def compute_rfm(df: pd.DataFrame, config: Config = CONFIG) -> Tuple[pd.DataFrame, Dict]:
    """Compute RFM values, scores and segments per customer.

    Returns (rfm_df, meta) where meta documents the fields and bins used.
    """
    r_col = config.rfm["recency_col"]
    f_col = config.rfm["frequency_col"]
    m_col = config.rfm["monetary_col"]
    n_bins = int(config.rfm["n_bins"])

    for col in (r_col, f_col, m_col):
        if col not in df.columns:
            raise KeyError(f"RFM column '{col}' not found in data.")

    rfm = pd.DataFrame(
        {
            config.id_col: df[config.id_col].values,
            "recency": df[r_col].values,
            "frequency": df[f_col].values,
            "monetary": df[m_col].values,
        }
    )
    if config.target in df.columns:
        rfm[config.target] = df[config.target].values

    r_scores, r_k = _score_lower_better(rfm["recency"], n_bins)
    f_scores, f_k = _score_higher_better(rfm["frequency"], n_bins)
    m_scores, m_k = _score_higher_better(rfm["monetary"], n_bins)

    rfm["R"] = r_scores.values
    rfm["F"] = f_scores.values
    rfm["M"] = m_scores.values
    rfm["RFM_score"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["RFM_cell"] = (
        rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    )
    rfm["segment"] = rfm.apply(lambda row: assign_segment(row, r_k, f_k, m_k), axis=1)

    meta = {
        "recency_field": r_col,
        "frequency_field": f_col,
        "monetary_field": m_col,
        "n_bins_requested": n_bins,
        "n_bins_used": {"R": r_k, "F": f_k, "M": m_k},
        "recency_direction": "lower days_since_last_purchase -> higher R score",
        "segment_rules": _segment_rule_doc(),
    }
    return rfm, meta


def assign_segment(row: pd.Series, r_k: int, f_k: int, m_k: int) -> str:
    """Rule-based segment label from R/F/M scores.

    Thresholds are expressed relative to the actual number of bins used per
    dimension so the rules still make sense if a dimension could not be split
    into 5 groups.
    """
    r, f, m = row["R"], row["F"], row["M"]
    r_hi = r >= max(r_k - 1, 1)          # recent
    r_lo = r <= 2
    fm = (f + m) / 2.0
    fm_hi = fm >= (max(f_k, m_k) - 1)
    fm_mid = fm >= (max(f_k, m_k) / 2.0)

    if r_hi and fm_hi:
        return "Champions"
    if r_hi and fm_mid:
        return "Loyal Customers"
    if r_hi and not fm_mid:
        return "Potential Loyalists"
    if r_lo and fm_hi:
        return "At Risk"
    if r_lo and not fm_hi:
        return "Needs Attention"
    return "About to Sleep"


def _segment_rule_doc() -> Dict[str, str]:
    return {
        "Champions": "recent (high R) AND high frequency+monetary",
        "Loyal Customers": "recent AND mid frequency+monetary",
        "Potential Loyalists": "recent AND low frequency+monetary",
        "At Risk": "not recent (low R) AND high frequency+monetary",
        "Needs Attention": "not recent AND low frequency+monetary",
        "About to Sleep": "mid recency, everything else",
    }


def segment_summary(rfm: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Per-segment size, average RFM values and churn rate."""
    agg = {
        "recency": "mean",
        "frequency": "mean",
        "monetary": "mean",
        config.id_col: "count",
    }
    has_target = config.target in rfm.columns
    if has_target:
        agg[config.target] = "mean"
    out = rfm.groupby("segment").agg(agg).reset_index()
    rename = {
        config.id_col: "n_customers",
        "recency": "avg_recency_days",
        "frequency": "avg_frequency",
        "monetary": "avg_monetary_usd",
    }
    if has_target:
        rename[config.target] = "churn_rate"
    out = out.rename(columns=rename)
    out["pct_customers"] = (out["n_customers"] / len(rfm) * 100).round(2)
    num_cols = [c for c in out.columns if c not in ("segment",)]
    out[num_cols] = out[num_cols].round(4)
    return out.sort_values("n_customers", ascending=False).reset_index(drop=True)
