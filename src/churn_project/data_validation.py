"""Data validation and quality assessment (pandas).

Produces the Phase-1 data-quality report: shape, dtypes, missing values,
duplicates, invalid/impossible ranges, target distribution and a redundancy /
correlation scan. Nothing here deletes data — it only *reports*.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CONFIG, Config


class SchemaError(Exception):
    """Raised when required columns are missing from a dataframe."""


def validate_schema(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, List[str]]:
    """Check the dataframe against the verified schema.

    Returns a dict describing missing/unexpected columns. Raises ``SchemaError``
    if the required target column is absent (the project cannot proceed).
    """
    expected = set(config.all_expected_cols)
    actual = set(df.columns)
    report = {
        "missing_columns": sorted(expected - actual),
        "unexpected_columns": sorted(actual - expected),
    }
    if config.target not in df.columns:
        raise SchemaError(
            f"Target column '{config.target}' missing. Cannot continue. "
            f"Present columns: {sorted(actual)}"
        )
    return report


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages."""
    n = len(df)
    miss = df.isna().sum()
    pct = (miss / n * 100).round(3) if n else miss * 0
    return (
        pd.DataFrame({"missing_count": miss, "missing_pct": pct})
        .sort_values("missing_count", ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )


def duplicate_report(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, int]:
    """Count fully-duplicated rows and duplicated customer IDs."""
    out = {"duplicate_rows": int(df.duplicated().sum())}
    if config.id_col in df.columns:
        out["duplicate_customer_ids"] = int(df[config.id_col].duplicated().sum())
    return out


def invalid_value_report(df: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Flag impossible / out-of-range values for known numeric fields.

    Rules are conservative and documented; they report counts only.
    """
    checks = []

    def _count(mask: pd.Series) -> int:
        return int(mask.sum()) if mask is not None else 0

    rules = {
        "age": ("age < 0 or age > 120", lambda s: (s < 0) | (s > 120)),
        "days_since_last_purchase": ("negative", lambda s: s < 0),
        "account_age_days": ("negative", lambda s: s < 0),
        "total_orders": ("negative", lambda s: s < 0),
        "total_spent_usd": ("negative", lambda s: s < 0),
        "avg_order_value": ("negative", lambda s: s < 0),
        "is_premium_member": ("not in {0,1}", lambda s: ~s.isin([0, 1])),
        config.target: ("not in {0,1}", lambda s: ~s.isin([0, 1])),
    }
    for col, (desc, fn) in rules.items():
        if col in df.columns:
            checks.append(
                {"column": col, "rule": desc, "n_violations": _count(fn(df[col]))}
            )
    return pd.DataFrame(checks)


def numeric_summary(df: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Descriptive statistics for numeric columns."""
    cols = [c for c in config.numeric_cols if c in df.columns]
    return df[cols].describe().T.reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, pd.Series]:
    """Value counts for each categorical column."""
    out = {}
    for c in config.categorical_cols:
        if c in df.columns:
            out[c] = df[c].value_counts(dropna=False)
    return out


def target_distribution(df: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Class counts, proportions and imbalance ratio for the target."""
    vc = df[config.target].value_counts(dropna=False).sort_index()
    prop = (vc / len(df)).round(4)
    out = pd.DataFrame({"class": vc.index, "count": vc.values, "proportion": prop.values})
    return out


def correlation_report(df: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Pairwise Pearson correlation of numeric features + target.

    Used both for redundancy detection and as an input to the leakage audit.
    """
    cols = [c for c in config.numeric_cols if c in df.columns] + [config.target]
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr(numeric_only=True).round(4)


def redundancy_report(
    df: pd.DataFrame, config: Config = CONFIG, threshold: float = 0.95
) -> pd.DataFrame:
    """Highly correlated numeric feature pairs (|r| >= threshold)."""
    corr = correlation_report(df, config)
    feats = [c for c in corr.columns if c != config.target]
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "corr": round(float(r), 4)})
    return pd.DataFrame(pairs)


def build_quality_summary(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, object]:
    """Assemble a single dict summarising all data-quality checks."""
    schema = validate_schema(df, config)
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "schema_check": schema,
        "missing": missing_value_report(df),
        "duplicates": duplicate_report(df, config),
        "invalid_values": invalid_value_report(df, config),
        "numeric_summary": numeric_summary(df, config),
        "target_distribution": target_distribution(df, config),
        "redundancy": redundancy_report(df, config),
    }
