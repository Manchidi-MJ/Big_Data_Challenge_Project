"""Tests for schema and data-quality validation."""
from __future__ import annotations

import pandas as pd
import pytest

from churn_project import data_validation as dv
from churn_project.data_validation import SchemaError


def test_validate_schema_ok(synthetic_df):
    report = dv.validate_schema(synthetic_df)
    assert report["missing_columns"] == []
    assert report["unexpected_columns"] == []


def test_validate_schema_missing_target_raises(synthetic_df):
    df = synthetic_df.drop(columns=["churned"])
    with pytest.raises(SchemaError):
        dv.validate_schema(df)


def test_missing_value_report_counts(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0:4, "age"] = None
    rep = dv.missing_value_report(df)
    age_row = rep.loc[rep["column"] == "age"].iloc[0]
    assert age_row["missing_count"] == 5


def test_duplicate_report(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    rep = dv.duplicate_report(df)
    assert rep["duplicate_rows"] == 1
    assert rep["duplicate_customer_ids"] == 1


def test_invalid_value_report_flags_bad_age(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "age"] = 999
    df.loc[1, "total_spent_usd"] = -5
    rep = dv.invalid_value_report(df)
    age_viol = rep.loc[rep["column"] == "age", "n_violations"].iloc[0]
    spent_viol = rep.loc[rep["column"] == "total_spent_usd", "n_violations"].iloc[0]
    assert age_viol >= 1
    assert spent_viol >= 1


def test_target_distribution_sums_to_one(synthetic_df):
    td = dv.target_distribution(synthetic_df)
    assert abs(td["proportion"].sum() - 1.0) < 1e-6
    assert set(td["class"]).issubset({0, 1})


def test_build_quality_summary_keys(synthetic_df):
    q = dv.build_quality_summary(synthetic_df)
    for key in ("n_rows", "n_cols", "missing", "duplicates", "target_distribution"):
        assert key in q
    assert q["n_rows"] == len(synthetic_df)
