"""Tests for the pandas cleaning pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn_project import preprocessing as pp


def test_clean_removes_duplicates(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    clean, log = pp.clean_pandas(df)
    assert log.duplicate_rows_removed == 1
    assert len(clean) == len(synthetic_df)


def test_clean_imputes_missing_numeric(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0:9, "total_spent_usd"] = np.nan
    clean, log = pp.clean_pandas(df)
    assert clean["total_spent_usd"].isna().sum() == 0
    assert log.imputed_cells.get("total_spent_usd") == 10


def test_clean_removes_invalid_target(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "churned"] = 7  # invalid label
    clean, log = pp.clean_pandas(df)
    assert log.rows_with_invalid_target_removed == 1
    assert set(clean["churned"].unique()).issubset({0, 1})


def test_clean_target_is_integer(synthetic_df):
    clean, _ = pp.clean_pandas(synthetic_df)
    assert clean["churned"].dtype.kind in ("i", "u")


def test_normalise_categories_trims_whitespace(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "country"] = "  France  "
    out = pp.normalise_categories(df)
    assert out.loc[0, "country"] == "France"
