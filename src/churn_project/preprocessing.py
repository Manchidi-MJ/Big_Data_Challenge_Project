"""Cleaning / preprocessing logic shared by the pandas and PySpark paths.

Design choices (documented for the report):
* The verified dataset has **no missing values and no duplicates**, so cleaning
  here is mostly defensive: it will still correctly handle those cases if a
  future dataset version introduces them.
* We DO NOT drop rows silently. Any removal is counted and logged.
* ``customer_id`` is retained for identification/joining but excluded from
  modelling features downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CONFIG, Config


@dataclass
class CleaningLog:
    """Records what the cleaning step did (for the data-quality narrative)."""

    rows_before: int = 0
    rows_after: int = 0
    duplicate_rows_removed: int = 0
    rows_with_invalid_target_removed: int = 0
    imputed_cells: Dict[str, int] = field(default_factory=dict)
    normalised_categories: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "rows_with_invalid_target_removed": self.rows_with_invalid_target_removed,
            "imputed_cells": self.imputed_cells,
            "normalised_categories": self.normalised_categories,
            "notes": self.notes,
        }


def normalise_categories(df: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Trim whitespace and apply any configured canonical spellings."""
    df = df.copy()
    norm_map = config.schema.get("category_normalisation", {}) or {}
    for col in config.categorical_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            if col in norm_map:
                df[col] = df[col].replace(norm_map[col])
    return df


def clean_pandas(df: pd.DataFrame, config: Config = CONFIG) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean the dataframe and return (clean_df, log).

    Steps:
      1. Parse dates.
      2. Drop fully-duplicated rows (counted).
      3. Remove rows whose target is not in {0,1} (counted) — these cannot be
         used for supervised learning.
      4. Median-impute numeric NaNs; mode-impute categorical NaNs (counted).
      5. Normalise category strings.
      6. Derive ``account_age_days`` from dates if it is missing but dates exist.
    """
    log = CleaningLog(rows_before=len(df))
    df = df.copy()

    # 1. Dates
    for col in config.date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 2. Duplicate rows
    dup_mask = df.duplicated()
    log.duplicate_rows_removed = int(dup_mask.sum())
    if log.duplicate_rows_removed:
        df = df.loc[~dup_mask].copy()

    # 3. Invalid target
    tgt = config.target
    bad_target = ~df[tgt].isin([0, 1])
    log.rows_with_invalid_target_removed = int(bad_target.sum())
    if log.rows_with_invalid_target_removed:
        df = df.loc[~bad_target].copy()
    df[tgt] = df[tgt].astype(int)

    # 4. Impute numeric (median) and categorical (mode)
    for col in config.numeric_cols:
        if col in df.columns:
            n_na = int(df[col].isna().sum())
            if n_na:
                df[col] = df[col].fillna(df[col].median())
                log.imputed_cells[col] = n_na
    for col in config.categorical_cols:
        if col in df.columns:
            n_na = int(df[col].isna().sum())
            if n_na:
                mode = df[col].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "Unknown"
                df[col] = df[col].fillna(fill)
                log.imputed_cells[col] = n_na

    # 5. Normalise categories
    before_counts = {c: df[c].nunique() for c in config.categorical_cols if c in df.columns}
    df = normalise_categories(df, config)
    for c, before in before_counts.items():
        after = df[c].nunique()
        if after != before:
            log.normalised_categories[c] = before - after

    # 6. Derive account_age_days if needed
    if (
        "account_age_days" not in df.columns
        and {"signup_date", "last_purchase_date"}.issubset(df.columns)
    ):
        df["account_age_days"] = (
            (df["last_purchase_date"] - df["signup_date"]).dt.days
        )
        log.notes.append("Derived account_age_days from (last_purchase_date - signup_date).")

    log.rows_after = len(df)
    if not log.imputed_cells and not log.duplicate_rows_removed:
        log.notes.append("No missing values or duplicates found in the verified dataset.")
    return df, log
