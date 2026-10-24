"""
Stage 1 - Data engineering & cleaning (Objective 2).

Reads the raw Kaggle CSV with PySpark, standardises categories, imputes
missing numeric values with the median, removes duplicates, and writes a
cleaned Parquet dataset for the downstream stages.

Usage:
    python -m src.pipeline_01_data_cleaning
"""
from __future__ import annotations

import shutil
import os

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from . import config
from .utils import get_spark, load_raw, null_report, ensure_dirs


def normalise_categories(df: DataFrame) -> DataFrame:
    """Fix known inconsistent category spellings (e.g. 'CC' -> 'Credit Card')."""
    for col, mapping in config.CATEGORY_NORMALISATION.items():
        if col not in df.columns:
            continue
        expr = F.col(col)
        for raw, canonical in mapping.items():
            expr = F.when(F.col(col) == raw, F.lit(canonical)).otherwise(expr)
        df = df.withColumn(col, expr)
    # Trim stray whitespace on all string columns.
    for col, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(col, F.trim(F.col(col)))
    return df


def impute_median(df: DataFrame) -> DataFrame:
    """Fill nulls in known-sparse numeric columns with the column median."""
    present = [c for c in config.NUMERIC_IMPUTE_COLS if c in df.columns]
    if not present:
        return df
    medians = df.approxQuantile(present, [0.5], 0.001)
    fill = {c: (m[0] if m else 0.0) for c, m in zip(present, medians)}
    return df.fillna(fill)


def clean(df: DataFrame) -> DataFrame:
    df = df.dropDuplicates()
    df = normalise_categories(df)
    df = impute_median(df)
    # Ensure the target is an integer 0/1.
    if config.TARGET in df.columns:
        df = df.withColumn(config.TARGET, F.col(config.TARGET).cast("int"))
    return df


def main() -> None:
    ensure_dirs()
    spark = get_spark("churn-cleaning")
    try:
        raw = load_raw(spark)
        print(f"Raw rows: {raw.count():,} | columns: {len(raw.columns)}")

        print("\nNull counts BEFORE cleaning:")
        null_report(raw).show(vertical=True, truncate=False)

        cleaned = clean(raw)
        print(f"Clean rows (after de-dup): {cleaned.count():,}")

        print("\nNull counts AFTER cleaning:")
        null_report(cleaned).show(vertical=True, truncate=False)

        if config.TARGET in cleaned.columns:
            print("\nChurn distribution:")
            cleaned.groupBy(config.TARGET).count().orderBy(config.TARGET).show()

        # Overwrite any previous clean output.
        if os.path.exists(config.CLEAN_PARQUET):
            shutil.rmtree(config.CLEAN_PARQUET)
        cleaned.write.mode("overwrite").parquet(config.CLEAN_PARQUET)
        print(f"\nCleaned data written to: {config.CLEAN_PARQUET}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
