"""
Shared helpers: Spark session factory, data loading and small utilities.
"""
from __future__ import annotations

import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from . import config


def get_spark(app_name: str = "EcommerceChurn") -> SparkSession:
    """Create (or reuse) a local SparkSession.

    The configuration targets a single-machine run on the 10K dataset while
    keeping the code identical to what would run on a cluster -- this is the
    'scalable pipeline design' justification for using Spark (see README).
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_raw(spark: SparkSession, path: str | None = None) -> DataFrame:
    """Load the raw Kaggle CSV with a schema inferred from the header."""
    csv_path = path or config.RAW_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}.\n"
            "Download the Kaggle 'E-Commerce Customer Churn 2025 - 10K' CSV "
            "and place it there (see data/README.md)."
        )
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
    )


def load_clean(spark: SparkSession, path: str | None = None) -> DataFrame:
    """Load the cleaned Parquet produced by 01_data_cleaning.py."""
    parquet_path = path or config.CLEAN_PARQUET
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"Cleaned data not found at {parquet_path}. "
            "Run 01_data_cleaning.py first."
        )
    return spark.read.parquet(parquet_path)


def null_report(df: DataFrame) -> DataFrame:
    """Return a one-row DataFrame counting nulls per column."""
    return df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    )


def ensure_dirs() -> None:
    """Create all output directories if they do not exist."""
    for d in (
        config.PROCESSED_DIR,
        config.OUTPUTS_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def save_pandas_csv(pdf, name: str) -> str:
    """Save a pandas DataFrame to outputs/tables and return the path."""
    ensure_dirs()
    out = os.path.join(config.TABLES_DIR, name)
    pdf.to_csv(out, index=False)
    return out
