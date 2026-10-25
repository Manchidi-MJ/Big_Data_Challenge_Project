"""Data loading helpers for pandas and PySpark.

Both paths read the *same* raw CSV so results are comparable. The PySpark
session is configured for a small local run and to minimise noisy Windows
Hadoop warnings.
"""
from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .config import CONFIG, Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession


def load_raw_pandas(config: Config = CONFIG, path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw dataset with pandas, parsing date columns.

    Raises a clear error if the file is missing (so users know to download it)
    or if the verified target column is absent.
    """
    csv_path = path or config.path("raw_csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'.\n"
            "Download it from Kaggle and place it there — see data/README.md.\n"
            "URL: " + config.raw["dataset"]["kaggle_url"]
        )
    df = pd.read_csv(csv_path)

    # Parse dates that are actually present (robust to schema variation).
    for col in config.date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if config.target not in df.columns:
        raise KeyError(
            f"Required target column '{config.target}' is missing from the data. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def get_spark_session(
    app_name: str = "ecommerce-churn", master: str = "local[*]"
) -> "SparkSession":
    """Create (or get) a local SparkSession tuned for a small dataset on Windows.

    Notes
    -----
    * We point Spark's local/temp dirs at a project-local folder to avoid the
      harmless-but-noisy ``ShutdownHookManager`` temp-cleanup errors on Windows.
    * ``spark.sql.shuffle.partitions`` is small because the data is ~10k rows;
      the default 200 would create needless overhead.
    """
    from pyspark.sql import SparkSession

    # Use the current interpreter for both driver and workers.
    py = os.environ.get("PYSPARK_PYTHON") or os.sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", py)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", py)

    local_tmp = os.path.join(tempfile.gettempdir(), "spark-churn-tmp")
    os.makedirs(local_tmp, exist_ok=True)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.local.dir", local_tmp)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_raw_spark(
    spark: "SparkSession", config: Config = CONFIG, path: Optional[str] = None
) -> "SparkDataFrame":
    """Load the raw dataset with PySpark using an explicit schema.

    An explicit schema is more reproducible than ``inferSchema`` (no extra pass
    over the data, no surprise type promotions).
    """
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    csv_path = path or config.path("raw_csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{csv_path}'. See data/README.md."
        )

    schema = StructType(
        [
            StructField("customer_id", StringType(), False),
            StructField("signup_date", StringType(), True),
            StructField("last_purchase_date", StringType(), True),
            StructField("days_since_last_purchase", IntegerType(), True),
            StructField("account_age_days", IntegerType(), True),
            StructField("country", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("gender", StringType(), True),
            StructField("device_type", StringType(), True),
            StructField("favorite_category", StringType(), True),
            StructField("total_orders", IntegerType(), True),
            StructField("total_spent_usd", DoubleType(), True),
            StructField("avg_order_value", DoubleType(), True),
            StructField("is_premium_member", IntegerType(), True),
            StructField("churned", IntegerType(), True),
        ]
    )

    df = (
        spark.read.option("header", True)
        .schema(schema)
        .csv(csv_path)
    )
    return df
