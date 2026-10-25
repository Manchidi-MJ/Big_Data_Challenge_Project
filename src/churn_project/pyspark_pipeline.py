"""Reproducible PySpark cleaning pipeline (Phase 2).

Why PySpark for ~10k rows? Honestly, 10,000 rows do NOT require distributed
computing — pandas handles this instantly. PySpark is used here to demonstrate a
*scalable data-processing design*: the same transformation code would run
unchanged on a cluster against a dataset of millions of customers. The report
states this explicitly rather than pretending the current data is "big".

The pipeline mirrors the pandas cleaning logic so the two paths are comparable:
  read -> validate schema -> parse dates -> dedupe -> validate target ->
  impute -> normalise -> derive tenure -> write parquet + csv + log.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict

from .config import CONFIG, Config
from .data_loading import get_spark_session, load_raw_spark

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession


def _spark_quality_checks(df: "SparkDataFrame", config: Config) -> Dict[str, int]:
    """Row count, duplicate rows and duplicate IDs via Spark (validation parity)."""
    from pyspark.sql import functions as F

    total = df.count()
    distinct = df.distinct().count()
    distinct_ids = df.select(config.id_col).distinct().count()
    return {
        "n_rows": int(total),
        "duplicate_rows": int(total - distinct),
        "duplicate_customer_ids": int(total - distinct_ids),
    }


def run_pipeline(
    spark: "SparkSession | None" = None, config: Config = CONFIG
) -> Dict[str, object]:
    """Execute the full PySpark cleaning pipeline and persist outputs.

    Returns a log dict (also written to outputs/tables/pyspark_cleaning_log.json).
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType

    config.ensure_output_dirs()
    owns_spark = spark is None
    if spark is None:
        spark = get_spark_session()

    log: Dict[str, object] = {"steps": []}

    # 1. Read with explicit schema -----------------------------------------
    df = load_raw_spark(spark, config)
    before = _spark_quality_checks(df, config)
    log["before"] = before
    log["steps"].append("read raw CSV with explicit schema")

    # 2. Parse dates ---------------------------------------------------------
    for col in config.date_cols:
        if col in df.columns:
            df = df.withColumn(col, F.to_date(F.col(col), "yyyy-MM-dd"))
    log["steps"].append("parsed date columns to DateType")

    # 3. Drop fully-duplicated rows -----------------------------------------
    df = df.dropDuplicates()
    log["steps"].append("dropped fully-duplicated rows")

    # 4. Keep only valid target values (0/1) --------------------------------
    tgt = config.target
    df = df.filter(F.col(tgt).isin(0, 1))
    df = df.withColumn(tgt, F.col(tgt).cast(IntegerType()))
    log["steps"].append("filtered rows to valid target values {0,1}")

    # 5. Impute numeric (median) + categorical (mode) -----------------------
    imputed: Dict[str, int] = {}
    for col in config.numeric_cols:
        if col in df.columns:
            n_na = df.filter(F.col(col).isNull()).count()
            if n_na:
                median = df.approxQuantile(col, [0.5], 0.001)[0]
                df = df.fillna({col: median})
                imputed[col] = int(n_na)
    for col in config.categorical_cols:
        if col in df.columns:
            n_na = df.filter(F.col(col).isNull()).count()
            if n_na:
                mode_row = (
                    df.filter(F.col(col).isNotNull())
                    .groupBy(col)
                    .count()
                    .orderBy(F.desc("count"))
                    .first()
                )
                fill = mode_row[col] if mode_row else "Unknown"
                df = df.fillna({col: fill})
                imputed[col] = int(n_na)
    log["imputed_cells"] = imputed

    # 6. Normalise categorical strings (trim) -------------------------------
    for col in config.categorical_cols:
        if col in df.columns:
            df = df.withColumn(col, F.trim(F.col(col)))
    log["steps"].append("trimmed categorical strings")

    # 7. Derive tenure if needed --------------------------------------------
    if "account_age_days" not in df.columns and {
        "signup_date",
        "last_purchase_date",
    }.issubset(set(df.columns)):
        df = df.withColumn(
            "account_age_days",
            F.datediff(F.col("last_purchase_date"), F.col("signup_date")),
        )
        log["steps"].append("derived account_age_days via datediff")

    # 8. Persist -------------------------------------------------------------
    after = _spark_quality_checks(df, config)
    log["after"] = after

    # Collect the cleaned data once; used for the CSV sink and (if possible)
    # verified before any native-Hadoop write is attempted.
    cleaned_pdf = df.coalesce(1).toPandas()

    # Parquet sink. On Windows, Spark's native Parquet writer needs winutils.exe
    # / HADOOP_HOME to set file permissions; when that is unavailable the write
    # raises. We attempt it and fall back cleanly so the pipeline still produces
    # its canonical CSV output and remains reproducible on any OS.
    parquet_path = config.path("clean_parquet")
    try:
        df.write.mode("overwrite").parquet(parquet_path)
        log["parquet_path"] = parquet_path
        log["parquet_status"] = "written"
    except Exception as exc:  # pragma: no cover - platform dependent
        log["parquet_path"] = None
        log["parquet_status"] = (
            f"skipped (native Hadoop write unavailable on this platform: "
            f"{type(exc).__name__}). CSV output is the canonical cleaned dataset."
        )

    # Single CSV for easy downstream pandas use (works on all platforms).
    csv_path = config.path("clean_csv")
    cleaned_pdf.to_csv(csv_path, index=False)
    log["csv_path"] = csv_path
    log["steps"].append("wrote cleaned single CSV (and Parquet where supported)")

    # Write the log.
    log_path = os.path.join(config.path("tables"), "pyspark_cleaning_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    if owns_spark:
        spark.stop()
    return log


if __name__ == "__main__":  # pragma: no cover
    result = run_pipeline()
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))
