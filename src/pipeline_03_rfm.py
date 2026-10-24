"""
Stage 3 - RFM segmentation (Objective 4; answers supporting Q2).

The Kaggle churn dataset has no transaction-level timestamps, so classic RFM
is approximated with the closest available proxies:
    Recency   = DaySinceLastOrder  (lower is better -> reversed score)
    Frequency = OrderCount         (higher is better)
    Monetary  = CashbackAmount     (higher is better, spend proxy)

Each dimension is scored 1-4 using quartiles (ntile). Customers are then
grouped into business segments and each segment's churn rate is reported.

Usage:
    python -m src.pipeline_03_rfm
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from . import config
from .utils import get_spark, load_clean, ensure_dirs, save_pandas_csv

sns.set_theme(style="whitegrid")


def add_rfm_scores(df):
    """Add R, F, M quartile scores (1-4) and a combined RFM score."""
    # Recency: lower DaySinceLastOrder is better -> ascending ntile then reverse.
    r_win = Window.orderBy(F.col(config.RFM_RECENCY).asc())
    f_win = Window.orderBy(F.col(config.RFM_FREQUENCY).asc())
    m_win = Window.orderBy(F.col(config.RFM_MONETARY).asc())

    df = (
        df.withColumn("R_raw", F.ntile(4).over(r_win))
        .withColumn("F", F.ntile(4).over(f_win))
        .withColumn("M", F.ntile(4).over(m_win))
    )
    # Reverse recency so that 4 = most recent (best).
    df = df.withColumn("R", F.lit(5) - F.col("R_raw")).drop("R_raw")
    df = df.withColumn("RFM_Score", F.col("R") + F.col("F") + F.col("M"))
    return df


def segment(df):
    """Map the combined RFM score to interpretable business segments."""
    score = F.col("RFM_Score")
    seg = (
        F.when(score >= 10, "Champions")
        .when(score >= 8, "Loyal")
        .when(score >= 6, "Potential")
        .when(score >= 4, "At Risk")
        .otherwise("Lost")
    )
    return df.withColumn("Segment", seg)


def main() -> None:
    ensure_dirs()
    spark = get_spark("churn-rfm")
    try:
        df = load_clean(spark)
        for col in (config.RFM_RECENCY, config.RFM_FREQUENCY, config.RFM_MONETARY):
            if col not in df.columns:
                raise ValueError(f"Required RFM proxy column missing: {col}")

        df = add_rfm_scores(df)
        df = segment(df)

        # Segment summary: size, average RFM, and churn rate (Q2).
        summary = (
            df.groupBy("Segment")
            .agg(
                F.count("*").alias("customers"),
                F.round(F.avg("R"), 2).alias("avg_R"),
                F.round(F.avg("F"), 2).alias("avg_F"),
                F.round(F.avg("M"), 2).alias("avg_M"),
                F.round(F.avg(config.TARGET) * 100, 2).alias("churn_rate_pct"),
            )
            .orderBy(F.desc("churn_rate_pct"))
            .toPandas()
        )
        save_pandas_csv(summary, "rfm_segment_summary.csv")
        print("RFM segment summary (sorted by churn rate):")
        print(summary.to_string(index=False))

        # Persist per-customer segments for reuse.
        seg_out = os.path.join(config.TABLES_DIR, "rfm_customer_segments.csv")
        keep = [config.ID_COL, "R", "F", "M", "RFM_Score", "Segment", config.TARGET]
        keep = [c for c in keep if c in df.columns]
        df.select(*keep).toPandas().to_csv(seg_out, index=False)
        print(f"\nPer-customer segments written to: {seg_out}")

        # Chart: churn rate by segment.
        plt.figure(figsize=(7, 4))
        order = summary.sort_values("churn_rate_pct", ascending=False)["Segment"].tolist()
        sns.barplot(
            data=summary, x="Segment", y="churn_rate_pct", hue="Segment",
            palette="Set2", legend=False, order=order,
        )
        plt.title("Churn rate by RFM segment")
        plt.ylabel("Churn rate (%)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.FIGURES_DIR, "05_churn_rate_by_segment.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close()

        # Chart: segment sizes.
        plt.figure(figsize=(7, 4))
        sns.barplot(
            data=summary.sort_values("customers", ascending=False),
            x="Segment", y="customers", hue="Segment", palette="Set3", legend=False,
        )
        plt.title("Customers per RFM segment")
        plt.ylabel("Customers")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.FIGURES_DIR, "06_segment_sizes.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close()

        print(f"RFM figures saved to: {config.FIGURES_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
