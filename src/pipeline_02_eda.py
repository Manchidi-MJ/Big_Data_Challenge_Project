"""
Stage 2 - Exploratory Data Analysis (Objective 3; answers supporting Q1).

Computes the overall churn rate, compares churned vs retained customers, and
produces charts + summary tables saved under outputs/.

Usage:
    python -m src.pipeline_02_eda
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F

from . import config
from .utils import get_spark, load_clean, ensure_dirs, save_pandas_csv

sns.set_theme(style="whitegrid")


def _savefig(name: str) -> str:
    path = os.path.join(config.FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def main() -> None:
    ensure_dirs()
    spark = get_spark("churn-eda")
    try:
        df = load_clean(spark)
        total = df.count()

        # --- Q1: overall churn rate -----------------------------------------
        churn_counts = (
            df.groupBy(config.TARGET).count().orderBy(config.TARGET).toPandas()
        )
        churn_counts["pct"] = (churn_counts["count"] / total * 100).round(2)
        save_pandas_csv(churn_counts, "churn_distribution.csv")
        churn_rate = float(
            churn_counts.loc[churn_counts[config.TARGET] == 1, "count"].sum()
        ) / total
        print(f"Overall churn rate: {churn_rate:.2%}  (n={total:,})")

        plt.figure(figsize=(5, 4))
        sns.barplot(
            data=churn_counts, x=config.TARGET, y="count", hue=config.TARGET,
            palette="Set2", legend=False,
        )
        plt.title("Churn distribution (0 = retained, 1 = churned)")
        plt.xlabel("Churn")
        plt.ylabel("Customers")
        _savefig("01_churn_distribution.png")

        # --- Numeric comparison: churned vs retained ------------------------
        num_cols = [c for c in config.NUMERIC_COLS if c in df.columns]
        means = (
            df.groupBy(config.TARGET)
            .agg(*[F.round(F.avg(c), 2).alias(c) for c in num_cols])
            .orderBy(config.TARGET)
            .toPandas()
        )
        save_pandas_csv(means, "feature_means_by_churn.csv")
        print("\nMean of numeric features by churn group:")
        print(means.to_string(index=False))

        # --- Churn rate by selected categorical / ordinal drivers -----------
        for col in ["Complain", "SatisfactionScore", "MaritalStatus",
                    "PreferedOrderCat", "CityTier"]:
            if col not in df.columns:
                continue
            rate = (
                df.groupBy(col)
                .agg(
                    F.count("*").alias("customers"),
                    F.round(F.avg(config.TARGET) * 100, 2).alias("churn_rate_pct"),
                )
                .orderBy(F.desc("churn_rate_pct"))
                .toPandas()
            )
            save_pandas_csv(rate, f"churn_rate_by_{col}.csv")

            plt.figure(figsize=(6, 4))
            order = rate[col].astype(str).tolist()
            sns.barplot(
                data=rate.assign(**{col: rate[col].astype(str)}),
                x=col, y="churn_rate_pct", hue=col, palette="Set2",
                legend=False, order=order,
            )
            plt.title(f"Churn rate by {col}")
            plt.ylabel("Churn rate (%)")
            plt.xticks(rotation=30, ha="right")
            _savefig(f"02_churn_rate_by_{col}.png")

        # --- Tenure distribution split by churn -----------------------------
        if "Tenure" in df.columns:
            pdf = df.select("Tenure", config.TARGET).toPandas()
            plt.figure(figsize=(7, 4))
            sns.histplot(
                data=pdf, x="Tenure", hue=config.TARGET, bins=30,
                element="step", stat="density", common_norm=False, palette="Set2",
            )
            plt.title("Tenure distribution: churned vs retained")
            _savefig("03_tenure_by_churn.png")

        # --- Correlation heatmap of numeric features ------------------------
        corr_pdf = df.select(num_cols + [config.TARGET]).toPandas()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_pdf.corr(numeric_only=True), annot=False, cmap="coolwarm",
            center=0, linewidths=0.4,
        )
        plt.title("Correlation matrix (numeric features + churn)")
        _savefig("04_correlation_heatmap.png")

        print(f"\nEDA figures saved to: {config.FIGURES_DIR}")
        print(f"EDA tables saved to:  {config.TABLES_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
