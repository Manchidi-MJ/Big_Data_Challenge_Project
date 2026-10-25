"""Exploratory data analysis (Phase 3): tables + figures.

Every figure answers a specific analytical question and is saved to
outputs/figures/. Functions return computed values so interpretations in the
report/notebook use REAL numbers, never invented ones.
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless backend for scripted runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CONFIG, Config
from .reporting import save_figure, save_table

sns.set_theme(style="whitegrid")


def churn_overview(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, float]:
    """Headline churn numbers."""
    n = len(df)
    churned = int(df[config.target].sum())
    retained = n - churned
    return {
        "total_customers": n,
        "churned": churned,
        "retained": retained,
        "churn_rate": round(churned / n, 4),
        "retained_rate": round(retained / n, 4),
        "imbalance_ratio_majority_to_minority": round(
            max(churned, retained) / max(min(churned, retained), 1), 3
        ),
    }


def plot_churn_distribution(df: pd.DataFrame, config: Config = CONFIG) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.target].map({0: "Retained", 1: "Churned"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                hue=counts.index, palette=["#2a9d8f", "#e76f51"], legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Customer churn distribution")
    ax.set_ylabel("Number of customers")
    ax.set_xlabel("")
    path = save_figure(fig, "01_churn_distribution.png", config)
    plt.close(fig)
    return path


def grouped_numeric_summary(df: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Mean of each numeric feature split by churn status."""
    cols = [c for c in config.numeric_cols if c in df.columns]
    out = df.groupby(config.target)[cols].mean().T
    out.columns = ["retained_mean", "churned_mean"]
    out["difference"] = out["churned_mean"] - out["retained_mean"]
    return out.round(3).reset_index().rename(columns={"index": "feature"})


def plot_numeric_boxplots(df: pd.DataFrame, config: Config = CONFIG) -> str:
    cols = [c for c in config.numeric_cols if c in df.columns and c != "is_premium_member"]
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=config.target, y=col, ax=ax,
                    hue=config.target, palette=["#2a9d8f", "#e76f51"], legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Retained", "Churned"])
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numeric features by churn status", y=1.02, fontsize=13)
    fig.tight_layout()
    path = save_figure(fig, "02_numeric_boxplots_by_churn.png", config)
    plt.close(fig)
    return path


def plot_numeric_histograms(df: pd.DataFrame, config: Config = CONFIG) -> str:
    cols = [c for c in config.numeric_cols if c in df.columns and c != "is_premium_member"]
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=30, color="#264653")
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distributions of numeric features", y=1.02, fontsize=13)
    fig.tight_layout()
    path = save_figure(fig, "03_numeric_histograms.png", config)
    plt.close(fig)
    return path


def categorical_churn_rates(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, pd.DataFrame]:
    """Churn rate within each category level, per categorical feature."""
    out = {}
    for col in config.categorical_cols:
        if col in df.columns:
            tbl = (
                df.groupby(col)[config.target]
                .agg(["mean", "count"])
                .rename(columns={"mean": "churn_rate", "count": "n_customers"})
                .sort_values("churn_rate", ascending=False)
                .round(4)
                .reset_index()
            )
            out[col] = tbl
    return out


def plot_categorical_churn(df: pd.DataFrame, config: Config = CONFIG) -> str:
    cols = [c for c in config.categorical_cols if c in df.columns]
    n = len(cols)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.5 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        rates = df.groupby(col)[config.target].mean().sort_values(ascending=False)
        sns.barplot(x=rates.values, y=rates.index, ax=ax, color="#e9c46a")
        ax.set_title(f"Churn rate by {col}")
        ax.set_xlabel("Churn rate")
        ax.set_ylabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    path = save_figure(fig, "04_categorical_churn_rates.png", config)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, config: Config = CONFIG) -> str:
    cols = [c for c in config.numeric_cols if c in df.columns] + [config.target]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap (numeric features + churn)")
    path = save_figure(fig, "05_correlation_heatmap.png", config)
    plt.close(fig)
    return path


def run_full_eda(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, object]:
    """Run every EDA step, save tables+figures, return a results dict."""
    config.ensure_output_dirs()
    overview = churn_overview(df, config)

    grouped = grouped_numeric_summary(df, config)
    save_table(grouped, "eda_numeric_by_churn.csv", config)

    cat_rates = categorical_churn_rates(df, config)
    for col, tbl in cat_rates.items():
        save_table(tbl, f"eda_churn_rate_by_{col}.csv", config)

    figures = {
        "churn_distribution": plot_churn_distribution(df, config),
        "numeric_boxplots": plot_numeric_boxplots(df, config),
        "numeric_histograms": plot_numeric_histograms(df, config),
        "categorical_churn": plot_categorical_churn(df, config),
        "correlation_heatmap": plot_correlation_heatmap(df, config),
    }
    return {"overview": overview, "grouped_numeric": grouped,
            "categorical_rates": cat_rates, "figures": figures}
