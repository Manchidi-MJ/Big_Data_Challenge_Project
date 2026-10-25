"""
Stage 4 - Churn prediction modelling (Objective 5; answers supporting Q3).

Builds a Spark ML pipeline (string indexing + one-hot + assembling + scaling),
trains and compares three classifiers, and reports precision, recall, F1 and
ROC-AUC. Feature importances from the tree model identify which features
matter most.

Usage:
    python -m src.pipeline_04_churn_model
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import functions as F

from . import config
from .utils import get_spark, load_clean, ensure_dirs

sns.set_theme(style="whitegrid")


def build_feature_stages():
    """Indexing + one-hot for categoricals, assemble + scale all features."""
    cat_cols = [c for c in config.CATEGORICAL_COLS]
    num_cols = [c for c in (config.NUMERIC_COLS + config.ENGINEERED_NUMERIC_COLS)]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in cat_cols
    ]
    assembler_inputs = [f"{c}_ohe" for c in cat_cols] + num_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs, outputCol="features_raw", handleInvalid="skip"
    )
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withMean=True, withStd=True
    )
    return indexers + encoders + [assembler, scaler], assembler_inputs, cat_cols, num_cols


def evaluate(predictions, label=config.TARGET):
    """Return precision, recall, F1 and ROC-AUC for a prediction DataFrame."""
    auc = BinaryClassificationEvaluator(
        labelCol=label, rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)

    def mc(metric):
        return MulticlassClassificationEvaluator(
            labelCol=label, predictionCol="prediction", metricName=metric
        ).evaluate(predictions)

    return {
        "accuracy": round(mc("accuracy"), 4),
        "precision": round(mc("weightedPrecision"), 4),
        "recall": round(mc("weightedRecall"), 4),
        "f1": round(mc("f1"), 4),
        "roc_auc": round(auc, 4),
    }


def main() -> None:
    ensure_dirs()
    spark = get_spark("churn-model")
    try:
        df = load_clean(spark)
        # Drop any columns not present to stay robust to schema differences.
        config.CATEGORICAL_COLS = [c for c in config.CATEGORICAL_COLS if c in df.columns]
        config.NUMERIC_COLS = [c for c in config.NUMERIC_COLS if c in df.columns]
        config.ENGINEERED_NUMERIC_COLS = [
            c for c in config.ENGINEERED_NUMERIC_COLS if c in df.columns
        ]

        feature_stages, assembler_inputs, cat_cols, num_cols = build_feature_stages()

        train, test = df.randomSplit(
            [config.TRAIN_FRACTION, 1 - config.TRAIN_FRACTION],
            seed=config.RANDOM_SEED,
        )
        print(f"Train rows: {train.count():,} | Test rows: {test.count():,}")

        models = {
            "LogisticRegression": LogisticRegression(
                labelCol=config.TARGET, featuresCol="features", maxIter=50
            ),
            "RandomForest": RandomForestClassifier(
                labelCol=config.TARGET, featuresCol="features",
                numTrees=100, maxDepth=8, seed=config.RANDOM_SEED,
            ),
            "GBTClassifier": GBTClassifier(
                labelCol=config.TARGET, featuresCol="features",
                maxIter=60, maxDepth=5, seed=config.RANDOM_SEED,
            ),
        }

        results = {}
        fitted = {}
        for name, clf in models.items():
            pipeline = Pipeline(stages=feature_stages + [clf])
            model = pipeline.fit(train)
            preds = model.transform(test)
            metrics = evaluate(preds)
            results[name] = metrics
            fitted[name] = model
            print(f"\n{name}: {metrics}")

        # Save metrics table + JSON.
        metrics_df = pd.DataFrame(results).T.reset_index().rename(
            columns={"index": "model"}
        )
        metrics_df.to_csv(
            os.path.join(config.TABLES_DIR, "model_metrics.csv"), index=False
        )
        with open(os.path.join(config.OUTPUTS_DIR, "model_metrics.json"), "w") as fh:
            json.dump(results, fh, indent=2)

        best = max(results, key=lambda k: results[k]["roc_auc"])
        print(f"\nBest model by ROC-AUC: {best} ({results[best]['roc_auc']})")

        # Metric comparison chart.
        plot_df = metrics_df.melt(id_vars="model", var_name="metric", value_name="value")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=plot_df, x="metric", y="value", hue="model", palette="Set2")
        plt.title("Model comparison across metrics")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.FIGURES_DIR, "07_model_comparison.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close()

        # Feature importances from the Random Forest (named features).
        rf_model = fitted["RandomForest"].stages[-1]
        importances = rf_model.featureImportances.toArray()

        # Recover expanded feature names (one-hot expands categoricals).
        rf_pipeline_model = fitted["RandomForest"]
        try:
            attrs = (
                rf_pipeline_model.transform(test.limit(1))
                .schema["features"]
                .metadata["ml_attr"]["attrs"]
            )
            name_map = {}
            for grp in attrs.values():
                for a in grp:
                    name_map[a["idx"]] = a["name"]
            feat_names = [name_map.get(i, f"f{i}") for i in range(len(importances))]
        except Exception:
            feat_names = [f"f{i}" for i in range(len(importances))]

        imp_df = (
            pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df.to_csv(
            os.path.join(config.TABLES_DIR, "feature_importances.csv"), index=False
        )
        print("\nTop 10 features (Random Forest):")
        print(imp_df.head(10).to_string(index=False))

        plt.figure(figsize=(8, 6))
        top = imp_df.head(15)
        sns.barplot(data=top, y="feature", x="importance", hue="feature",
                    palette="viridis", legend=False)
        plt.title("Top 15 feature importances (Random Forest)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.FIGURES_DIR, "08_feature_importances.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close()

        print(f"\nModel outputs saved to: {config.OUTPUTS_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
