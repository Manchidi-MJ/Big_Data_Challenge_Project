"""End-to-end orchestration: run every phase and persist all deliverables.

Usage (from project root):
    python -m churn_project.run_all            # full run (includes PySpark)
    python -m churn_project.run_all --no-spark # skip PySpark, clean with pandas

Writes:
    data/processed/clean_customers.csv, rfm_customers.csv
    outputs/tables/*.csv, outputs/metrics/*.csv/json
    outputs/figures/*.png
    outputs/models/churn_model_experiment_B.joblib
    outputs/tables/run_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import joblib
import pandas as pd

from . import (
    data_loading,
    data_validation,
    exploratory_analysis,
    feature_engineering,
    model_evaluation,
    model_training,
    preprocessing,
    rfm_analysis,
)
from .config import CONFIG
from .reporting import save_json, save_table


def phase1_quality(df: pd.DataFrame) -> Dict:
    print("[Phase 1] Data understanding & quality assessment")
    q = data_validation.build_quality_summary(df)
    # Persist the human-readable parts.
    save_table(q["missing"], "quality_missing_values.csv")
    save_table(q["numeric_summary"], "quality_numeric_summary.csv")
    save_table(q["target_distribution"], "quality_target_distribution.csv")
    save_table(q["invalid_values"], "quality_invalid_values.csv")
    save_table(q["redundancy"], "quality_redundant_features.csv")
    summary = {
        "n_rows": q["n_rows"], "n_cols": q["n_cols"], "columns": q["columns"],
        "dtypes": q["dtypes"], "schema_check": q["schema_check"],
        "duplicates": q["duplicates"],
    }
    save_json(summary, "data_quality_summary.json")
    return q


def phase2_clean(df: pd.DataFrame, use_spark: bool) -> pd.DataFrame:
    if use_spark:
        print("[Phase 2] PySpark cleaning pipeline")
        try:
            log = __import__(
                "churn_project.pyspark_pipeline", fromlist=["run_pipeline"]
            ).run_pipeline()
            print(f"          Spark cleaned {log['after']['n_rows']} rows -> {log['csv_path']}")
            clean = pd.read_csv(CONFIG.path("clean_csv"))
            for c in CONFIG.date_cols:
                if c in clean.columns:
                    clean[c] = pd.to_datetime(clean[c], errors="coerce")
            return clean
        except Exception as exc:  # fall back to pandas path
            print(f"          PySpark path failed ({exc!r}); falling back to pandas cleaning.")
    print("[Phase 2] Pandas cleaning pipeline")
    clean, log = preprocessing.clean_pandas(df)
    clean.to_csv(CONFIG.path("clean_csv"), index=False)
    save_json(log.as_dict(), "pandas_cleaning_log.json")
    print(f"          Cleaned {log.rows_after} rows -> {CONFIG.path('clean_csv')}")
    return clean


def phase3_eda(df: pd.DataFrame) -> Dict:
    print("[Phase 3] Exploratory data analysis")
    res = exploratory_analysis.run_full_eda(df)
    save_json(res["overview"], "eda_overview.json")
    print(f"          Churn rate = {res['overview']['churn_rate']:.3f}")
    return res


def phase4_rfm(df: pd.DataFrame) -> Dict:
    print("[Phase 4] RFM segmentation")
    rfm, meta = rfm_analysis.compute_rfm(df)
    rfm.to_csv(CONFIG.path("rfm_csv"), index=False)
    seg = rfm_analysis.segment_summary(rfm)
    save_table(seg, "rfm_segment_summary.csv")
    save_json(meta, "rfm_meta.json")
    print(f"          {len(seg)} segments; bins used = {meta['n_bins_used']}")
    return {"rfm": rfm, "segments": seg, "meta": meta}


def phase5_models(df: pd.DataFrame) -> Dict:
    print("[Phase 5] Leakage audit + churn modelling")
    audit = feature_engineering.audit_leakage(df)
    save_json(audit, "leakage_audit.json", subdir="metrics")
    print(f"          Leakage verdict: {audit['verdict']['days_since_last_purchase']}")

    all_rows = []
    trained: Dict[str, object] = {}
    importances: Dict[str, pd.DataFrame] = {}

    for exp in ("A", "B"):
        X, y, feats = feature_engineering.build_xy(df, exp)
        X_tr, X_te, y_tr, y_te = model_training.split_data(X, y)
        models = model_training.build_models(feats["numeric"], feats["categorical"])
        for name, pipe in models.items():
            cv_f1 = model_evaluation.cross_validate_f1(pipe, X_tr, y_tr)
            pipe.fit(X_tr, y_tr)
            row = model_evaluation.evaluate_model(f"{name}", pipe, X_te, y_te)
            row["experiment"] = exp
            row["cv_f1_train"] = cv_f1
            all_rows.append(row)
            trained[f"{exp}:{name}"] = pipe

        # Curves per experiment
        model_evaluation.plot_roc_pr(models, X_te, y_te, suffix=f"experiment_{exp}")

        # Importance + confusion for the RandomForest of this experiment
        rf = models["RandomForest"]
        model_evaluation.plot_confusion(rf, X_te, y_te, f"RF_exp{exp}")
        imp = model_evaluation.permutation_importance_df(rf, X_te, y_te)
        importances[exp] = imp
        save_table(imp, f"feature_importance_experiment_{exp}.csv")
        model_evaluation.plot_feature_importance(imp, f"RF_exp{exp}")

    comp = pd.DataFrame(all_rows)[
        ["experiment", "model", "accuracy", "precision", "recall", "f1",
         "roc_auc", "pr_auc", "cv_f1_train"]
    ].sort_values(["experiment", "f1"], ascending=[True, False])
    comp_path = os.path.join(CONFIG.path("metrics"), "model_comparison.csv")
    comp.to_csv(comp_path, index=False)
    print(f"          Model comparison saved -> {comp_path}")

    # Primary model = leakage-resistant RandomForest (Experiment B).
    primary = trained["B:RandomForest"]
    model_path = os.path.join(CONFIG.path("models"), "churn_model_experiment_B.joblib")
    X_b, y_b, feats_b = feature_engineering.build_xy(df, "B")
    joblib.dump(
        {"pipeline": primary, "numeric": feats_b["numeric"],
         "categorical": feats_b["categorical"], "target": CONFIG.target,
         "experiment": "B", "note": "Leakage-resistant primary model."},
        model_path,
    )
    print(f"          Primary model (Experiment B RF) saved -> {model_path}")
    return {"audit": audit, "comparison": comp, "model_path": model_path,
            "importances": importances}


def main(use_spark: bool = True) -> Dict:
    CONFIG.ensure_output_dirs()
    df_raw = data_loading.load_raw_pandas()
    q = phase1_quality(df_raw)
    clean = phase2_clean(df_raw, use_spark=use_spark)
    eda = phase3_eda(clean)
    rfm = phase4_rfm(clean)
    models = phase5_models(clean)

    summary = {
        "n_rows": q["n_rows"],
        "n_cols": q["n_cols"],
        "churn_overview": eda["overview"],
        "rfm_segments": int(len(rfm["segments"])),
        "leakage_verdict": models["audit"]["verdict"],
        "model_comparison": models["comparison"].to_dict(orient="records"),
        "primary_model": models["model_path"],
    }
    save_json(summary, "run_summary.json")
    print("\n[DONE] run_summary.json written.")
    return summary


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-spark", action="store_true", help="Skip PySpark, clean with pandas")
    args = parser.parse_args()
    main(use_spark=not args.no_spark)
