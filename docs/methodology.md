# Methodology

This document describes *how* each phase of the project was carried out. It is a
technical companion to the final report.

## 1. Project architecture

```
config/           project_config.yaml — single source of truth (paths, schema, params)
src/churn_project/ reusable modules (one per concern)
tests/            pytest suite (synthetic fixtures only)
notebooks/        narrative notebooks (01–04) wrapping the src modules
outputs/          figures, tables, metrics, models (generated)
reports/ docs/    written deliverables
```

All logic lives in `src/churn_project/`; notebooks and `run_all.py` orchestrate
it. This keeps the analysis reproducible and testable rather than trapped in
notebook cells.

## 2. Data understanding (Phase 1)

`data_validation.py` computes: shape, dtypes, missing values, duplicate rows and
IDs, invalid/impossible value checks, numeric summaries, categorical value
counts, target distribution, a correlation matrix and a redundancy scan. Nothing
is deleted at this stage — it only reports. Outputs are written to
`outputs/tables/` and `data_quality_summary.json`.

## 3. Cleaning + PySpark pipeline (Phase 2)

Two parity paths implement the same steps:

* **Pandas** (`preprocessing.clean_pandas`) — default, fast.
* **PySpark** (`pyspark_pipeline.run_pipeline`) — demonstrates a scalable design.

Steps: read with an explicit schema → parse dates → drop duplicate rows → keep
valid target values {0,1} → median-impute numeric and mode-impute categorical
(none needed here) → trim category strings → derive `account_age_days` if absent
→ persist. Before/after counts and a step log are saved.

### Why PySpark for ~10k rows?

Honestly, 10,000 rows do **not** require distributed computing; pandas handles
them instantly. PySpark is included to demonstrate a *scalable processing
design*: the identical transformation code would run unchanged on a cluster
against millions of customers. We do not claim the current data is "big".

### Windows note

Spark's native Parquet writer needs `winutils.exe` / `HADOOP_HOME` to set file
permissions on Windows. When unavailable, the pipeline logs this and writes the
canonical cleaned dataset as a single CSV instead. All Spark *processing*
(read, dedupe, impute, transform, aggregate) still executes.

## 4. Exploratory data analysis (Phase 3)

`exploratory_analysis.py` produces churn overview stats, churned-vs-retained
numeric comparisons, categorical churn rates, histograms, box plots, bar charts
and a correlation heatmap. Every interpretation in the report uses the saved,
computed numbers.

## 5. RFM segmentation (Phase 4)

True RFM from real fields:

| Dimension | Field | Direction |
|-----------|-------|-----------|
| Recency | `days_since_last_purchase` | lower is better |
| Frequency | `total_orders` | higher is better |
| Monetary | `total_spent_usd` | higher is better |

Scoring uses rank-based quantile binning (`pd.qcut` on ranks,
`duplicates='drop'`) with up to 5 bins; the number of bins actually used per
dimension is reported. Segment labels (Champions, Loyal Customers, Potential
Loyalists, At Risk, Needs Attention, About to Sleep) come from documented
score-combination rules and are interpretive, not a universal taxonomy.

`recency` vs `days_since_last_purchase`: the dataset has **no** separate
`recency` column, so `days_since_last_purchase` is used for Recency.
`frequency` vs `total_orders`: there is **no** separate `frequency` column, so
`total_orders` is used for Frequency.

## 6. Leakage audit (Phase 5, mandatory)

`feature_engineering.audit_leakage` tests whether `churned` can be reconstructed
from candidate predictors. It found a threshold at
`days_since_last_purchase = 90` reproduces the label with **100% accuracy**, so
that field (and the date columns that encode the same signal) are
target-defining leakage.

Two experiments follow:

* **Experiment A** — all plausible predictors (includes the leakage field).
  Exploratory sanity check only.
* **Experiment B** — leakage-resistant (drops `days_since_last_purchase`, never
  uses raw dates or `customer_id`). **Primary model.**

## 7. Modelling (Phase 5)

`model_training.py` builds scikit-learn Pipelines so preprocessing (median
impute + standardise numeric; impute + one-hot categorical) is fit on training
folds only. Models: `DummyClassifier` (baseline), `LogisticRegression`,
`RandomForestClassifier`. `class_weight='balanced'` handles the training
imbalance. Split: stratified 80/20, fixed seed 42. Model selection uses 5-fold
CV F1 on the training set; the test set is touched once for final reporting.

## 8. Evaluation (Phase 5/6)

`model_evaluation.py` reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrices, ROC and PR curves, and permutation importance. The model
comparison table is saved to `outputs/metrics/model_comparison.csv`.

## 9. Persistence + prediction

The leakage-resistant RandomForest pipeline (preprocessing + model) is saved
with joblib. `predict.py` loads it and scores new customers, returning a churn
probability and a 0/1 prediction, with an explicit caveat that the score is not
a guarantee of future churn.

## 10. Reproducibility

`python -m churn_project.run_all` runs every phase from the raw CSV to all
outputs, without manual edits to intermediate files. Tests run with `pytest`.
