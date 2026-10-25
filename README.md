# E-Commerce Customer Churn & Behaviour

**Big Data Analytics, RFM Customer Segmentation and Machine Learning-Based Churn Prediction**

A complete, tested, reproducible university Big Data project: data engineering
(PySpark), exploratory analysis, true RFM segmentation, a mandatory data-leakage
audit, and leakage-aware machine-learning churn classification — built around the
Kaggle *E-Commerce Customer Churn 2025 – 10K* dataset.

---

## TL;DR — the headline finding

The churn label in this dataset is a **deterministic rule**:
`churned == (days_since_last_purchase > 90)`, reproduced with **100% accuracy**
by a single threshold. So:

* A model that sees that field (or the purchase dates) scores ~100% — it just
  **restates the label**, it does not predict.
* With those fields removed, no other behaviour predicts churn (ROC-AUC ≈ 0.54,
  barely above a majority baseline).

This project's most valuable output is the **leakage audit** and the honest,
leakage-aware pipeline built around it. See
[`reports/final_report.md`](reports/final_report.md).

---

## Project structure

```
config/            project_config.yaml (paths, verified schema, params)
data/              raw / interim / processed (data not committed)
notebooks/         01 understanding, 02 cleaning (PySpark), 03 EDA+RFM, 04 modelling
src/churn_project/ reusable modules (config, loading, validation, preprocessing,
                   pyspark_pipeline, rfm, eda, feature_engineering, model_*, predict, run_all)
tests/             pytest suite (synthetic fixtures only)
outputs/           figures / tables / metrics / models (generated)
reports/           final_report.md, references.md, data_dictionary.md
docs/              methodology.md, limitations.md, recommendations.md
presentation/      presentation_outline.md, build_pptx.py, final_presentation.pptx
```

---

## Requirements

* Python **3.11** (tested on 3.11.9)
* Java **17** (OpenJDK) on PATH — required by PySpark
* See [`requirements.txt`](requirements.txt)

---

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate    |  macOS/Linux:  source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip install -e .            # install the churn_project package (src layout)
```

### Get the dataset

The raw data is **not** committed. Download it from Kaggle and place it at
`data/raw/ecommerce_customer_churn_10k.csv` — full instructions in
[`data/README.md`](data/README.md). Dataset licence: **MIT**.

---

## Run the project

```bash
# Full pipeline: quality report -> PySpark cleaning -> EDA -> RFM -> leakage audit
# -> modelling -> saved model. Writes everything under outputs/.
python -m churn_project.run_all

# Pandas-only cleaning (skip PySpark):
python -m churn_project.run_all --no-spark

# Score new customers with the saved (leakage-resistant) model:
python -m churn_project.predict --input data/interim/sample_new_customers.csv \
    --output outputs/tables/predictions.csv
```

### Notebooks

```bash
python notebooks/_build_notebooks.py   # (re)generate the four notebooks
jupyter lab                            # then run 01 -> 04
```

### Tests

```bash
python -m pytest                       # 25 tests, synthetic fixtures only
```

---

## What gets produced

* **Metrics:** `outputs/metrics/model_comparison.csv`, `leakage_audit.json`
* **Tables:** data-quality, EDA, RFM segment summary, cleaning logs
  (`outputs/tables/`)
* **Figures:** 14 PNGs (churn distribution, box plots, histograms, categorical
  churn rates, correlation heatmap, ROC/PR curves, confusion matrices, feature
  importance) in `outputs/figures/`
* **Model:** `outputs/models/churn_model_experiment_B.joblib` (primary,
  leakage-resistant)
* **Report / presentation:** `reports/final_report.md`,
  `presentation/final_presentation.pptx`

---

## Verified facts (for quick reference)

| Quantity | Value |
|---|---|
| Dataset | Kaggle *E-Commerce Customer Churn 2025 – 10K* (MIT) |
| Shape | 10,000 rows × 15 columns, one row per customer |
| Missing / duplicates | none / none |
| Churn rate | 37.71% (3,771 churned) |
| Leakage | `churned == days_since_last_purchase > 90` (100% accuracy) |
| Primary model (Exp B RF) | ROC-AUC 0.54, F1 0.37 — near baseline |

---

## Honesty & scope notes

* 10k rows are **not** "big data"; PySpark demonstrates a scalable *design*.
* Dataset provenance (real vs synthetic) **could not be confirmed**.
* The dataset supports *describing* the current base; it does **not** support
  honest *future* churn prediction (see `docs/limitations.md`).
* Findings are associations, not causation. AI-assistance disclosure is in the
  report (§25).

## Team

| Member | Role |
|---|---|
| Mkhawana Thamelia Hleketani | Data Engineer / Big Data |
| Burgos Carrera, Maria de Los Angeles | Data Analyst / Customer Behaviour |
| Manchidi Magaseng John | Machine Learning |
