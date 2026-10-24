# E-Commerce Customer Churn & Behaviour — Big Data Project

An end-to-end **PySpark** pipeline that turns raw e-commerce customer data into
retention insights: data engineering → exploratory analysis → **RFM
segmentation** → **machine-learning churn prediction** → business
recommendations.

Dataset: [E-Commerce Customer Churn 2025 – 10K (Kaggle)](https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k)
(~10,000 customer records; target `Churn`, 1 = churned, 0 = retained).

---

## Research questions

**Main:** Which customer behaviours and characteristics are most strongly
associated with churn, and how accurately can machine learning predict whether
a customer will churn?

1. What proportion of customers churn, and how do churned and retained
   customers differ? → **Stage 2 (EDA)**
2. Which RFM-based segments are most at risk of churning? → **Stage 3 (RFM)**
3. Which model predicts churn best, and which features matter most? →
   **Stage 4 (modelling)**

## How the project maps to the objectives

| Objective | Where it lives |
|---|---|
| 1. Select & document the dataset | `data/README.md` |
| 2. Clean & preprocess; reproducible PySpark pipeline | `src/pipeline_01_data_cleaning.py`, `src/config.py` |
| 3. EDA on churn rate & feature relationships | `src/pipeline_02_eda.py` |
| 4. RFM segmentation & business interpretation | `src/pipeline_03_rfm.py` |
| 5. Build & compare classifiers (precision/recall/F1/ROC-AUC) | `src/pipeline_04_churn_model.py` |
| 6. Connect analysis + model into retention recommendations | `docs/recommendations.md` |
| 7. Limitations & future work | `docs/limitations_and_future_work.md` |

---

## Project structure

```
Big_Data_Challenge_Project/
├── README.md
├── requirements.txt
├── Makefile
├── data/
│   ├── README.md                       # dataset source & download steps
│   ├── raw/                            # put ecommerce_churn.csv here (git-ignored)
│   └── processed/                      # cleaned Parquet (git-ignored)
├── src/
│   ├── config.py                       # all paths, schema, parameters
│   ├── utils.py                        # Spark session + loaders
│   ├── generate_sample_data.py         # synthetic 10K sample (schema match)
│   ├── pipeline_01_data_cleaning.py    # Stage 1
│   ├── pipeline_02_eda.py              # Stage 2
│   ├── pipeline_03_rfm.py              # Stage 3
│   ├── pipeline_04_churn_model.py      # Stage 4
│   └── run_pipeline.py                 # run everything
├── notebooks/
│   └── churn_analysis.ipynb            # narrative walkthrough
├── docs/
│   ├── recommendations.md              # Objective 6
│   └── limitations_and_future_work.md  # Objective 7
└── outputs/                            # figures, tables, metrics (git-ignored)
```

---

## Quick start

```bash
# 1. Environment (needs Java 8/11/17 for Spark + Python 3.9+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get data: download the Kaggle CSV to data/raw/ecommerce_churn.csv
#    (or skip — a synthetic 10K sample is auto-generated if missing)

# 3. Run the whole pipeline
python -m src.run_pipeline
```

Run individual stages instead:

```bash
python -m src.generate_sample_data --rows 10000   # optional
python -m src.pipeline_01_data_cleaning
python -m src.pipeline_02_eda
python -m src.pipeline_03_rfm
python -m src.pipeline_04_churn_model
```

Or use the `Makefile`:

```bash
make install   # pip install -r requirements.txt
make data      # generate synthetic sample
make run       # full pipeline
```

---

## What each stage produces

**Stage 1 — Cleaning** (`data/processed/clean/`): de-duplicated, median-imputed,
category-normalised Parquet. Prints before/after null reports and the churn
split.

**Stage 2 — EDA** (`outputs/figures/`, `outputs/tables/`): overall churn rate,
mean features by churn group, churn rate by `Complain` / `SatisfactionScore` /
`MaritalStatus` / `PreferedOrderCat` / `CityTier`, tenure distribution, and a
correlation heatmap.

**Stage 3 — RFM** (`outputs/tables/rfm_segment_summary.csv`,
`rfm_customer_segments.csv`): quartile-based R/F/M scores and segments
(Champions / Loyal / Potential / At Risk / Lost) with churn rate per segment.

**Stage 4 — Modelling** (`outputs/model_metrics.json`,
`outputs/tables/feature_importances.csv`): Logistic Regression, Random Forest
and GBT compared on accuracy, precision, recall, F1 and ROC-AUC, plus Random
Forest feature importances.

---

## RFM proxy mapping

This dataset has **no transaction timestamps**, so classic RFM uses the closest
available proxies (stated openly in the write-up):

| RFM | Proxy column | Direction |
|---|---|---|
| Recency | `DaySinceLastOrder` | lower = better (recent) |
| Frequency | `OrderCount` | higher = better |
| Monetary | `CashbackAmount` | higher = better (spend proxy) |

---

## "Big data" justification (10K rows)

10,000 rows is modest, so Spark is justified by **pipeline design, not raw
volume**: the same code runs unchanged from a laptop (`local[*]`) to a cluster,
uses Spark DataFrames/MLlib and lazy evaluation, and reads/writes partitioned
Parquet. It is a **scalable design** that would handle millions of rows without
rewrites. This is called out honestly in the Big Data section of the report.

---

## Notes before submitting

- Confirm the real Kaggle column names against `src/config.py` and update if
  they differ (Objective 1).
- State in the write-up whether the final dataset used is real or synthetic.
- Add real citations — especially for the "retaining customers is cheaper than
  acquiring new ones" claim in the introduction.
