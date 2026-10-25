# E-Commerce Customer Churn & Behaviour: Big Data Analytics, RFM Customer Segmentation and Machine Learning-Based Churn Prediction

---

## Title page

**Project title:** E-Commerce Customer Churn & Behaviour: Big Data Analytics, RFM
Customer Segmentation and Machine Learning-Based Churn Prediction

**Module:** Big Data Analytics and Machine Learning

**Team and contributions:**

| Team member | Role | Primary contribution |
|---|---|---|
| Mkhawana Thamelia Hleketani | Data Engineer / Big Data | Data source verification, data quality, cleaning, PySpark pipeline, data dictionary, Big Data section |
| Burgos Carrera, Maria de Los Angeles | Data Analyst / Customer Behaviour | EDA, RFM segmentation, customer behaviour, churn analysis, visualisations, business insights |
| Manchidi Magaseng John | Machine Learning | Churn problem definition, feature preparation, model training/evaluation/comparison, interpretation, prediction |

*The contribution table reflects the planned division of responsibilities in the
project workflow. The work is integrated into one coherent project; each member
should review and be able to defend the sections attributed to them.*

**Date:** October 2026

---

## 1. Abstract

This project analyses the *E-Commerce Customer Churn 2025 – 10K* dataset
(Kaggle, MIT licence) — a verified **10,000-row, 15-column** customer snapshot —
to understand churn behaviour, segment customers with RFM analysis, and assess
how well machine learning can classify churn. The overall churn rate is
**37.71%** (3,771 churned, 6,229 retained).

The central finding is methodological. A mandatory leakage audit showed that the
churn label is a **deterministic function of a single field**: every churned
customer has `days_since_last_purchase > 90` and every retained customer has
`days_since_last_purchase ≤ 90`, so a threshold at 90 reproduces the label with
**100% accuracy**. A model that includes this field (or the equivalent purchase
dates) trivially scores ~100% — it merely restates the label. When these
target-defining fields are removed, a leakage-resistant Random Forest performs
no better than guessing the majority class (ROC-AUC ≈ 0.54, F1 ≈ 0.37), because
all other behavioural features correlate with churn at |r| < 0.12. RFM
segmentation corroborates this: at-risk segments (long recency) carry ~94% churn
while recent segments carry ~0%.

The practical conclusion is that this dataset supports *describing* who is
currently labelled churned (customers who lapsed beyond 90 days) but does **not**
support honestly *predicting future* churn from other behaviour. The most
valuable deliverable is the leakage audit itself and the reproducible,
leakage-aware pipeline built around it. All results reported here were computed
by the project's own code; none are copied from the literature.

---

## 2. Introduction

E-commerce platforms accumulate behavioural and transactional data that can be
analysed to understand loyalty and churn. Identifying customers at risk of
leaving can help a business target retention effort. This project turns one
public customer dataset into measurable findings and practical, clearly-bounded
recommendations through data engineering, exploratory analysis, RFM segmentation
and machine learning.

The customer-retention literature (e.g. Reichheld & Sasser, 1990; Neslin et al.,
2006) has long argued that retaining existing customers is valuable. The
frequently-repeated claim that *retaining customers is generally cheaper than
acquiring new ones* is context-dependent, so we state it cautiously:
retention is widely regarded as an important lever for profitability, and the
magnitude of any cost advantage varies by industry and is not a universal law.

---

## 3. Problem statement

Online retailers may struggle to identify which customers are at risk of leaving
before those customers disengage. Records contain several behavioural
indicators, but the relationship between those indicators and churn is not
obvious without structured analysis. Without it, a business may misallocate
retention resources. This project analyses behaviours associated with churn,
segments customers via RFM where the real fields support it, and compares
machine-learning classifiers — while explicitly auditing whether the churn label
can be trusted as a prediction target.

---

## 4. Main research question

> Which customer behaviours and characteristics are most strongly associated
> with churn in e-commerce, and how accurately can machine learning predict
> whether a customer will churn?

---

## 5. Supporting research questions

1. What proportion of customers churn, and how do churned and retained customers
   differ?
2. Which RFM-based customer segments are most at risk of churning?
3. Which machine-learning model predicts churn best, and which features matter
   most?

---

## 6. Research objectives

1. Select and document the dataset (source, size, features, provenance).
2. Clean and preprocess the data via a reproducible PySpark pipeline.
3. Perform EDA on churn rates and feature–churn relationships.
4. Segment customers with RFM and interpret the segments.
5. Build and compare ML classifiers with appropriate metrics.
6. Connect behavioural analysis and model results to retention recommendations.
7. Document data/model limitations and recommend future work.

---

## 7. Brief literature review and theoretical background

**Customer churn and retention.** Churn modelling frames retention as a binary
classification problem over customer features (Neslin et al., 2006). Retention
has long been treated as a profitability lever (Reichheld & Sasser, 1990),
though cost claims are context-dependent.

**RFM.** RFM scores customers on Recency, Frequency and Monetary value and is
associated with Hughes (1994); the common implementation bins each dimension into
five equal-frequency groups (Kohavi & Parekh, 2004). Segment labels such as
Champions and At Risk are interpretive conventions rather than a fixed scientific
taxonomy.

**Classification and evaluation.** Logistic Regression (Hosmer et al., 2013) and
Random Forests (Breiman, 2001) are standard baselines, implemented here with
scikit-learn (Pedregosa et al., 2011). On imbalanced data, ROC-AUC can be
optimistic and precision–recall analysis is more informative (Davis & Goadrich,
2006; Saito & Rehmsmeier, 2015).

**Data leakage.** Leakage occurs when information about the target enters the
model that would not legitimately be available at prediction time, producing
misleadingly strong results (Kaufman et al., 2012). The recommended defence is a
clear *learn–predict separation*. This concept is central to the present project.

**Distributed processing.** Apache Spark provides a unified engine for scalable
data processing (Zaharia et al., 2016). We use PySpark to demonstrate a scalable
*design*, not because 10k rows require it.

---

## 8. Dataset description and provenance

| Field | Value |
|---|---|
| Name | E-Commerce Customer Churn 2025 – 10K |
| Source / URL | Kaggle — <https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k> |
| Author | Mohammad Maaz (`mohammadmaaz052`) |
| Licence | **MIT** (from Kaggle API metadata) |
| Retrieved | 2026-01 |
| File | `ecommerce_customer_churn_10k.csv` (≈ 950 KB; 207 KB zipped) |
| Rows × columns | **10,000 × 15** (verified by inspection) |
| One row per customer | Yes (no duplicate rows or IDs) |
| Missing values | None |
| Target | `churned` (1 = churned, 0 = retained) |
| Real or synthetic | **Unverified.** The author provides no description; provenance could not be confirmed. Distributional patterns are consistent with synthetic generation, but this is not established. |

The provisional column list in the original brief did **not** match the file.
The verified schema is used throughout (see §10 and `reports/data_dictionary.md`).

---

## 9. Data dictionary

The full data dictionary is in `reports/data_dictionary.md`. Summary of roles:

* **Identifier:** `customer_id` (excluded from features).
* **Dates:** `signup_date`, `last_purchase_date` (the latter is leakage; used only
  to derive tenure, never as a raw feature).
* **Target-defining / leakage:** `days_since_last_purchase`.
* **Features:** `account_age_days`, `age`, `country`, `gender`, `device_type`,
  `favorite_category`, `total_orders`, `total_spent_usd`, `avg_order_value`,
  `is_premium_member`.
* **Target:** `churned`.

---

## 10. Why PySpark is used, and an honest assessment of data size

10,000 rows (≈ 950 KB) is a **small** tabular dataset; pandas processes it
instantly and it is **not** "big data" in any meaningful sense. PySpark is
included to demonstrate a *scalable data-processing design*: the same
transformation code would run unchanged on a cluster against millions of
customers. The PySpark path (read with explicit schema → dedupe → impute →
normalise → derive tenure → aggregate → persist) executes successfully and
produces the same cleaned dataset as the pandas path. On Windows the native
Parquet writer requires `winutils.exe`/`HADOOP_HOME`; when unavailable the
pipeline logs this and writes the canonical cleaned CSV instead, while still
performing all Spark processing.

---

## 11. Methodology and project architecture

The project separates reusable logic (`src/churn_project/`) from orchestration
(`run_all.py`, notebooks) and outputs (`outputs/`). Full detail is in
`docs/methodology.md`. Phases: data understanding → cleaning/PySpark → EDA → RFM
→ leakage audit → modelling → evaluation → persistence/prediction → reporting.

---

## 12. Data understanding and quality assessment

Verified quality profile (`outputs/tables/data_quality_summary.json`):

* 10,000 rows × 15 columns; one row per customer.
* **0** missing values; **0** duplicate rows; **0** duplicate customer IDs.
* No impossible values detected (ages 18–69; non-negative spend/orders; binary
  flags valid).
* Target distribution: 6,229 retained (62.29%), 3,771 churned (37.71%);
  majority:minority ≈ 1.65:1.
* Redundancy: `total_spent_usd` and `avg_order_value` correlate at 0.85 (expected,
  since average ≈ total / orders).

---

## 13. Data cleaning and preprocessing

Because the data is already clean, cleaning is defensive and fully logged
(`outputs/tables/pandas_cleaning_log.json`, `pyspark_cleaning_log.json`):

* Before and after cleaning: **10,000 → 10,000** rows (nothing removed).
* No duplicates removed; no missing values imputed.
* Category strings trimmed; `account_age_days` confirmed equal to
  `last_purchase_date − signup_date` for 100% of rows.
* `customer_id` retained for identification but excluded from modelling features.

The pandas and PySpark paths implement the same steps and agree on before/after
counts.

---

## 14. Exploratory data analysis

(Figures in `outputs/figures/`; tables in `outputs/tables/`.)

* **Churn distribution** (`01_churn_distribution.png`): 37.7% churn.
* **Numeric features by churn** (`eda_numeric_by_churn.csv`): the only large
  difference is recency — retained ≈ 8.8 days vs churned ≈ 300.6 days since last
  purchase (a 291-day gap). Tenure differs modestly (retained ≈ 257 vs churned
  ≈ 214 days). Age, orders, spend and average order value are **nearly identical**
  across the two groups.
* **Categorical churn rates** (`04_categorical_churn_rates.png`): churn rates are
  flat across country, gender, device and favourite category — no categorical
  feature separates churners.
* **Correlation heatmap** (`05_correlation_heatmap.png`): `churned` correlates
  0.84 with `days_since_last_purchase` and |r| < 0.12 with everything else.

**Key pattern:** churn in this dataset is explained almost entirely by purchase
recency; no other recorded behaviour meaningfully distinguishes churners.

---

## 15. RFM methodology and customer segmentation

RFM uses real fields (`outputs/tables/rfm_meta.json`):

| Dimension | Field | Direction |
|---|---|---|
| Recency | `days_since_last_purchase` | lower is better |
| Frequency | `total_orders` | higher is better |
| Monetary | `total_spent_usd` | higher is better |

There is no separate `recency` or `frequency` column, so
`days_since_last_purchase` and `total_orders` are used. Each dimension is scored
1–5 via rank-based quantile binning (robust to ties); all three used the full 5
bins. Segment summary (`rfm_segment_summary.csv`):

| Segment | % of base | Avg recency (days) | Avg freq | Avg spend (USD) | Churn rate |
|---|---:|---:|---:|---:|---:|
| Needs Attention | 31.8% | 290.4 | 4.1 | 242.7 | **94.4%** |
| Loyal Customers | 22.4% | 1.0 | 5.4 | 219.5 | 0.0% |
| About to Sleep | 20.0% | 16.1 | 9.0 | 316.9 | 0.0% |
| Champions | 11.3% | 1.0 | 15.3 | 632.9 | 0.0% |
| At Risk | 8.3% | 278.9 | 19.5 | 585.5 | **93.7%** |
| Potential Loyalists | 6.3% | 1.0 | 1.0 | 45.5 | 0.0% |

Churn aligns perfectly with recency: lapsed segments (Needs Attention, At Risk)
carry ~94% churn; recent segments carry ~0%. This is expected given the label
definition, and it makes RFM a clean descriptive lens on the current base.

---

## 16. Machine-learning methodology

Binary classification of `churned`. Preprocessing lives inside scikit-learn
Pipelines (median-impute + standardise numeric; impute + one-hot categorical),
fit on training folds only. Stratified 80/20 split, seed 42, 5-fold CV F1 for
model selection, `class_weight='balanced'` for LR and RF. Models: DummyClassifier
(baseline), Logistic Regression, Random Forest.

---

## 17. Leakage analysis and feature engineering

**Mandatory audit** (`outputs/metrics/leakage_audit.json`):

* Threshold reconstruction: `churned == (days_since_last_purchase > 90)` at
  **100% accuracy**.
* Per-class ranges: retained ∈ [1, 90] days; churned ∈ [91, 719] days.
* Target correlations: `days_since_last_purchase` = 0.84; all others |r| < 0.12.

**Verdict:** `days_since_last_purchase` is **target-defining leakage**; the date
columns encode the same recency signal and are also leakage.

**Feature engineering for two experiments:**

* **Experiment A (exploratory):** all plausible predictors, including the leakage
  field. Identifiers and raw dates always excluded.
* **Experiment B (primary, leakage-resistant):** drops `days_since_last_purchase`
  and never uses raw dates or `customer_id`.

---

## 18. Model training and experimental design

Both experiments use the identical split and preprocessing; only the feature set
differs. The leakage-resistant Experiment B model is designated primary a priori,
per good practice (Kaufman et al., 2012).

---

## 19. Model evaluation and comparison

From `outputs/metrics/model_comparison.csv` (test set):

| Exp | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV-F1 (train) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | RandomForest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| A | LogisticRegression | 0.994 | 0.983 | 1.000 | 0.992 | 1.000 | 1.000 | 0.994 |
| A | DummyClassifier | 0.623 | 0.000 | 0.000 | 0.000 | 0.500 | 0.377 | 0.000 |
| B | LogisticRegression | 0.503 | 0.390 | 0.566 | 0.462 | 0.542 | 0.407 | 0.479 |
| B | RandomForest | 0.553 | 0.396 | 0.353 | 0.373 | 0.542 | 0.409 | 0.365 |
| B | DummyClassifier | 0.623 | 0.000 | 0.000 | 0.000 | 0.500 | 0.377 | 0.000 |

Curves and confusion matrices: `roc_experiment_*.png`, `pr_experiment_*.png`,
`cm_rf_exp*.png`. Permutation importance:
`feature_importance_experiment_*.csv`. In Experiment B every importance is tiny
(≤ 0.024 mean F1 drop) with overlapping error bars — no feature stands out.

---

## 20. Results and discussion

* **Experiment A** is a sanity check, not a result: ~100% performance comes
  entirely from the field that *defines* churn. We do **not** report it as
  predictive success.
* **Experiment B** is the honest estimate. ROC-AUC ≈ 0.54 and F1 ≈ 0.37–0.46 are
  barely above a majority-class baseline (accuracy 0.623, F1 0.0). The modest
  improvement in recall/F1 over Dummy comes at the cost of many false positives.
* **Precision vs recall:** for retention, a false negative (missing a real
  churner) is usually costlier than a false positive (contacting someone who
  would have stayed). Experiment B's low recall and precision mean it would both
  miss churners and waste outreach — it is not fit for targeting.
* **Credibility check:** the near-perfect Experiment A scores triggered the
  leakage investigation (as recommended when scores look too good). Duplicates,
  train/test contamination and target definition were all checked; the cause is
  the label definition, not a pipeline bug.
* **Answering the research questions:** (1) 37.7% churn; churned and retained
  differ essentially only in recency. (2) RFM "Needs Attention" and "At Risk"
  segments are most at risk (~94% churn). (3) No model genuinely predicts churn
  from non-recency behaviour; the "best" apparent model is an artefact of
  leakage.

---

## 21. Business insights and retention recommendations

Full detail in `docs/recommendations.md`. Headlines, each tied to a computed
result and kept separate from causal claims:

* **R1.** Re-engage lapsing customers *before* day 90 (recency is the whole
  signal).
* **R2.** Protect Champions/Loyal customers whose recency is currently low.
* **R3.** Prioritise the high-value "At Risk" segment for win-back.
* **R4.** Do **not** deploy a predictive churn model from this dataset; collect
  forward-labelled, leakage-free data first.

---

## 22. Limitations

See `docs/limitations.md`. In brief: small snapshot dataset; unverified
provenance; a target that is a deterministic recency rule, which prevents honest
future-churn prediction; associations only, not causation.

---

## 23. Conclusion

The project delivers a complete, tested, reproducible pipeline: verified data,
quality report, PySpark + pandas cleaning, EDA, true RFM segmentation, a
leakage-aware two-experiment modelling design, saved model and prediction script.
Its most important result is the **leakage finding**: churn in this dataset is
defined by `days_since_last_purchase > 90`, so apparent 100% models are
restating the label, and once recency is removed no other behaviour predicts
churn. The dataset is therefore suitable for *describing* the current base and
for *teaching* leakage-aware practice, but not for building an honest
future-churn predictor.

---

## 24. Future work

Collect event-level data with a proper future-observation window; add
pre-prediction behavioural signals; verify dataset provenance or switch to a
documented real dataset; run a controlled A/B retention experiment for causal
evidence; scale-test the PySpark pipeline on genuinely large data. (Details in
`docs/limitations.md`.)

---

## 25. AI usage disclosure

Portions of this project — code scaffolding, documentation drafting and
refactoring — were produced with the assistance of an AI coding assistant, under
the team's direction and review. All **analytical results** (data profiling,
cleaning counts, EDA statistics, RFM segments, model metrics, the leakage audit)
were produced by executing the project's own code on the real dataset and were
verified against the saved outputs in `outputs/`. No metrics, findings or
citations were invented or copied from other studies. Team members are
responsible for understanding and being able to defend all submitted material.
*(This disclosure is a template; adapt it to your institution's specific AI-use
policy.)*

---

## 26. References

See `reports/references.md` for the full, numbered bibliography. Every reference
there is cited in this report, and every in-text citation corresponds to a real,
consulted source.

---

## 27. Appendices

**A. Reproduction commands**

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m churn_project.run_all            # full pipeline incl. PySpark
python -m pytest                           # run the test suite
python -m churn_project.predict --input data/interim/sample_new_customers.csv \
    --output outputs/tables/sample_predictions.csv
```

**B. Generated artefacts**

* Tables: `outputs/tables/` (quality, EDA, RFM, cleaning logs).
* Metrics: `outputs/metrics/model_comparison.csv`, `leakage_audit.json`.
* Figures: `outputs/figures/` (14 PNGs).
* Model: `outputs/models/churn_model_experiment_B.joblib` (primary, leakage-resistant).

**C. Environment**

Python 3.11.9; PySpark 3.5.1; pandas 2.3.x; scikit-learn 1.x; OpenJDK 17;
Windows. Random seed 42 throughout.

**D. Key verified numbers (for defence)**

| Quantity | Value |
|---|---|
| Rows × cols | 10,000 × 15 |
| Churn rate | 37.71% |
| Leakage threshold / accuracy | days_since > 90 / 100% |
| Experiment B RF ROC-AUC | 0.542 |
| Experiment B LR F1 | 0.462 |
| Baseline accuracy (Dummy) | 0.623 |
