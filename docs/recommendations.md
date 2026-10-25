# Business insights and retention recommendations

Every recommendation below is tied to a **computed result** from this project
(see `outputs/`). Measured findings are kept strictly separate from proposed
business actions. The dataset establishes *associations*, not causation, so no
recommendation should be read as a guarantee that an intervention will reduce
churn.

## What the data actually shows

| Finding | Evidence (source) |
|---|---|
| Churn is a recency rule: every churned customer last purchased > 90 days before the snapshot | `outputs/metrics/leakage_audit.json` (threshold 90 → 100% accuracy) |
| Overall churn rate is 37.7% (3,771 of 10,000) | `outputs/tables/eda_overview.json` |
| Spend, orders, device, payment and demographics barely differ between churned and retained customers | `outputs/tables/eda_numeric_by_churn.csv`, correlation heatmap |
| RFM "Needs Attention" (31.8% of base) and "At Risk" (8.3%) segments have ~94% churn; recent segments have ~0% | `outputs/tables/rfm_segment_summary.csv` |
| Once recency is removed, no behavioural feature predicts churn (ROC-AUC ≈ 0.54) | `outputs/metrics/model_comparison.csv` |

## Recommendations

### R1 — Re-engage long-lapsed customers early, before day 90
* **Motivating result:** retained customers average ~9 days since last purchase;
  churned average ~301 days. The label flips at 90 days.
* **Segment:** customers approaching the 60–90 day recency window (before they
  are labelled churned).
* **Proposed action:** triggered win-back outreach (email/app) at ~45–60 days of
  inactivity.
* **How to measure:** compare the share of targeted customers who make a purchase
  within 30 days against a holdout control group.
* **Uncertainty:** the dataset cannot show that outreach *causes* a return — only
  an experiment can.

### R2 — Protect high-value recent customers (Champions, Loyal)
* **Motivating result:** Champions (11.3% of base) average ~15 orders and ~\$633
  spend with ~0% current churn; Loyal Customers are similar.
* **Segment:** Champions and Loyal Customers.
* **Proposed action:** recognition/loyalty perks to keep recency low.
* **How to measure:** track whether these customers' recency stays under the
  90-day threshold over the next quarter.
* **Uncertainty:** these customers already look healthy; the benefit of extra
  spend on them is unproven here.

### R3 — Prioritise the "Needs Attention" and "At Risk" segments
* **Motivating result:** these two segments hold ~40% of customers and ~94%
  churn; "At Risk" also has high historical frequency/spend (worth saving).
* **Segment:** RFM At Risk (high F/M, lapsed) first, then Needs Attention.
* **Proposed action:** differentiated win-back — stronger incentives for the
  previously high-value "At Risk" group.
* **How to measure:** reactivation rate and 90-day retained rate vs control.
* **Uncertainty:** many may have already churned for reasons absent from the
  data.

### R4 — Do NOT rely on a predictive churn model from this dataset
* **Motivating result:** the only feature that "predicts" churn is the one that
  *defines* it. A leakage-resistant model is no better than guessing the
  majority class (ROC-AUC ≈ 0.54).
* **Proposed action:** before building a production churn model, collect
  event-level data with a proper future-labelling window (see
  `docs/limitations.md`).
* **How to measure:** a future model should beat a majority baseline on a
  genuinely forward-looking, leakage-free test set.
* **Uncertainty:** none — this is a data-availability conclusion, not a modelling
  tweak.

## Honesty note

The strongest, most defensible output of this project is the **leakage finding
itself**: it demonstrates good analytical practice (auditing the target before
trusting a model) and prevents the business from acting on a model that only
restates its own label.
