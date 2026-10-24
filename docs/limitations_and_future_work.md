# Limitations and Future Work (Objective 7)

## Limitations

### Data
- **Scale vs "big data".** The dataset is ~10,000 rows. Spark is justified by
  scalable *design*, not data volume; genuinely large data would be needed to
  demonstrate performance benefits.
- **No transaction timestamps.** Classic RFM is approximated with proxies
  (`DaySinceLastOrder`, `OrderCount`, `CashbackAmount`). True recency/frequency
  from an event log would be more faithful.
- **Possibly synthetic / curated.** Confirm the dataset's provenance; a
  synthetic or heavily curated set may show cleaner relationships than
  real-world data. State this in the write-up (Objective 1).
- **Snapshot, not longitudinal.** One row per customer at a point in time means
  we cannot model churn *timing* or behaviour trajectories.
- **Churn definition is given, not derived** — we cannot audit how the label
  was created.

### Modelling
- **Class imbalance.** Churners are the minority, so accuracy is misleading;
  ROC-AUC, precision, recall and F1 are reported, but resampling / class
  weighting was not applied.
- **No hyperparameter tuning.** Models use reasonable defaults; cross-validated
  grid search would likely improve and better validate performance.
- **Importances are correlational**, not causal — they indicate association
  with churn, not that changing a feature causes retention.
- **Proxy leakage risk.** Some features (e.g. recency) may partly encode the
  outcome; worth checking for target leakage.

## Future work
- Acquire a larger, transaction-level dataset to use true RFM and justify big
  data tooling on volume.
- Add cross-validation and hyperparameter tuning (`CrossValidator` /
  `ParamGridBuilder`), and handle imbalance (class weights / SMOTE-style
  oversampling).
- Add model explainability (e.g. SHAP) for per-customer churn reasons.
- Deploy as a scheduled scoring job producing a ranked at-risk list, and
  A/B-test the retention actions in `recommendations.md` to measure real uplift.
- Engineer richer features (tenure buckets, interaction terms, spend velocity).
