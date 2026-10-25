# Limitations and future work

## Data limitations

1. **Size.** 10,000 rows is a small, tabular dataset. It is suitable for a
   teaching project but is **not** inherently "big data". PySpark is used here
   to demonstrate a scalable *design*, not because the current volume requires
   it.

2. **Provenance unverified.** The Kaggle author provides no description of how
   the data was produced. We could **not confirm** whether it is real,
   anonymised real, or synthetic. The feature distributions (uniform-looking
   ages, a single snapshot date, a clean recency rule) are consistent with
   synthetic generation, but this is an observation, not a confirmed fact. The
   licence is MIT (per Kaggle API metadata).

3. **Snapshot, not longitudinal.** Each customer appears once, at a single
   snapshot (2026-01-01, implied by `last_purchase_date + days_since`). There
   are no transaction-level records, no time series and no future labels.

## Target-definition (leakage) limitation — the central caveat

The churn label is a **deterministic rule**:
`churned = (days_since_last_purchase > 90)`, reproduced with 100% accuracy by a
single threshold. Consequences:

* A model given `days_since_last_purchase` (or the equivalent dates) will score
  ~100% — this is **not** prediction, it is restating the label.
* After removing those fields (Experiment B), the remaining behavioural features
  carry **almost no signal**: ROC-AUC ≈ 0.54 and F1 ≈ 0.37–0.46, barely above a
  majority-class baseline. All non-recency features correlate with churn at
  |r| < 0.12.
* Therefore this dataset supports *describing* who is currently labelled churned
  (long recency), but it does **not** support genuinely *predicting future*
  churn from other behaviour. Predicting a pre-existing snapshot label is not
  the same as forecasting a future event.

## Modelling limitations

* No hyperparameter search beyond sensible defaults; the leakage-resistant
  signal is too weak for tuning to matter materially.
* Permutation importances in Experiment B are all tiny and within overlapping
  error bars — no feature can be singled out with confidence.
* Business recommendations are grounded in observed associations and the RFM
  split, **not** in causal evidence. The data cannot establish that any
  intervention reduces churn.

## Future work

1. **Obtain event-level data** with timestamps and a clearly defined future
   observation window, so churn can be labelled *after* a feature snapshot — the
   only way to train an honest forward-looking model.
2. **Add richer behavioural signals** (sessions, support tickets, discounts
   redeemed, product returns) that are available *before* the prediction point.
3. **Validate provenance** with the dataset author or replace it with a
   documented real-world dataset.
4. **Run a controlled retention experiment** (A/B test) to move from correlation
   to causal evidence for any recommended intervention.
5. **Scale test** the PySpark pipeline on a genuinely large dataset to
   demonstrate the distributed design in practice.
