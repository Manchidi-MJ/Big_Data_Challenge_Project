"""Feature preparation and the mandatory leakage audit (Phase 5).

Leakage audit summary (verified on the real data)
--------------------------------------------------
The churn label in this dataset is a DETERMINISTIC function of
``days_since_last_purchase``: every retained customer has
``days_since_last_purchase <= 90`` and every churned customer has
``days_since_last_purchase >= 91``. A single threshold at 90 reproduces the
label with 100% accuracy. Because ``days_since_last_purchase`` is just
``snapshot_date - last_purchase_date``, the two date-derived fields encode the
same rule and are therefore target-defining leakage.

Consequences for modelling:
  * Experiment A (all predictors) will trivially achieve ~100% — this is a
    sanity check, not a real predictive result.
  * Experiment B (leakage-resistant) removes the target-defining fields and is
    the PRIMARY model. Its (much lower) performance is the honest estimate of
    how well the *other* behavioural features relate to the churn rule.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG, Config


def audit_leakage(df: pd.DataFrame, config: Config = CONFIG) -> Dict[str, object]:
    """Investigate whether churn is derivable from candidate predictors.

    Checks the exact threshold reconstruction for days_since_last_purchase and
    reports per-class ranges + correlations for the usual suspects.
    """
    tgt = config.target
    result: Dict[str, object] = {}

    # 1. Threshold reconstruction from days_since_last_purchase
    if "days_since_last_purchase" in df.columns:
        dsl = df["days_since_last_purchase"]
        best = {"threshold": None, "accuracy": 0.0}
        for t in range(int(dsl.min()), int(dsl.max()) + 1, 1):
            acc = float(((dsl > t).astype(int) == df[tgt]).mean())
            if acc > best["accuracy"]:
                best = {"threshold": int(t), "accuracy": round(acc, 6)}
            if acc == 1.0:
                break
        result["days_since_threshold_reconstruction"] = best
        result["days_since_by_class"] = (
            df.groupby(tgt)["days_since_last_purchase"]
            .agg(["min", "max", "mean"])
            .round(3)
            .to_dict(orient="index")
        )

    # 2. Correlations of candidate leakage columns with target
    candidates = ["days_since_last_purchase", "account_age_days", "age",
                  "total_orders", "total_spent_usd", "avg_order_value",
                  "is_premium_member"]
    corrs = {}
    for c in candidates:
        if c in df.columns:
            corrs[c] = round(float(df[c].corr(df[tgt])), 4)
    result["target_correlations"] = corrs

    # 3. Verdict
    recon = result.get("days_since_threshold_reconstruction", {})
    result["verdict"] = {
        "days_since_last_purchase": (
            "TARGET-DEFINING LEAKAGE"
            if recon.get("accuracy", 0) >= 0.999
            else "strong predictor (not perfect)"
        ),
        "date_columns": "LEAKAGE (encode the same recency signal as days_since)",
        "note": (
            "Churn is a snapshot rule, not a future event. Removing recency/date "
            "fields gives a leakage-resistant but weaker and more honest model."
        ),
    }
    return result


def get_feature_lists(experiment: str, config: Config = CONFIG) -> Dict[str, List[str]]:
    """Return numeric/categorical feature lists for an experiment.

    experiment == 'A' : all plausible predictors (includes leakage) -> exploratory
    experiment == 'B' : leakage-resistant (excludes target-defining fields) -> primary
    """
    numeric = [c for c in config.numeric_cols]
    categorical = [c for c in config.categorical_cols]

    if experiment.upper() == "B":
        # Drop target-defining leakage numeric fields.
        drop = {"days_since_last_purchase"}
        numeric = [c for c in numeric if c not in drop]
    elif experiment.upper() != "A":
        raise ValueError("experiment must be 'A' or 'B'")
    return {"numeric": numeric, "categorical": categorical}


def build_xy(
    df: pd.DataFrame, experiment: str, config: Config = CONFIG
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """Build X (features) and y (target) for an experiment.

    ``customer_id`` and date columns are always excluded from X. For both
    experiments, raw date columns are never used directly as features.
    """
    feats = get_feature_lists(experiment, config)
    use_cols = feats["numeric"] + feats["categorical"]
    use_cols = [c for c in use_cols if c in df.columns]

    # Safety: never leak the id or the raw dates into X.
    forbidden = {config.id_col, *config.date_cols, config.target}
    use_cols = [c for c in use_cols if c not in forbidden]

    X = df[use_cols].copy()
    y = df[config.target].astype(int).copy()
    return X, y, {"numeric": [c for c in feats["numeric"] if c in use_cols],
                  "categorical": [c for c in feats["categorical"] if c in use_cols]}
