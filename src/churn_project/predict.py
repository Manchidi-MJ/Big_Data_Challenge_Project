"""Prediction helper: load the saved pipeline and score new customers.

Usage (from project root):
    python -m churn_project.predict --input path/to/new_customers.csv \
        --output outputs/tables/predictions.csv

The input CSV must contain the feature columns used by the primary
(leakage-resistant, Experiment B) model. Missing feature columns raise a clear
error. The output adds:
    churn_probability  -> P(churn) in [0, 1]
    churn_prediction   -> 0/1 at a 0.5 threshold

IMPORTANT: the probability is the model's estimate that a customer matches the
historical churn pattern in this snapshot. It is NOT a guarantee that the
customer will leave, and the dataset's churn label is itself a snapshot rule
(days_since_last_purchase > 90), not a verified future outcome.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import joblib
import pandas as pd

from .config import CONFIG

DEFAULT_MODEL = os.path.join(CONFIG.path("models"), "churn_model_experiment_B.joblib")


def load_model(model_path: str = DEFAULT_MODEL) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run `python -m churn_project.run_all` first."
        )
    return joblib.load(model_path)


def predict(df: pd.DataFrame, model_path: str = DEFAULT_MODEL, threshold: float = 0.5) -> pd.DataFrame:
    bundle = load_model(model_path)
    pipeline = bundle["pipeline"]
    feature_cols: List[str] = list(bundle["numeric"]) + list(bundle["categorical"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input is missing required feature columns for the model: {missing}. "
            f"Expected: {feature_cols}"
        )

    X = df[feature_cols].copy()
    proba = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= threshold).astype(int)
    return out


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with customer feature columns")
    parser.add_argument("--output", default="outputs/tables/predictions.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = predict(df, model_path=args.model, threshold=args.threshold)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions -> {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
