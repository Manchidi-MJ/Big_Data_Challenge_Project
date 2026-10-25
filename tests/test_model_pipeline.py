"""Tests for feature preparation, leakage audit, modelling and persistence."""
from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pandas as pd

from churn_project import feature_engineering as fe
from churn_project import model_evaluation as me
from churn_project import model_training as mt


def test_customer_id_excluded_from_features(synthetic_df):
    X, y, feats = fe.build_xy(synthetic_df, "B")
    assert "customer_id" not in X.columns
    assert "signup_date" not in X.columns
    assert "last_purchase_date" not in X.columns
    assert "churned" not in X.columns


def test_experiment_b_drops_leakage_column(synthetic_df):
    X_a, _, _ = fe.build_xy(synthetic_df, "A")
    X_b, _, _ = fe.build_xy(synthetic_df, "B")
    assert "days_since_last_purchase" in X_a.columns
    assert "days_since_last_purchase" not in X_b.columns


def test_leakage_audit_detects_defining_column(synthetic_df):
    audit = fe.audit_leakage(synthetic_df)
    recon = audit["days_since_threshold_reconstruction"]
    assert recon["accuracy"] >= 0.999  # synthetic churn is days_since>90
    assert audit["verdict"]["days_since_last_purchase"] == "TARGET-DEFINING LEAKAGE"


def test_train_test_split_is_stratified(synthetic_df):
    X, y, _ = fe.build_xy(synthetic_df, "B")
    X_tr, X_te, y_tr, y_te = mt.split_data(X, y)
    # Train and test churn rates should be close (stratified).
    assert abs(y_tr.mean() - y_te.mean()) < 0.1
    assert len(X_tr) + len(X_te) == len(X)


def test_models_fit_and_predict(synthetic_df):
    X, y, feats = fe.build_xy(synthetic_df, "A")
    X_tr, X_te, y_tr, y_te = mt.split_data(X, y)
    models = mt.build_models(feats["numeric"], feats["categorical"])
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert len(preds) == len(X_te)
        assert set(np.unique(preds)).issubset({0, 1})


def test_evaluation_metrics_well_formed(synthetic_df):
    X, y, feats = fe.build_xy(synthetic_df, "A")
    X_tr, X_te, y_tr, y_te = mt.split_data(X, y)
    pipe = mt.build_models(feats["numeric"], feats["categorical"])["LogisticRegression"]
    pipe.fit(X_tr, y_tr)
    row = me.evaluate_model("LR", pipe, X_te, y_te)
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= row[metric] <= 1.0


def test_experiment_a_is_near_perfect_due_to_leakage(synthetic_df):
    """Sanity check: with the defining column present, accuracy is ~1.0."""
    X, y, feats = fe.build_xy(synthetic_df, "A")
    X_tr, X_te, y_tr, y_te = mt.split_data(X, y)
    pipe = mt.build_models(feats["numeric"], feats["categorical"])["RandomForest"]
    pipe.fit(X_tr, y_tr)
    row = me.evaluate_model("RF", pipe, X_te, y_te)
    assert row["accuracy"] >= 0.95


def test_model_persistence_roundtrip(synthetic_df):
    X, y, feats = fe.build_xy(synthetic_df, "B")
    X_tr, X_te, y_tr, y_te = mt.split_data(X, y)
    pipe = mt.build_models(feats["numeric"], feats["categorical"])["RandomForest"]
    pipe.fit(X_tr, y_tr)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        joblib.dump({"pipeline": pipe, "numeric": feats["numeric"],
                     "categorical": feats["categorical"]}, path)
        loaded = joblib.load(path)
        preds_before = pipe.predict(X_te)
        preds_after = loaded["pipeline"].predict(X_te)
        assert np.array_equal(preds_before, preds_after)
