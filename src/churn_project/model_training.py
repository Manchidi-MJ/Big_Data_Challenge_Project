"""Model training (Phase 5).

Builds leakage-safe scikit-learn pipelines. All preprocessing (impute, scale,
one-hot) lives INSIDE the pipeline so it is fit on training folds only — this
prevents test-set information from leaking during cross-validation and the final
fit. ``customer_id`` and raw dates are already excluded upstream in build_xy.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, Config


def make_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    """Numeric: median-impute + standardise. Categorical: impute + one-hot."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_models(
    numeric: List[str],
    categorical: List[str],
    config: Config = CONFIG,
    class_weight: str | None = "balanced",
) -> Dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines sharing the same preprocessor.

    class_weight='balanced' is applied to LR and RF because the training target
    is imbalanced (~38% churn). The DummyClassifier is the baseline.
    """
    seed = config.random_seed

    def pipe(estimator):
        return Pipeline(
            steps=[
                ("prep", make_preprocessor(numeric, categorical)),
                ("clf", estimator),
            ]
        )

    return {
        "DummyClassifier": pipe(
            DummyClassifier(strategy="most_frequent", random_state=seed)
        ),
        "LogisticRegression": pipe(
            LogisticRegression(
                max_iter=2000, class_weight=class_weight, random_state=seed
            )
        ),
        "RandomForest": pipe(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=seed,
            )
        ),
    }


def split_data(
    X: pd.DataFrame, y: pd.Series, config: Config = CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with a fixed seed."""
    test_size = float(config.modelling["test_size"])
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=config.random_seed,
    )
