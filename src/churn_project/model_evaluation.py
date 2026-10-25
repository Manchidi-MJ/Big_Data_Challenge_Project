"""Model evaluation (Phase 5/6): metrics, curves, comparison table, importance."""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

from .config import CONFIG, Config
from .reporting import save_figure


def _proba(model, X) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return (d - d.min()) / (d.max() - d.min() + 1e-12)
    return None


def evaluate_model(name: str, model, X_test, y_test) -> Dict[str, float]:
    """Compute the full metric suite for one fitted model on the test set."""
    y_pred = model.predict(X_test)
    y_score = _proba(model, X_test)

    row = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    if y_score is not None and len(np.unique(y_test)) > 1:
        row["roc_auc"] = round(roc_auc_score(y_test, y_score), 4)
        row["pr_auc"] = round(average_precision_score(y_test, y_score), 4)
    else:
        row["roc_auc"] = np.nan
        row["pr_auc"] = np.nan
    return row


def cross_validate_f1(model, X_train, y_train, config: Config = CONFIG) -> float:
    """Mean CV F1 on the training set (used for model selection, not test-set)."""
    folds = int(config.modelling["cv_folds"])
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1", n_jobs=-1)
    return round(float(scores.mean()), 4)


def confusion_df(model, X_test, y_test) -> pd.DataFrame:
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["actual_retained", "actual_churned"],
        columns=["pred_retained", "pred_churned"],
    )


def plot_confusion(model, X_test, y_test, name: str, config: Config = CONFIG) -> str:
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Retained", "Churned"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Retained", "Churned"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    path = save_figure(fig, f"cm_{name.lower()}.png", config)
    plt.close(fig)
    return path


def plot_roc_pr(models: Dict[str, object], X_test, y_test, suffix: str,
                config: Config = CONFIG) -> Dict[str, str]:
    """ROC and PR curves for all models that expose scores."""
    # ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, m in models.items():
        s = _proba(m, X_test)
        if s is None or len(np.unique(y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves ({suffix})"); ax.legend(loc="lower right", fontsize=8)
    roc_path = save_figure(fig, f"roc_{suffix}.png", config)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, m in models.items():
        s = _proba(m, X_test)
        if s is None or len(np.unique(y_test)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_test, s)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, s):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curves ({suffix})"); ax.legend(loc="lower left", fontsize=8)
    pr_path = save_figure(fig, f"pr_{suffix}.png", config)
    plt.close(fig)
    return {"roc": roc_path, "pr": pr_path}


def permutation_importance_df(
    model, X_test, y_test, config: Config = CONFIG, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance on the test set (model-agnostic)."""
    r = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        random_state=config.random_seed, scoring="f1", n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {"feature": X_test.columns,
             "importance_mean": r.importances_mean.round(5),
             "importance_std": r.importances_std.round(5)}
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(imp: pd.DataFrame, name: str, config: Config = CONFIG,
                            top: int = 15) -> str:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#2a9d8f")
    ax.set_title(f"Permutation importance — {name}")
    ax.set_xlabel("Mean F1 decrease when feature is shuffled")
    path = save_figure(fig, f"importance_{name.lower()}.png", config)
    plt.close(fig)
    return path
