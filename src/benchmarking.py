"""Shared cross-validation benchmarking infrastructure.

Extends the frozen Build 2 validation harness (src/validation.py) with
out-of-fold prediction storage and test-prediction averaging, without
prescribing how any individual model is built or fit. Each gradient-
boosting model in Build 3 supplies a small fold callback (model
construction, categorical handling, early stopping); this module owns
only what is genuinely shared: fold generation, ROC AUC scoring, OOF
storage, and averaging test predictions across folds.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.validation import get_cv_splitter

FoldFitPredict = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[pd.DataFrame]],
    tuple[np.ndarray, Optional[np.ndarray], Optional[int]],
]


@dataclass
class CVBenchmarkResult:
    """Result of running the shared CV harness for one model.

    `best_iterations` is populated only when the fold callback reports an
    early-stopping best iteration; it is `None` for models without early
    stopping.
    """

    fold_scores: list[float]
    cv_mean: float
    cv_std: float
    oof_predictions: np.ndarray
    test_predictions: Optional[np.ndarray] = None
    best_iterations: Optional[list[int]] = None


def run_cv_benchmark(
    fit_predict_fold: FoldFitPredict,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> CVBenchmarkResult:
    """Runs the frozen 5-fold StratifiedKFold harness for one model.

    `fit_predict_fold(X_train, y_train, X_val, y_val, X_test)` must return
    `(val_probabilities, test_probabilities_or_None, best_iteration_or_None)`.
    Model construction, categorical handling, and early stopping all live
    inside that callback — this function only generates folds, stores OOF
    predictions, scores each fold, and averages test predictions across
    folds.
    """
    splitter = get_cv_splitter()
    oof_predictions = np.zeros(len(X))
    fold_scores: list[float] = []
    test_fold_predictions: list[np.ndarray] = []
    best_iterations: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        val_proba, test_proba, best_iteration = fit_predict_fold(
            X_train, y_train, X_val, y_val, X_test
        )

        oof_predictions[val_idx] = val_proba
        fold_auc = roc_auc_score(y_val, val_proba)
        fold_scores.append(fold_auc)
        if verbose:
            print(f"fold {fold_idx}: ROC AUC = {fold_auc:.5f}")

        if test_proba is not None:
            test_fold_predictions.append(test_proba)
        if best_iteration is not None:
            best_iterations.append(best_iteration)

    test_predictions = (
        np.mean(test_fold_predictions, axis=0) if test_fold_predictions else None
    )

    return CVBenchmarkResult(
        fold_scores=fold_scores,
        cv_mean=float(np.mean(fold_scores)),
        cv_std=float(np.std(fold_scores)),
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        best_iterations=best_iterations or None,
    )
