"""XGBoost hyperparameter tuning infrastructure for Build 6.

Wraps `src.benchmarking.run_cv_benchmark` with a parameterized XGBoost fold
callback so every tuning candidate runs through the exact same frozen
5-fold `StratifiedKFold` harness used for E004/E006, differing only in
model configuration. Nothing here changes `src.boosting_models.xgboost_fold`
(used by prior builds) or the CV/preprocessing framework.

`E006_XGB_PARAMS` is the exact reconstructed E006 configuration (see
`notebooks/06_xgboost_tuning.ipynb`, Section 2) and is the tuning baseline
every candidate is compared against.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.benchmarking import CVBenchmarkResult, run_cv_benchmark
from src.config import RANDOM_SEED
from src.validation import get_cv_splitter

EARLY_STOPPING_ROUNDS: int = 50

E006_XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_estimators": 800,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
}


def make_xgboost_fold(params: dict, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    """Builds a `src.benchmarking.FoldFitPredict` callback for `params`.

    Matches `src.boosting_models.xgboost_fold`'s categorical/early-stopping
    behavior exactly, but takes the model configuration as an argument
    instead of hardcoding it, so it plugs into the existing
    `run_cv_benchmark` harness for any candidate parameter set.
    """

    def fold(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: Optional[pd.DataFrame],
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        model = XGBClassifier(
            **params,
            n_jobs=-1,
            enable_categorical=True,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1] if X_test is not None else None
        return val_proba, test_proba, model.best_iteration

    return fold


def run_xgboost_trial(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    verbose: bool = True,
) -> CVBenchmarkResult:
    """Runs one XGBoost parameter set through the frozen full 5-fold CV harness.

    This is the authoritative result for any formal Build 6 experiment —
    every number entered in `experiments/experiments.csv` must come from
    this function, not from `screen_xgboost_single_fold`.
    """
    fold_fn = make_xgboost_fold(params, early_stopping_rounds)
    return run_cv_benchmark(fold_fn, X, y, X_test=X_test, verbose=verbose)


@dataclass
class ScreeningResult:
    """Result of a single-fold screening trial. Never a substitute for full CV."""

    fold_auc: float
    best_iteration: Optional[int]


def screen_xgboost_single_fold(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> ScreeningResult:
    """Coarse single-fold screen using fold 1 of the frozen 5-fold splitter.

    Deterministic (same splitter, same seed, same fold every call) and
    cheap relative to `run_xgboost_trial`, used only to eliminate obviously
    poor parameter regions before committing to a full 5-fold CV run.
    Screening results are never recorded as formal experiments and never
    compared against full-CV numbers as if they were equivalent.
    """
    splitter = get_cv_splitter()
    train_idx, val_idx = next(splitter.split(X, y))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_fn = make_xgboost_fold(params, early_stopping_rounds)
    val_proba, _, best_iteration = fold_fn(X_train, y_train, X_val, y_val, None)

    from sklearn.metrics import roc_auc_score

    return ScreeningResult(
        fold_auc=float(roc_auc_score(y_val, val_proba)),
        best_iteration=best_iteration,
    )


def paired_fold_deltas(candidate_scores: list[float], control_scores: list[float]) -> dict:
    """Fold-by-fold delta summary of `candidate_scores` vs `control_scores`.

    Mirrors the paired-comparison helper used in
    `notebooks/04_feature_engineering.ipynb`, generalized for reuse here.
    """
    deltas = [c - b for c, b in zip(candidate_scores, control_scores)]
    return {
        "mean_delta": float(np.mean(deltas)),
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "folds_improved": int(sum(d > 0 for d in deltas)),
        "fold_deltas": deltas,
    }


def tuning_result_row(
    trial_id: str,
    stage: str,
    formal_experiment: Optional[str],
    params: dict,
    result: CVBenchmarkResult,
    control_scores: Optional[list[float]] = None,
    runtime_minutes: Optional[float] = None,
    decision: str = "",
) -> dict:
    """Builds one row for `outputs/xgboost_tuning_results.csv` from a trial result."""
    deltas = paired_fold_deltas(result.fold_scores, control_scores) if control_scores else None
    best_iterations = result.best_iterations or []
    return {
        "trial_or_experiment_id": trial_id,
        "stage": stage,
        "formal_experiment": formal_experiment or "",
        "learning_rate": params.get("learning_rate"),
        "n_estimators": params.get("n_estimators"),
        "max_depth": params.get("max_depth"),
        "min_child_weight": params.get("min_child_weight"),
        "subsample": params.get("subsample"),
        "colsample_bytree": params.get("colsample_bytree"),
        "reg_alpha": params.get("reg_alpha"),
        "reg_lambda": params.get("reg_lambda"),
        "gamma": params.get("gamma"),
        "cv_mean": round(result.cv_mean, 5),
        "cv_std": round(result.cv_std, 5),
        "delta_vs_control": round(deltas["mean_delta"], 5) if deltas else "",
        "folds_improved": deltas["folds_improved"] if deltas else "",
        "mean_best_iteration": round(float(np.mean(best_iterations)), 1) if best_iterations else "",
        "runtime_minutes": round(runtime_minutes, 2) if runtime_minutes is not None else "",
        "decision": decision,
    }
