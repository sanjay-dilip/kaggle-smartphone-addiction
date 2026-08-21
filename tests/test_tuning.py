"""Tests for src/tuning.py.

Uses the same small synthetic data pattern as tests/test_boosting_models.py
so these run fast without exercising the real competition data.
"""

import numpy as np
import pandas as pd
import pytest

from src.benchmarking import CVBenchmarkResult
from src.config import CATEGORICAL_COLS, NUMERIC_COLS
from src.preprocessing import build_boosting_frame
from src.tuning import (
    E006_XGB_PARAMS,
    make_xgboost_fold,
    paired_fold_deltas,
    run_xgboost_trial,
    screen_xgboost_single_fold,
    tuning_result_row,
)


def _synthetic_data(n_rows: int = 300):
    rng = np.random.default_rng(seed=0)
    raw = {}
    for col in NUMERIC_COLS:
        values = rng.uniform(0, 10, n_rows)
        values[::7] = np.nan
        raw[col] = values
    for col in CATEGORICAL_COLS:
        values = rng.choice(["A", "B", "C"], n_rows).astype(object)
        values[::7] = np.nan
        raw[col] = values
    X = build_boosting_frame(pd.DataFrame(raw))
    y = pd.Series(
        (X[NUMERIC_COLS[0]].fillna(0) > X[NUMERIC_COLS[0]].fillna(0).median())
        .astype(int)
        .values
    )
    return X, y


def test_e006_params_match_known_reconstruction() -> None:
    """The reconstructed E006 config must match the frozen experiments.csv record."""
    assert E006_XGB_PARAMS["learning_rate"] == 0.1
    assert E006_XGB_PARAMS["max_depth"] == 6
    assert E006_XGB_PARAMS["min_child_weight"] == 1
    assert E006_XGB_PARAMS["subsample"] == 0.9
    assert E006_XGB_PARAMS["colsample_bytree"] == 0.9
    assert E006_XGB_PARAMS["reg_alpha"] == 0.0
    assert E006_XGB_PARAMS["reg_lambda"] == 1.0
    assert E006_XGB_PARAMS["n_estimators"] == 800
    assert E006_XGB_PARAMS["tree_method"] == "hist"


def test_make_xgboost_fold_returns_valid_predictions() -> None:
    X, y = _synthetic_data()
    X_train, X_val = X.iloc[:240], X.iloc[240:]
    y_train, y_val = y.iloc[:240], y.iloc[240:]
    fold_fn = make_xgboost_fold(E006_XGB_PARAMS)

    val_proba, test_proba, best_iteration = fold_fn(X_train, y_train, X_val, y_val, None)

    assert val_proba.shape == (len(X_val),)
    assert np.isfinite(val_proba).all()
    assert ((val_proba >= 0) & (val_proba <= 1)).all()
    assert test_proba is None
    assert best_iteration is not None and best_iteration >= 0


def test_run_xgboost_trial_produces_full_cv_result() -> None:
    X, y = _synthetic_data()
    result = run_xgboost_trial(E006_XGB_PARAMS, X, y, verbose=False)

    assert isinstance(result, CVBenchmarkResult)
    assert len(result.fold_scores) == 5
    assert result.best_iterations is not None and len(result.best_iterations) == 5


def test_screen_xgboost_single_fold_is_deterministic() -> None:
    X, y = _synthetic_data()

    first = screen_xgboost_single_fold(E006_XGB_PARAMS, X, y)
    second = screen_xgboost_single_fold(E006_XGB_PARAMS, X, y)

    assert first.fold_auc == pytest.approx(second.fold_auc)
    assert 0.0 <= first.fold_auc <= 1.0
    assert first.best_iteration is not None


def test_paired_fold_deltas_computes_expected_summary() -> None:
    candidate = [0.90, 0.92, 0.89, 0.95, 0.91]
    control = [0.89, 0.92, 0.90, 0.93, 0.90]

    deltas = paired_fold_deltas(candidate, control)

    assert deltas["folds_improved"] == 3
    assert deltas["mean_delta"] == pytest.approx(np.mean(
        [c - b for c, b in zip(candidate, control)]
    ))
    assert deltas["min_delta"] == pytest.approx(-0.01)
    assert deltas["max_delta"] == pytest.approx(0.02)
    assert len(deltas["fold_deltas"]) == 5


def test_tuning_result_row_has_expected_schema() -> None:
    X, y = _synthetic_data()
    result = run_xgboost_trial(E006_XGB_PARAMS, X, y, verbose=False)
    control_scores = [s - 0.01 for s in result.fold_scores]

    row = tuning_result_row(
        "screen-001", "phase1", None, E006_XGB_PARAMS, result,
        control_scores=control_scores, runtime_minutes=1.5, decision="test row",
    )

    expected_keys = {
        "trial_or_experiment_id", "stage", "formal_experiment", "learning_rate",
        "n_estimators", "max_depth", "min_child_weight", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "gamma", "cv_mean",
        "cv_std", "delta_vs_control", "folds_improved", "mean_best_iteration",
        "runtime_minutes", "decision",
    }
    assert set(row.keys()) == expected_keys
    assert row["folds_improved"] == 5
    assert row["runtime_minutes"] == 1.5


def test_tuning_result_row_without_control_leaves_delta_blank() -> None:
    X, y = _synthetic_data()
    result = run_xgboost_trial(E006_XGB_PARAMS, X, y, verbose=False)

    row = tuning_result_row("screen-002", "phase1", None, E006_XGB_PARAMS, result)

    assert row["delta_vs_control"] == ""
    assert row["folds_improved"] == ""
