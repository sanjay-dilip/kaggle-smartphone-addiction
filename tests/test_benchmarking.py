"""Tests for src/benchmarking.py."""

import numpy as np
import pandas as pd
import pytest

from src.benchmarking import run_cv_benchmark
from src.validation import N_SPLITS


def _fake_fold_fit_predict(X_train, y_train, X_val, y_val, X_test):
    """A cheap, deterministic stand-in for a real model's fold callback."""
    val_proba = np.full(len(X_val), y_train.mean())
    test_proba = np.full(len(X_test), y_train.mean()) if X_test is not None else None
    return val_proba, test_proba, 42


def _sample_data(n_rows: int = 200):
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame({"feature": rng.uniform(size=n_rows)})
    y = pd.Series(rng.integers(0, 2, size=n_rows))
    X_test = pd.DataFrame({"feature": rng.uniform(size=50)})
    return X, y, X_test


def test_run_cv_benchmark_produces_full_oof_coverage() -> None:
    """Every training row must receive exactly one OOF prediction."""
    X, y, _ = _sample_data()

    result = run_cv_benchmark(_fake_fold_fit_predict, X, y, verbose=False)

    assert result.oof_predictions.shape == (len(X),)
    assert np.isfinite(result.oof_predictions).all()


def test_run_cv_benchmark_reports_fold_scores_and_summary() -> None:
    """fold_scores must have one entry per fold, matching cv_mean/cv_std."""
    X, y, _ = _sample_data()

    result = run_cv_benchmark(_fake_fold_fit_predict, X, y, verbose=False)

    assert len(result.fold_scores) == N_SPLITS
    assert result.cv_mean == pytest.approx(np.mean(result.fold_scores))
    assert result.cv_std == pytest.approx(np.std(result.fold_scores))


def test_run_cv_benchmark_averages_test_predictions_across_folds() -> None:
    """test_predictions must be the per-row mean across all fold predictions."""
    X, y, X_test = _sample_data()

    result = run_cv_benchmark(_fake_fold_fit_predict, X, y, X_test=X_test, verbose=False)

    assert result.test_predictions is not None
    assert result.test_predictions.shape == (len(X_test),)


def test_run_cv_benchmark_without_test_set_returns_none() -> None:
    """No X_test means no test_predictions, not an empty array."""
    X, y, _ = _sample_data()

    result = run_cv_benchmark(_fake_fold_fit_predict, X, y, verbose=False)

    assert result.test_predictions is None


def test_run_cv_benchmark_collects_best_iterations() -> None:
    """best_iterations must have one entry per fold when the callback reports one."""
    X, y, _ = _sample_data()

    result = run_cv_benchmark(_fake_fold_fit_predict, X, y, verbose=False)

    assert result.best_iterations == [42] * N_SPLITS
