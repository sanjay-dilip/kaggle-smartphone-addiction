"""Tests for src/boosting_models.py.

Uses small synthetic data with early stopping already configured with a
generous budget (src.boosting_models.MAX_ITERATIONS), so these run fast
without weakening the fold callbacks used for the real benchmarks.
"""

import numpy as np
import pandas as pd
import pytest

from src.boosting_models import catboost_fold, lightgbm_fold, xgboost_fold
from src.config import CATEGORICAL_COLS, NUMERIC_COLS
from src.preprocessing import build_boosting_frame


def _synthetic_split(n_rows: int = 300, n_test: int = 50):
    rng = np.random.default_rng(seed=0)
    raw = {}
    for col in NUMERIC_COLS:
        values = rng.uniform(0, 10, n_rows + n_test)
        values[::7] = np.nan
        raw[col] = values
    for col in CATEGORICAL_COLS:
        values = rng.choice(["A", "B", "C"], n_rows + n_test).astype(object)
        values[::7] = np.nan
        raw[col] = values
    frame = build_boosting_frame(pd.DataFrame(raw))

    y = pd.Series(
        (frame[NUMERIC_COLS[0]].fillna(0) > frame[NUMERIC_COLS[0]].fillna(0).median())
        .astype(int)
        .values
    )

    X, X_test = frame.iloc[:n_rows], frame.iloc[n_rows:]
    y_train_full = y.iloc[:n_rows]
    X_train, X_val = X.iloc[:240], X.iloc[240:]
    y_train, y_val = y_train_full.iloc[:240], y_train_full.iloc[240:]
    return X_train, y_train, X_val, y_val, X_test


@pytest.mark.parametrize("fold_fn", [catboost_fold, lightgbm_fold, xgboost_fold])
def test_fold_callback_returns_valid_predictions(fold_fn) -> None:
    """Each fold callback must return finite [0, 1] probabilities of the right shape."""
    X_train, y_train, X_val, y_val, X_test = _synthetic_split()

    val_proba, test_proba, best_iteration = fold_fn(X_train, y_train, X_val, y_val, X_test)

    assert val_proba.shape == (len(X_val),)
    assert np.isfinite(val_proba).all()
    assert ((val_proba >= 0) & (val_proba <= 1)).all()

    assert test_proba is not None
    assert test_proba.shape == (len(X_test),)
    assert np.isfinite(test_proba).all()
    assert ((test_proba >= 0) & (test_proba <= 1)).all()

    assert best_iteration is not None
    assert best_iteration >= 0


@pytest.mark.parametrize("fold_fn", [catboost_fold, lightgbm_fold, xgboost_fold])
def test_fold_callback_handles_no_test_set(fold_fn) -> None:
    """test_proba must be None when X_test is None, not an error."""
    X_train, y_train, X_val, y_val, _ = _synthetic_split()

    _, test_proba, _ = fold_fn(X_train, y_train, X_val, y_val, None)

    assert test_proba is None
