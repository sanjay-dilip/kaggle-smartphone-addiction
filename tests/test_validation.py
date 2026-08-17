"""Tests for src/validation.py."""

import numpy as np

from src.config import RANDOM_SEED
from src.validation import N_SPLITS, get_cv_splitter


def test_get_cv_splitter_uses_expected_settings() -> None:
    """The splitter must match the frozen Build 1 validation decision."""
    splitter = get_cv_splitter()

    assert splitter.get_n_splits() == N_SPLITS
    assert splitter.shuffle is True
    assert splitter.random_state == RANDOM_SEED


def test_get_cv_splitter_is_deterministic() -> None:
    """Two splitters built the same way must produce identical folds."""
    rng = np.random.default_rng(seed=1)
    X = rng.uniform(size=(100, 1))
    y = rng.integers(0, 2, size=100)

    folds_a = [val_idx.tolist() for _, val_idx in get_cv_splitter().split(X, y)]
    folds_b = [val_idx.tolist() for _, val_idx in get_cv_splitter().split(X, y)]

    assert folds_a == folds_b


def test_get_cv_splitter_preserves_class_balance_per_fold() -> None:
    """Stratification should keep each fold's class ratio close to the whole."""
    rng = np.random.default_rng(seed=2)
    y = np.array([0] * 30 + [1] * 70)
    rng.shuffle(y)
    X = np.zeros((len(y), 1))

    overall_rate = y.mean()
    for _, val_idx in get_cv_splitter().split(X, y):
        fold_rate = y[val_idx].mean()
        assert abs(fold_rate - overall_rate) < 0.05
