"""Smoke checks for local competition data.

Verifies the raw files are present and structurally usable. Does not
compute statistics, distributions, or missingness — that belongs to Build 1.
"""

import pandas as pd

from src.config import (
    SAMPLE_SUBMISSION_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)


def test_paths_resolve_to_existing_files() -> None:
    """Configured competition file paths point to files that exist."""
    assert TRAIN_PATH.is_file()
    assert TEST_PATH.is_file()
    assert SAMPLE_SUBMISSION_PATH.is_file()


def test_train_contains_target_column() -> None:
    """train.csv has the target column."""
    train_columns = pd.read_csv(TRAIN_PATH, nrows=5).columns
    assert TARGET_COLUMN in train_columns


def test_test_does_not_contain_target_column() -> None:
    """test.csv does not leak the target column."""
    test_columns = pd.read_csv(TEST_PATH, nrows=5).columns
    assert TARGET_COLUMN not in test_columns


def test_train_and_test_have_compatible_predictor_columns() -> None:
    """train.csv predictor columns (excluding target) match test.csv columns."""
    train_columns = pd.read_csv(TRAIN_PATH, nrows=5).columns
    test_columns = pd.read_csv(TEST_PATH, nrows=5).columns
    train_predictor_columns = train_columns.drop(TARGET_COLUMN)
    assert list(train_predictor_columns) == list(test_columns)


def test_sample_submission_contains_target_column() -> None:
    """sample_submission.csv has the expected target column."""
    submission_columns = pd.read_csv(SAMPLE_SUBMISSION_PATH, nrows=5).columns
    assert TARGET_COLUMN in submission_columns


def test_sample_submission_row_count_matches_test() -> None:
    """sample_submission.csv has one row per test.csv row."""
    test_row_count = sum(1 for _ in open(TEST_PATH, encoding="utf-8")) - 1
    submission_row_count = (
        sum(1 for _ in open(SAMPLE_SUBMISSION_PATH, encoding="utf-8")) - 1
    )
    assert submission_row_count == test_row_count
