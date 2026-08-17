"""Tests for src/submission_validation.py."""

import numpy as np
import pandas as pd
import pytest

from src.config import ID_COLUMN, TARGET_COLUMN
from src.submission_validation import validate_submission


def _sample_submission(n_rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame({ID_COLUMN: range(n_rows), TARGET_COLUMN: [0.0] * n_rows})


def test_valid_submission_passes() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission[TARGET_COLUMN] = 0.5

    validate_submission(submission, sample)


def test_column_mismatch_is_rejected() -> None:
    sample = _sample_submission()
    submission = sample.rename(columns={TARGET_COLUMN: "prediction"})

    with pytest.raises(AssertionError, match="column mismatch"):
        validate_submission(submission, sample)


def test_row_count_mismatch_is_rejected() -> None:
    sample = _sample_submission(n_rows=10)
    submission = _sample_submission(n_rows=9)

    with pytest.raises(AssertionError, match="row count mismatch"):
        validate_submission(submission, sample)


def test_duplicate_ids_are_rejected() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission.loc[1, ID_COLUMN] = submission.loc[0, ID_COLUMN]

    with pytest.raises(AssertionError, match="duplicate ids"):
        validate_submission(submission, sample)


def test_id_order_mismatch_is_rejected() -> None:
    sample = _sample_submission()
    submission = sample.iloc[::-1].reset_index(drop=True)

    with pytest.raises(AssertionError, match="ids"):
        validate_submission(submission, sample)


def test_missing_predictions_are_rejected() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission.loc[0, TARGET_COLUMN] = np.nan

    with pytest.raises(AssertionError, match="missing predictions"):
        validate_submission(submission, sample)


def test_out_of_range_predictions_are_rejected() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission[TARGET_COLUMN] = 1.5

    with pytest.raises(AssertionError, match=r"\[0, 1\]"):
        validate_submission(submission, sample)


def test_non_finite_predictions_are_rejected() -> None:
    sample = _sample_submission()
    submission = sample.copy()
    submission.loc[0, TARGET_COLUMN] = np.inf

    with pytest.raises(AssertionError):
        validate_submission(submission, sample)
