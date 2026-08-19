"""Tests for src/features.py.

Covers row count, missing-value propagation, non-mutation, train/test
consistency, and absence of infinities — deliberately does not assert
model performance (see docs/DECISIONS.md, Build 4 scope).
"""

import numpy as np
import pandas as pd

from src.features import add_component_sum, add_missing_flag, add_screen_residual


def _sample_frame() -> pd.DataFrame:
    """A small frame covering: all-observed, single-missing, all-missing rows."""
    return pd.DataFrame(
        {
            "daily_screen_time_hours": [10.0, 5.0, np.nan, 8.0, np.nan],
            "social_media_hours": [3.0, 2.0, 1.0, np.nan, np.nan],
            "gaming_hours": [2.0, 1.0, 1.0, 1.0, np.nan],
            "work_study_hours": [1.0, 0.5, 1.0, 1.0, np.nan],
        }
    )


def test_component_sum_expected_values() -> None:
    """component_sum equals the exact sum on fully-observed rows."""
    frame = _sample_frame()

    result = add_component_sum(frame)

    assert result.loc[0, "component_sum"] == 6.0
    assert result.loc[1, "component_sum"] == 3.5


def test_component_sum_missing_propagation() -> None:
    """component_sum is NaN whenever any required component is missing."""
    frame = _sample_frame()

    result = add_component_sum(frame)

    assert pd.isna(result.loc[3, "component_sum"])  # social_media_hours missing
    assert pd.isna(result.loc[4, "component_sum"])  # all components missing


def test_screen_residual_expected_values() -> None:
    """screen_residual equals daily_screen_time_hours - component_sum."""
    frame = _sample_frame()

    result = add_screen_residual(frame)

    assert result.loc[0, "screen_residual"] == 4.0
    assert result.loc[1, "screen_residual"] == 1.5


def test_screen_residual_missing_propagation() -> None:
    """screen_residual is NaN if daily time or any component is missing."""
    frame = _sample_frame()

    result = add_screen_residual(frame)

    assert pd.isna(result.loc[2, "screen_residual"])  # daily_screen_time_hours missing
    assert pd.isna(result.loc[3, "screen_residual"])  # social_media_hours missing


def test_screen_residual_uses_canonical_component_sum() -> None:
    """screen_residual is derived from the same component_sum definition."""
    frame = _sample_frame()

    residual_result = add_screen_residual(frame)
    sum_result = add_component_sum(frame)

    pd.testing.assert_series_equal(
        residual_result["component_sum"], sum_result["component_sum"]
    )


def test_engineered_functions_preserve_row_count() -> None:
    """No feature function may drop or duplicate rows."""
    frame = _sample_frame()

    assert len(add_component_sum(frame)) == len(frame)
    assert len(add_screen_residual(frame)) == len(frame)


def test_engineered_functions_do_not_mutate_input() -> None:
    """Feature functions must return copies, not mutate the input frame."""
    frame = _sample_frame()
    original = frame.copy()

    add_component_sum(frame)
    add_screen_residual(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_engineered_functions_no_infinite_values() -> None:
    """Additive engineered features must never produce infinities."""
    frame = _sample_frame()

    result = add_screen_residual(frame)

    assert not np.isinf(result["component_sum"]).any()
    assert not np.isinf(result["screen_residual"]).any()


def test_engineered_functions_no_target_or_id_reference() -> None:
    """Feature functions must not require or produce target/id columns."""
    frame = _sample_frame()

    result = add_screen_residual(frame)

    assert "addicted_label" not in result.columns
    assert "id" not in result.columns


def test_engineered_functions_identical_definition_train_test() -> None:
    """The same function applied to two frames yields the same column set/order."""
    frame = _sample_frame()
    other_frame = _sample_frame().iloc[:3]

    train_cols = list(add_screen_residual(frame).columns)
    test_cols = list(add_screen_residual(other_frame).columns)

    assert train_cols == test_cols


def test_missing_flag_expected_values() -> None:
    """add_missing_flag marks exactly the null rows as 1."""
    frame = _sample_frame()

    result = add_missing_flag(frame, "daily_screen_time_hours")

    assert result["daily_screen_time_hours_is_missing"].tolist() == [0, 0, 1, 0, 1]
    assert result["daily_screen_time_hours_is_missing"].dtype == np.int8


def test_missing_flag_does_not_mutate_input() -> None:
    """add_missing_flag must return a copy, not mutate the input frame."""
    frame = _sample_frame()
    original = frame.copy()

    add_missing_flag(frame, "daily_screen_time_hours")

    pd.testing.assert_frame_equal(frame, original)


def test_feature_names_are_stable() -> None:
    """Engineered column names must not depend on data content."""
    frame = _sample_frame()

    assert "component_sum" in add_component_sum(frame).columns
    assert "screen_residual" in add_screen_residual(frame).columns
    assert (
        "daily_screen_time_hours_is_missing"
        in add_missing_flag(frame, "daily_screen_time_hours").columns
    )
