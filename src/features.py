"""Reusable, deterministic feature-engineering functions for Build 4.

Every function takes a DataFrame already containing the raw predictor
columns (e.g. the output of `src.preprocessing.build_boosting_frame`) and
returns a *new* DataFrame with one additional engineered column appended.
Functions never mutate their input, never reference the target column,
apply identically to train and test, and contain no model imports.

Feature definitions trace to Build 1 evidence (`notebooks/01_eda.ipynb`,
Section 11 for the screen-time family, Section 4.2 for missingness) and
are documented further in `docs/DECISIONS.md`.
"""

import pandas as pd

SCREEN_TIME_COMPONENT_COLS: list[str] = [
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
]


def add_component_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `component_sum` = social_media + gaming + work_study hours.

    Build 1 (Section 11) found `daily_screen_time_hours >= component_sum`
    holds in 100% of the 421k fully-observed train rows, consistent with
    `daily_screen_time_hours` being generated as component_sum plus a
    non-negative residual.

    `component_sum` is `NaN` whenever any of the three required columns is
    missing (`skipna=False`): an incomplete sum is not the true component
    total, so missing components are not approximated with a zero fill.
    """
    result = df.copy()
    result["component_sum"] = df[SCREEN_TIME_COMPONENT_COLS].sum(axis=1, skipna=False)
    return result


def add_screen_residual(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `screen_residual` = daily_screen_time_hours - component_sum.

    Built on top of `add_component_sum` so both features share one
    canonical `component_sum` definition. Build 1 (Section 11) found the
    residual is always non-negative among fully-observed rows and is
    noticeably larger for `addicted_label == 1` (mean 1.63h) than
    `addicted_label == 0` (mean 0.64h) — a candidate signal not fully
    captured by the three named components alone.

    `screen_residual` is `NaN` whenever `daily_screen_time_hours` or any
    screen-time component is missing, since an incomplete residual is not
    meaningful.
    """
    result = add_component_sum(df)
    result["screen_residual"] = result["daily_screen_time_hours"] - result["component_sum"]
    return result


def add_missing_flag(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Adds `{column}_is_missing`: 1 if `column` is null in this row, else 0.

    A pure structural flag derived only from `column`'s own missingness —
    no target statistics are used. Applies identically to train and test.
    """
    result = df.copy()
    result[f"{column}_is_missing"] = df[column].isna().astype("int8")
    return result
