"""Preprocessing pipeline for the Logistic Regression baseline.

Decisions here trace to docs/DECISIONS.md, frozen from the Build 1 data
audit: median imputation for numeric predictors, an explicit "Missing"
category for categoricals (not mode imputation), one-hot encoding, and
`id` excluded from the feature matrix.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, NUMERIC_COLS


def build_preprocessor() -> ColumnTransformer:
    """Builds the numeric/categorical preprocessing pipeline.

    Numeric predictors: median imputation, then standardization.
    Categorical predictors: missing values filled with an explicit
    "Missing" category, then one-hot encoding.

    Only NUMERIC_COLS and CATEGORICAL_COLS are referenced, so `id` and the
    target column are excluded by construction.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def build_boosting_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the shared raw feature frame for gradient-boosting benchmarks.

    Numeric predictors are left untouched, including missing values —
    CatBoost, LightGBM, and XGBoost all handle missing numeric values
    natively, so imputation would remove information these models can use
    directly. Categorical predictors get the same explicit "Missing"
    category used for E001 (docs/DECISIONS.md), then a pandas `category`
    dtype so LightGBM and XGBoost can use native categorical splits;
    CatBoost's `cat_features` accepts this dtype directly as well.
    """
    frame = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    for col in CATEGORICAL_COLS:
        frame[col] = frame[col].fillna("Missing").astype("category")
    return frame
