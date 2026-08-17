"""Model-specific fold fit/predict callbacks for the Build 3 benchmarks.

Each function matches src.benchmarking.FoldFitPredict and is meant to be
passed to src.benchmarking.run_cv_benchmark. Model construction,
categorical handling, and early stopping are deliberately kept here
rather than hidden behind a shared abstraction, per the Build 3
instruction to separate model-family effects from artificial sameness.

All three use one untuned, sensible benchmark configuration — this is not
a hyperparameter search. Early stopping uses a shared MAX_ITERATIONS /
EARLY_STOPPING_ROUNDS budget so the three models are compared under
similar training effort.
"""

from typing import Optional

import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import CATEGORICAL_COLS, RANDOM_SEED

MAX_ITERATIONS: int = 2000
EARLY_STOPPING_ROUNDS: int = 50
LEARNING_RATE: float = 0.05


def catboost_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    """CatBoost with native categorical handling via `cat_features`."""
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=MAX_ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=CATEGORICAL_COLS,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_proba, test_proba, model.get_best_iteration()


def lightgbm_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    """LightGBM with native categorical handling via `category` dtype."""
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_estimators=MAX_ITERATIONS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="auc",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_proba, test_proba, model.best_iteration_


def xgboost_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: Optional[pd.DataFrame],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    """XGBoost with native categorical handling via `category` dtype."""
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=LEARNING_RATE,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_estimators=MAX_ITERATIONS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1] if X_test is not None else None
    return val_proba, test_proba, model.best_iteration
