"""Cross-validation harness.

Build 1 found no duplicate/group leakage risk and no train/test
distribution shift (see docs/DECISIONS.md), so plain stratified K-fold is
the frozen starting validation scheme for this competition.
"""

from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED

N_SPLITS: int = 5


def get_cv_splitter(n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Returns the standard stratified K-fold splitter for this competition."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
