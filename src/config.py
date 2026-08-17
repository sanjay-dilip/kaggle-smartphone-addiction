"""Centralized project paths and constants.

Kept side-effect free and cheap to import: no file I/O, no logging setup.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
SUBMISSIONS_DIR: Path = PROJECT_ROOT / "submissions"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

TRAIN_PATH: Path = DATA_DIR / "train.csv"
TEST_PATH: Path = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH: Path = DATA_DIR / "sample_submission.csv"

TARGET_COLUMN: str = "addicted_label"
RANDOM_SEED: int = 42
