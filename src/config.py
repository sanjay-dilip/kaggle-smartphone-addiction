"""Centralized project paths and constants.

Kept side-effect free and cheap to import: no file I/O, no logging setup.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
DELIVERABLES_DIR: Path = PROJECT_ROOT / "deliverables"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

TRAIN_PATH: Path = DATA_DIR / "train.csv"
TEST_PATH: Path = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH: Path = DATA_DIR / "sample_submission.csv"

TARGET_COLUMN: str = "addicted_label"
ID_COLUMN: str = "id"
RANDOM_SEED: int = 42

NUMERIC_COLS: list[str] = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CATEGORICAL_COLS: list[str] = ["gender", "stress_level", "academic_work_impact"]
