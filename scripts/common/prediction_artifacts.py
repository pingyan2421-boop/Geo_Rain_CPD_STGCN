from pathlib import Path
from typing import Iterable

import numpy as np


TRAIN_INDICES = "train_indices.npy"
VAL_INDICES = "val_indices.npy"
TEST_INDICES = "test_indices.npy"
TRAIN_TRUE_REAL = "train_true_real.npy"
TRAIN_PRED_REAL = "train_pred_real.npy"
TRAIN_PRED_RAW_REAL = "train_pred_raw_real.npy"
TRAIN_PRED_CAL_REAL = "train_pred_cal_real.npy"
TRAIN_PERSISTENCE_REAL = "train_persistence_real.npy"
VAL_TRUE_REAL = "val_true_real.npy"
VAL_PRED_REAL = "val_pred_real.npy"
VAL_PRED_RAW_REAL = "val_pred_raw_real.npy"
VAL_PRED_CAL_REAL = "val_pred_cal_real.npy"
VAL_PERSISTENCE_REAL = "val_persistence_real.npy"
TEST_TRUE_REAL = "test_true_real.npy"
TEST_PRED_REAL = "test_pred_real.npy"
TEST_PRED_RAW_REAL = "test_pred_raw_real.npy"
TEST_PRED_CAL_REAL = "test_pred_cal_real.npy"
TEST_PERSISTENCE_REAL = "test_persistence_real.npy"
TRAIN_RAIN = "train_rain.npy"
TRAIN_FORECAST_CONTEXT = "train_forecast_context.npy"
TEST_RAIN = "test_rain.npy"
TEST_FORECAST_CONTEXT = "test_forecast_context.npy"
TEST_ALIGNED_RAIN_WINDOWS = "test_aligned_rain_windows.csv"
RAIN_FEATURE_COLS = "rain_feature_cols.txt"
FORECAST_FEATURE_COLS = "forecast_feature_cols.txt"
METADATA = "prediction_artifacts_metadata.csv"

REQUIRED_ARRAYS = (
    TRAIN_INDICES,
    VAL_INDICES,
    TEST_INDICES,
    VAL_TRUE_REAL,
    VAL_PRED_REAL,
    VAL_PRED_RAW_REAL,
    VAL_PRED_CAL_REAL,
    VAL_PERSISTENCE_REAL,
    TEST_TRUE_REAL,
    TEST_PRED_REAL,
    TEST_PRED_RAW_REAL,
    TEST_PRED_CAL_REAL,
    TEST_PERSISTENCE_REAL,
)

OPTIONAL_TRAIN_ARRAYS = (
    TRAIN_TRUE_REAL,
    TRAIN_PRED_REAL,
    TRAIN_PRED_RAW_REAL,
    TRAIN_PRED_CAL_REAL,
    TRAIN_PERSISTENCE_REAL,
)


def prediction_dir(output_dir: str, fold_name: str) -> Path:
    return Path(output_dir) / fold_name / "predictions"


def missing_required_files(predictions_dir: str, filenames: Iterable[str] = REQUIRED_ARRAYS):
    root = Path(predictions_dir)
    return [name for name in filenames if not (root / name).exists()]


def validate_predictions_dir(predictions_dir: str) -> Path:
    root = Path(predictions_dir)
    missing = missing_required_files(str(root))
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing prediction artifact file(s) in {root}: {joined}")
    return root


def load_array(predictions_dir: str, filename: str):
    return np.load(Path(predictions_dir) / filename)
