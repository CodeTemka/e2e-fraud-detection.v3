"""Utilities for preparing training data splits and scaling columns for fraud detection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mltable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

from fraud_detection.utils.logging import get_logger
from fraud_detection.config import get_settings
from fraud_detection.data.data_val.data_validate import (
    DEFAULT_AMOUNT_COLUMN,
    DEFAULT_TIME_COLUMN,
    DEFAULT_LABEL_COLUMN,
)

logger = get_logger(__name__)

_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n"}


def read_is_valid_flag(path: str | Path | None) -> tuple[bool, str | None]:
    if not path:
        return True, None

    flag_path = Path(path)
    try:
        content = flag_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False, "is_valid file not found"
    except OSError as exc:
        return False, f"failed to read is_valid file: {exc}"

    normalized = content.lower()
    if normalized in _TRUE_VALUES:
        return True, None
    if normalized in _FALSE_VALUES:
        return False, "validation flag false"
    return False, f"unexpected is_valid value: {content!r}"


def write_validation_failure_metadata(metadata_output: str | Path, *, reason: str) -> dict[str, object]:
    metadata_path = Path(metadata_output)
    payload = {
        "validation_passed": False,
        "status": "skipped",
        "reason": reason,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def split_data(
    df: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COLUMN,
    test: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a valid pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not 0 <= test < 1:
        raise ValueError("Test size must be between 0 and 1.")
    if stratify and label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    
    stratify_values = df[label_col] if stratify else None

    if test > 0:
        train_df, test_df = train_test_split(
            df,
            test_size=test,
            random_state=seed,
            stratify=stratify_values,
        )
    else:
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)
    
    return train_df, test_df


def fit_scalers(train_df: pd.DataFrame) -> dict[str, RobustScaler]:
    """Fit RobustScaler instances for Amount and Time columns."""
    scalers: dict[str, RobustScaler] = {}
    for column in (DEFAULT_AMOUNT_COLUMN, DEFAULT_TIME_COLUMN):
        if column in train_df.columns:
            scaler = RobustScaler()
            scaler.fit(train_df[[column]])
            scalers[column] = scaler
        else:
            logger.warning(f"Column '{column}' not found in training DataFrame. Skipping scaler fit.")
    return scalers


def transform_with_scalers(df: pd.DataFrame, scalers: dict[str, RobustScaler]) -> pd.DataFrame:
    """Transform Amount and Time columns using the provided scalers."""
    transformed_df = df.copy()
    for column, scaler in scalers.items():
        if column in transformed_df.columns:
            transformed_df[column] = scaler.transform(transformed_df[[column]]).reshape(-1)
        else:
            logger.warning(f"Column '{column}' not found in DataFrame. Skipping transformation.")
    return transformed_df


def save_outputs(
    ml_client: MLClient, 
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scalers: dict[str, RobustScaler],
    scalers_output: str | Path,
    metadata_output: str | Path,
    label_col: str = DEFAULT_LABEL_COLUMN,
    seed: int = 42,
    split_ratio: float = 0.2,
) -> dict:
    """Register train/test data into Azure ML and save scalers and metadata to the provided output locations."""
    settings = get_settings()
    scalers_dir = Path(scalers_output)
    metadata_path = Path(metadata_output)

    scalers_dir.mkdir(parents=True, exist_ok=True)
    scaler_files: dict[str, str] = {}
    for column, scaler in scalers.items():
        filename = f"{column.lower()}_scaler.joblib"
        joblib.dump(scaler, scalers_dir / filename)
        scaler_files[column] = filename

    metadata = {
        "split_sizes": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
        },
        "columns": list(train_df.columns),
        "label_column": label_col,
        "random_seed": seed,
        "scalers": scaler_files,
        "split_ratio": split_ratio,
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path = tmp_path / "train_mltable"
        test_path = tmp_path / "test_mltable"

        train_mltable = mltable.from_pandas(train_df)
        train_mltable.save(train_path)
        train_asset = Data(
            name=settings.registered_train,
            version="1",
            type=AssetTypes.MLTABLE,
            path=str(train_path),
            description="Training data for fraud detection model",
        )
        train_asset = ml_client.data.create_or_update(train_asset)
        logger.info(
            "Registered training data",
            extra={"data": f"{train_asset.name}:{train_asset.version}"},
        )

        test_mltable = mltable.from_pandas(test_df)
        test_mltable.save(test_path)
        test_asset = Data(
            name=settings.registered_test,
            version="1",
            type=AssetTypes.MLTABLE,
            path=str(test_path),
            description="Testing data for fraud detection model",
        )
        test_asset = ml_client.data.create_or_update(test_asset)
        logger.info(
            "Registered testing data",
            extra={"data": f"{test_asset.name}:{test_asset.version}"},
        )

    return metadata


__all__ = [
    "read_is_valid_flag",
    "split_data",
    "fit_scalers",
    "transform_with_scalers",
    "save_outputs",
    "write_validation_failure_metadata",
]
