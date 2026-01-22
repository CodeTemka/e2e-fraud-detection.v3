"""Utilities for validating credit card fraud datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import mltable 
import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from azure.ai.ml import MLClient

from fraud_detection.azure.client import get_ml_client

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
)


# Expected imbalance ratio for the Kaggle credit card fraud dataset is ~0.017%
# Allow a reasonable band to accomodate filtered or augmented samples
DEFAULT_CLASS_RATIO_BOUNDS = (0.0005, 0.03)
DEFAULT_LABEL_COLUMN = "Class"
DEFAULT_AMOUNT_COLUMN = "Amount"
DEFAULT_TIME_COLUMN = "Time"
DEFAULT_MISSING_VALUE_RATIO_THRESHOLD = 0.0
DEFAULT_MINIMUM_ROWS = 1000


class DataValidationError(ValueError):
    """Error raised when input data fails validation checks."""


@dataclass(frozen=True)
class ValidationOptions:
    """Options for dataset validation."""

    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    check_balance: bool = True
    class_ratio_bounds: tuple[float, float] = DEFAULT_CLASS_RATIO_BOUNDS
    label_column: str = DEFAULT_LABEL_COLUMN
    missing_value_ratio_threshold: float = DEFAULT_MISSING_VALUE_RATIO_THRESHOLD
    minimum_rows: int = DEFAULT_MINIMUM_ROWS
    amount_column: str = DEFAULT_AMOUNT_COLUMN
    time_column: str = DEFAULT_TIME_COLUMN
    check_amount_non_negative: bool = True
    check_time_non_negative: bool = True



def load_from_mltable(data_name: str, sample_rows: int | None, mlclient: MLClient) -> pd.DataFrame:
    data = mlclient.data.get(name=data_name)
    mltable_data = mltable.load(data.path)
    if sample_rows is not None:
        mltable_data = mltable_data.take(sample_rows)
    return mltable_data.to_pandas_dataframe()


def validate_data(
    df: pd.DataFrame,
    *,
    required_columns: Iterable[str] | None = None,
    label_column: str = DEFAULT_LABEL_COLUMN,
    missing_value_ratio_threshold: float = DEFAULT_MISSING_VALUE_RATIO_THRESHOLD,
    minimum_rows: int = DEFAULT_MINIMUM_ROWS,
    amount_column: str = DEFAULT_AMOUNT_COLUMN,
    time_column: str = DEFAULT_TIME_COLUMN,
    check_balance: bool = False,
    class_ratio_bounds: tuple[float, float] = DEFAULT_CLASS_RATIO_BOUNDS,
    check_amount_non_negative: bool = True,
    check_time_non_negative: bool = True,
) -> tuple[bool, dict]:
    """Validate the dataset and return (is_valid, report)."""

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input data is not a valid pandas DataFrame.")
    if minimum_rows < 0:
        raise ValueError("minimum_rows must be non-negative.")
    if not 0 <= missing_value_ratio_threshold <= 1:
        raise ValueError("missing_value_ratio_threshold must be in 0 and 1.")
    
    row_count = int(len(df))
    column_count = int(df.shape[1])
    checks: list[dict[str, object]] = []
    is_valid = True
    
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if minimum_rows < 0:
        raise ValueError("minimum_rows must be >= 0")
    if not 0 <= missing_value_ratio_threshold <= 1:
        raise ValueError("missing_value_ratio_threshold must be between 0 and 1")

    row_count = int(len(df))
    column_count = int(df.shape[1])
    checks: list[dict[str, object]] = []
    is_valid = True

    def add_check(name: str, passed: bool, message: str) -> None:
        nonlocal is_valid
        passed = bool(passed)
        checks.append({"check": name, "passed": passed, "message": message})
        if not passed:
            is_valid = False

    add_check(
        "minimum_rows",
        row_count >= minimum_rows,
        f"Dataset has {row_count} rows; minimum required is {minimum_rows}.",
    )

    resolved_required = list(required_columns) if required_columns is not None else list(REQUIRED_COLUMNS)
    if resolved_required:
        missing_required = [col for col in resolved_required if col not in df.columns]
        add_check(
            "required_columns",
            not missing_required,
            "All required columns present."
            if not missing_required
            else "Missing required columns: " + ", ".join(sorted(missing_required)),
        )

    if df.columns.size == 0:
        missing_value_stats: dict[str, dict[str, float]] = {}
        add_check("missing_values", False, "Dataset has no columns to check missing values.")
    else:
        missing_counts = df.isna().sum()
        if row_count > 0:
            missing_ratios = missing_counts / row_count
        else:
            missing_ratios = missing_counts.astype(float) * 0.0

        missing_value_stats = {
            col: {
                "missing_count": int(missing_counts[col]),
                "missing_ratio": float(missing_ratios[col]),
            }
            for col in df.columns
        }

        over_threshold = [
            f"{col} ({missing_ratios[col]:.2%})"
            for col in df.columns
            if missing_ratios[col] > missing_value_ratio_threshold
        ]
        add_check(
            "missing_values",
            not over_threshold,
            f"Missing value ratios within threshold ({missing_value_ratio_threshold:.2%})."
            if not over_threshold
            else "Columns exceed missing value ratio threshold: " + ", ".join(over_threshold),
        )

    label_stats: dict[str, object] = {"label_column": label_column}
    raw_labels = None
    numeric_labels = None
    invalid_count = 0
    invalid_values: list[str] = []
    count_0 = count_1 = count_missing = 0

    if label_column in df.columns:
        raw_labels = df[label_column]
        numeric_labels = pd.to_numeric(raw_labels, errors="coerce")
        count_0 = int((numeric_labels == 0).sum())
        count_1 = int((numeric_labels == 1).sum())
        count_missing = int(raw_labels.isna().sum())

        invalid_mask = ~raw_labels.isna() & (~numeric_labels.isin([0, 1]) | numeric_labels.isna())
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            invalid_values = sorted({str(value) for value in raw_labels[invalid_mask].unique()})

        label_stats.update(
            {
                "count_0": count_0,
                "count_1": count_1,
                "count_missing": count_missing,
                "invalid_label_count": invalid_count,
                "positive_ratio": float(count_1 / row_count) if row_count > 0 else None,
            }
        )
        if invalid_values:
            label_stats["invalid_labels"] = invalid_values[:10]
    else:
        label_stats["missing"] = True

    if label_column not in df.columns:
        add_check("label_column", False, f"Missing label column '{label_column}'.")
    elif row_count == 0:
        add_check("label_column", False, f"Label column '{label_column}' is present but dataset is empty.")
    elif invalid_count > 0:
        preview = ", ".join(invalid_values[:5]) if invalid_values else "invalid values"
        add_check(
            "label_column",
            False,
            f"Label column '{label_column}' contains invalid values: {preview}.",
        )
    elif count_0 + count_1 == 0:
        add_check(
            "label_column",
            False,
            f"Label column '{label_column}' has no valid 0/1 values.",
        )
    else:
        add_check("label_column", True, f"Label column '{label_column}' contains only 0/1 values.")

    if check_amount_non_negative:
        if amount_column in df.columns:
            amount_values = pd.to_numeric(df[amount_column], errors="coerce")
            invalid_mask = ~df[amount_column].isna() & amount_values.isna()
            invalid_count = int(invalid_mask.sum())
            negative_count = int((amount_values < 0).sum())
            add_check(
                "amount_non_negative",
                invalid_count == 0 and negative_count == 0,
                "Amount column has no negative or non-numeric values."
                if invalid_count == 0 and negative_count == 0
                else (
                    f"Amount column has {negative_count} negative values and "
                    f"{invalid_count} non-numeric values."
                    if invalid_count and negative_count
                    else (
                        f"Amount column has {invalid_count} non-numeric values."
                        if invalid_count
                        else f"Amount column has {negative_count} negative values."
                    )
                ),
            )
        else:
            add_check(
                "amount_non_negative",
                True,
                f"Amount column '{amount_column}' not present; check skipped.",
            )

    if check_time_non_negative:
        if time_column in df.columns:
            time_values = pd.to_numeric(df[time_column], errors="coerce")
            invalid_mask = ~df[time_column].isna() & time_values.isna()
            invalid_count = int(invalid_mask.sum())
            negative_count = int((time_values < 0).sum())
            add_check(
                "time_non_negative",
                invalid_count == 0 and negative_count == 0,
                "Time column has no negative or non-numeric values."
                if invalid_count == 0 and negative_count == 0
                else (
                    f"Time column has {negative_count} negative values and "
                    f"{invalid_count} non-numeric values."
                    if invalid_count and negative_count
                    else (
                        f"Time column has {invalid_count} non-numeric values."
                        if invalid_count
                        else f"Time column has {negative_count} negative values."
                    )
                ),
            )
        else:
            add_check(
                "time_non_negative",
                True,
                f"Time column '{time_column}' not present; check skipped.",
            )

    if check_balance:
        if label_column not in df.columns:
            add_check(
                "class_balance",
                False,
                f"Missing label column '{label_column}' for class balance check.",
            )
        elif row_count == 0:
            add_check(
                "class_balance",
                False,
                "Dataset is empty; cannot compute class balance.",
            )
        elif invalid_count > 0:
            add_check(
                "class_balance",
                False,
                "Label column has invalid values; cannot compute class balance.",
            )
        else:
            fraud_ratio = float(count_1 / row_count) if row_count > 0 else 0.0
            lower, upper = class_ratio_bounds
            add_check(
                "class_balance",
                lower <= fraud_ratio <= upper,
                "Class balance ratio within bounds."
                if lower <= fraud_ratio <= upper
                else (
                    "Class balance ratio out of bounds: "
                    f"observed {fraud_ratio:.4%}, expected between {lower:.2%} and {upper:.2%}."
                ),
            )

    report = {
        "dataset_shape": {"rows": row_count, "columns": column_count},
        "columns": list(df.columns),
        "missing_value_stats": missing_value_stats,
        "label_stats": label_stats,
        "checks": checks,
        "is_valid": is_valid,
    }
    return is_valid, report



def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError("Missing required column(s): " + ", ".join(sorted(missing)))
    

def _validate_dtypes(df: pd.DataFrame, required: list[str]) -> None:
    non_numeric = [col for col in required if not ptypes.is_numeric_dtype(df[col])]
    if non_numeric:
        raise DataValidationError(
            "Non-numeric dtypes detected for: " + ", ".join(sorted(non_numeric))
        )
    

def _validate_nan_counts(df: pd.DataFrame, required: list[str]) -> None:
    nan_counts = df[required].isna().sum()
    failing = nan_counts[nan_counts > 0]
    if not failing.empty:
        formatted = ", ".join(f"{col} ({int(count)} NaN)" for col, count in failing.items())
        raise DataValidationError("Nan values found in column(s): " + formatted)
    

def _validate_class_balance(df: pd.DataFrame, ratio_bounds: tuple[float, float], label_column: str) -> None:
    if label_column not in df.columns:
        raise DataValidationError(f"Missing required columns: {label_column}")
    
    total = len(df)
    if total == 0:
        raise DataValidationError("Dataset is empty after loading")
    
    class_counts = df[label_column].value_counts(dropna=False)

    invalid_labels = [label for label in class_counts.index if label not in (0, 1)]
    if invalid_labels:
        raise DataValidationError(
            f"Unexpected labels in '{label_column}' column: " + ", ".join(map(str, invalid_labels))
        )
    
    fraud_count = int(class_counts.get(1, 0))
    ratio = fraud_count / total
    lower, upper = ratio_bounds
    if not (lower <= ratio <= upper):
        raise DataValidationError(
            "Class imbalance ratio out of expected range: "
            f"observed {ratio:.4%}, expected between {lower:.2%} and {upper:.2%}"
        )
    

def validate_creditcard_data(df: pd.DataFrame, *, options: ValidationOptions | None = None) -> bool:
    """Validate that the dataframe conforms to expected schema for credit card fraud dataset."""
    if options is None:
        options = ValidationOptions()
    
    _validate_required_columns(df, list(options.required_columns))
    _validate_dtypes(df, list(options.required_columns))
    _validate_nan_counts(df, list(options.required_columns))
    
    if options.check_balance:
        _validate_class_balance(df, options.class_ratio_bounds, options.label_column)

    return True


def collect_validation_metrics(
    df: pd.DataFrame, *, options: ValidationOptions | None = None
) -> dict[str, float]:
    """Collect schema/quality metrics without raising exceptions."""
    opts = options or ValidationOptions()
    required = list(opts.required_columns)

    missing = [col for col in required if col not in df.columns]
    missing_count = float(len(missing))
    total_rows = float(len(df))

    nan_total = float(df.isna().sum().sum())
    non_numeric_count = float(
        sum(1 for col in df.columns if not ptypes.is_numeric_dtype(df[col]))
    )

    class_ratio = np.nan
    label_column = opts.label_column
    if label_column in df.columns and len(df) > 0:
        class_ratio = float(df[label_column].sum() / len(df))

    return {
        "validation.total_rows": total_rows,
        "validation.missing_columns_count": missing_count,
        "validation.non_numeric_columns_count": non_numeric_count,
        "validation.nan_total": nan_total,
        "validation.class_ratio": float(class_ratio),
    }



def _psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    
    quantiles = np.linspace(0.0, 1.0, bins+1)
    edges = np.quantile(reference, quantiles)
    edges[0] = -np.inf
    edges[-1] = -np.inf
    edges = np.unique(edges)
    if edges.size < 2:
        return np.nan
    
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return np.nan
    
    ref_dist = ref_counts / ref_counts.sum()
    cur_dist = cur_counts / cur_counts.sum()

    eps = 1e-6
    ref_dist = np.clip(ref_dist, eps, None)
    cur_dist = np.clip(cur_dist, eps, None)
    ref_dist /= ref_dist.sum()
    cur_dist /= cur_dist.sum()

    return float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.sort(reference)
    current = np.sort(current)
    all_vals = np.concatenate([reference, current])
    ref_cdf = np.searchsorted(reference, all_vals, side="right") / reference.size
    cur_cdf = np.searchsorted(current, all_vals, side='right') / current.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def compute_drift_metrics(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    method: str = "psi",
    bins: int = 10,
) -> dict[str, float]:
    """Compute drift metrics between reference and current datasets."""
    method = method.lower()
    if method not in {"psi", "ks"}:
        raise ValueError(f"Unsupported drift method: {method!r}")
    
    metrics: dict[str, float] = {}
    shared_columns = [
        col 
        for col in reference_df.columns
        if col in current_df.columns
        and ptypes.is_numeric_dtype(reference_df[col])
        and ptypes.is_numeric_dtype(current_df[col])
    ]

    for col in shared_columns:
        ref = pd.to_numeric(reference_df[col], errors="coerce").to_numpy()
        cur = pd.to_numeric(current_df[col], errors="coerce").to_numpy()
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if ref.size == 0 or cur.size == 0:
            metrics[f"drift.{method}.{col}"] = np.nan
            continue

        if method == "psi":
            metrics[f"drift.{method}.{col}"] = _psi(ref, cur, bins)
        else:
            metrics[f"drift.{method}.{col}"] = _ks_statistic(ref, cur)

    return metrics



__all__ = [
    "DataValidationError",
    "ValidationOptions",
    "load_from_mltable",
    "validate_data",
    "validate_creditcard_data",
    "collect_validation_metrics",
    "compute_drift_metrics",
    "DEFAULT_CLASS_RATIO_BOUNDS",
    "DEFAULT_LABEL_COLUMN",
    "DEFAULT_MISSING_VALUE_RATIO_THRESHOLD",
    "DEFAULT_MINIMUM_ROWS",
    "REQUIRED_COLUMNS",
]