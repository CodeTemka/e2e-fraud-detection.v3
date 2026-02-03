import pandas as pd

from fraud_detection.data.data_val.data_validate import (
    ValidationOptions,
    collect_validation_metrics,
    validate_creditcard_data,
    validate_data,
)


def test_validate_data_missing_required_columns_reports_failure():
    df = pd.DataFrame({"Time": [1, 2], "Amount": [10.0, 12.0], "Class": [0, 1]})
    is_valid, report = validate_data(
        df,
        required_columns=["Time", "Amount", "Class", "V1"],
        minimum_rows=0,
        check_balance=False,
    )

    checks = {entry["check"]: entry for entry in report["checks"]}
    assert is_valid is False and checks["required_columns"]["passed"] is False


def test_validate_data_invalid_labels_marks_failure():
    df = pd.DataFrame(
        {
            "Time": [1, 2, 3],
            "Amount": [5.0, 6.0, 7.0],
            "Class": [0, 2, 1],
        }
    )
    is_valid, report = validate_data(
        df,
        required_columns=["Time", "Amount", "Class"],
        minimum_rows=0,
        check_balance=False,
    )

    checks = {entry["check"]: entry for entry in report["checks"]}
    assert is_valid is False and checks["label_column"]["passed"] is False


def test_validate_creditcard_data_accepts_valid_schema(sample_credit_df):
    options = ValidationOptions(check_balance=True)
    assert validate_creditcard_data(sample_credit_df, options=options) is True


def test_collect_validation_metrics_counts_missing_and_non_numeric():
    df = pd.DataFrame({"Time": [1], "Amount": [2.0], "Class": [0], "V1": ["x"]})
    metrics = collect_validation_metrics(
        df,
        options=ValidationOptions(required_columns=("Time", "Amount", "Class", "V1", "V2")),
    )
    assert metrics["validation.missing_columns_count"] == 1.0 and metrics["validation.non_numeric_columns_count"] == 1.0
