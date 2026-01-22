"""Command line interface for the fraud detection toolkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer

from fraud_detection.config import get_settings
from fraud_detection.data.data_val.data_validate import (
    DEFAULT_CLASS_RATIO_BOUNDS,
    DEFAULT_LABEL_COLUMN,
    REQUIRED_COLUMNS,
    load_from_mltable,
    validate_data as validate_dataframe,
)
from fraud_detection.utils.logging import get_logger

app = typer.Typer(help="Utilities to orchestrate Azure ML jobs")
logger = get_logger(__name__)


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@app.command()
def validate_data(
    dataset_name: Annotated[
        str,
        typer.Option(
            "--dataset-name",
            help="Name of the registered dataset to validate.",
        ),
    ],
    validation_report: Annotated[
        str | None,
        typer.Option(
            "--validation-report",
            help="Path to write the JSON validation report.",
        ),
    ] = None,
    is_valid_output: Annotated[
        str | None,
        typer.Option(
            "--is-valid",
            help="Path to write the validation result as 'true' or 'false'.",
        ),
    ] = None,
    sample_rows: Annotated[
        int | None,
        typer.Option(
            "--sample-rows",
            help="Only load the first N rows for a quick validation pass.",
            min=1,
        ),
    ] = None,
    required_columns: Annotated[
        list[str] | None,
        typer.Option(
            "--required-column",
            help="Expected column name. Provide multiple times for many columns.",
        ),
    ] = None,
    label_column: Annotated[
        str,
        typer.Option("--label-column", help="Name of the label column."),
    ] = DEFAULT_LABEL_COLUMN,
    max_missing_ratio: Annotated[
        float,
        typer.Option(
            "--max-missing-ratio",
            help="Maximum allowed missing value ratio per column.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.0,
    min_rows: Annotated[
        int,
        typer.Option("--min-rows", help="Minimum number of rows required.", min=1),
    ] = 1,
    balance_check: Annotated[
        bool,
        typer.Option(
            "--balance-check/--no-balance-check",
            help="Enable class balance check against expected ratio bounds.",
        ),
    ] = False,
    balance_lower: Annotated[
        float,
        typer.Option(
            "--balance-lower",
            help="Lower bound for the class balance ratio (fraudulent / total).",
        ),
    ] = DEFAULT_CLASS_RATIO_BOUNDS[0],
    balance_upper: Annotated[
        float,
        typer.Option(
            "--balance-upper",
            help="Upper bound for the class balance ratio (fraudulent / total).",
        ),
    ] = DEFAULT_CLASS_RATIO_BOUNDS[1],
) -> None:
    """Validate a registered fraud dataset."""
    settings = get_settings()
    dataset_name = dataset_name or settings.registered_dataset_name
    resolved_required = list(REQUIRED_COLUMNS) if required_columns is None else required_columns

    try:
        df = load_from_mltable(dataset_name, sample_rows=sample_rows)
        is_valid, report = validate_dataframe(
            df,
            required_columns=resolved_required,
            label_column=label_column,
            missing_value_ratio_threshold=max_missing_ratio,
            minimum_rows=min_rows,
            check_balance=balance_check,
            class_ratio_bounds=(balance_lower, balance_upper),
        )
    except Exception as exc:
        logger.exception("Data validation failed", extra={"name": dataset_name})
        raise typer.BadParameter(str(exc)) from exc

    if validation_report:
        _write_json(Path(validation_report), report)
    if is_valid_output:
        _write_text(Path(is_valid_output), "true" if is_valid else "false")

    typer.echo("Data validation passed." if is_valid else "Data validation failed.")



def run() -> None:
    app()


if __name__ == "__main__":
    run()
