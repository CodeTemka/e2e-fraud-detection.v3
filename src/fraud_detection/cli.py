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
from fraud_detection.data.data_for_train.data_for_train import (
    fit_scalers,
    read_is_valid_flag,
    save_outputs,
    split_data,
    transform_with_scalers,
    write_validation_failure_metadata,
)
from fraud_detection.pipeline.data_pipeline import submit_data_pipeline_job
from fraud_detection.utils.logging import get_logger
from fraud_detection.azure.client import get_ml_client

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
        str | None,
        typer.Option(
            "--dataset-name",
            help="Name of the registered dataset to validate.",
        ),
    ] = None,
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
    ml_client = get_ml_client()
    dataset_name = dataset_name or settings.registered_dataset_name
    resolved_required = list(REQUIRED_COLUMNS) if required_columns is None else required_columns

    try:
        df = load_from_mltable(dataset_name, sample_rows=sample_rows, mlclient=ml_client)
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
        logger.exception("Data validation failed", extra={"dataset_name": dataset_name})
        raise typer.BadParameter(str(exc)) from exc

    if validation_report:
        _write_json(Path(validation_report), report)
    if is_valid_output:
        _write_text(Path(is_valid_output), "true" if is_valid else "false")

    typer.echo("Data validation passed." if is_valid else "Data validation failed.")


@app.command()
def prep_data_for_train(
    data_name: Annotated[
        str | None,
        typer.Option(
            "--data-name",
            help="Name of the registered dataset to prepare for training.",
        ),
    ] = None,
    is_valid: Annotated[
        str | None,
        typer.Option(
            "--is-valid",
            help="Path to validation flag file ('true'/'false').",
        ),
    ] = None,
    test_ratio: Annotated[
        float,
        typer.Option("--test-ratio", help="Test split ratio.", min=0.0, max=1.0),
    ] = 0.2,
    label_col: Annotated[
        str,
        typer.Option("--label-col", help="Name of the label column."),
    ] = DEFAULT_LABEL_COLUMN,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for splitting.", min=0),
    ] = 42,
    scalers: Annotated[
        str | None,
        typer.Option("--scalers", help="Output folder for scaler artifacts."),
    ] = None,
    metadata: Annotated[
        str | None,
        typer.Option("--metadata", help="Output path for metadata JSON."),
    ] = None,
) -> None:
    """Prepare train/test splits with scaled Amount/Time columns (test remains unscaled)."""
    is_valid_result, invalid_reason = read_is_valid_flag(is_valid)
    if not is_valid_result:
        if not metadata:
            raise typer.BadParameter("Missing output path for: metadata")
        if scalers:
            Path(scalers).mkdir(parents=True, exist_ok=True)
        write_validation_failure_metadata(
            metadata,
            reason=invalid_reason or "Validation flag false; skipping data preparation.",
        )
        typer.echo("Data validation failed; skipping data preparation.")
        return

    output_paths = {
        "scalers": scalers,
        "metadata": metadata,
    }
    missing_outputs = [name for name, path in output_paths.items() if not path]
    if missing_outputs:
        raise typer.BadParameter(f"Missing output paths for: {', '.join(missing_outputs)}")

    scalers = cast(str, scalers)
    metadata = cast(str, metadata)

    settings = get_settings()
    ml_client = get_ml_client()
    dataset_name = data_name or settings.registered_dataset_name

    try:
        df = load_from_mltable(dataset_name, mlclient=ml_client)
        train_df, test_df = split_data(
            df,
            label_col=label_col,
            test=test_ratio,
            seed=seed,
            stratify=True,
        )
        scalers_dict = fit_scalers(train_df)
        train_scaled = transform_with_scalers(train_df, scalers_dict)

        save_outputs(
            train_df=train_scaled,
            test_df=test_df,
            scalers=scalers_dict,
            scalers_output=scalers,
            metadata_output=metadata,
            ml_client=ml_client,
            label_col=label_col,
            seed=seed,
            split_ratio=test_ratio,
        )
    except Exception as exc:
        logger.exception("Data preparation for training failed", extra={"dataset_name": dataset_name})
        raise typer.BadParameter(str(exc)) from exc
    
    typer.echo("Data preparation for training completed successfully.")


@app.command()
def run_data_pipeline(
    data_name: Annotated[
        str | None,
        typer.Option("--data-name", help="Registered dataset name to validate and prepare."),
    ] = None,
    label_col: Annotated[
        str,
        typer.Option("--label-col", help="Name of the label column."),
    ] = DEFAULT_LABEL_COLUMN,
    test_ratio: Annotated[
        float,
        typer.Option("--test-ratio", help="Test split ratio.", min=0.0, max=1.0),
    ] = 0.2,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for splitting.", min=0),
    ] = 42,
    validation_component_version: Annotated[
        str | None,
        typer.Option("--validation-component-version", help="Version for data validation component."),
    ] = None,
    prep_component_version: Annotated[
        str | None,
        typer.Option("--prep-component-version", help="Version for data prep component."),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option("--experiment-name", help="Azure ML experiment name."),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Stream pipeline job logs."),
    ] = False,
) -> None:
    """Run the data validation + preparation pipeline."""
    settings = get_settings()
    ml_client = get_ml_client()
    dataset_name = data_name or settings.registered_dataset_name

    try:
        job = submit_data_pipeline_job(
            data_name=dataset_name,
            label_col=label_col,
            test_ratio=test_ratio,
            seed=seed,
            validation_component_version=validation_component_version,
            prep_component_version=prep_component_version,
            experiment_name=experiment_name,
            ml_client=ml_client,
        )
    except Exception as exc:
        logger.exception("Failed to submit data pipeline", extra={"dataset_name": dataset_name})
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Submitted pipeline job: {job.name}")
    if wait:
        ml_client.jobs.stream(job.name)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
