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
from fraud_detection.training.automl import SUPPORTED_CLASSIFICATION_METRICS, automl_job_builder, create_automl_job, submit_job
from fraud_detection.training.xgb.submit_xgb import xgb_sweep_job_builder, submit_xgb_sweep_job, resolve_xgb_environment, create_xgb_sweep_job
from fraud_detection.utils.compute import ensure_training_compute
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
    data_label: Annotated[
        str | None,
        typer.Option(
            "--data-label",
            help="Label for the registered dataset asset.",
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
    resolved_data_label = data_label or settings.registered_dataset_label
    resolved_required = list(REQUIRED_COLUMNS) if required_columns is None else required_columns

    try:
        df = load_from_mltable(
            dataset_name,
            sample_rows=sample_rows,
            mlclient=ml_client,
            data_label=resolved_data_label,
        )
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
    data_label: Annotated[
        str | None,
        typer.Option(
            "--data-label",
            help="Label for the registered dataset asset.",
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
    resolved_data_label = data_label or settings.registered_dataset_label

    try:
        df = load_from_mltable(
            dataset_name,
            mlclient=ml_client,
            data_label=resolved_data_label,
        )
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
    ] = 0.1,
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


@app.command()
def train_automl(
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help=f"Please choose from {list(SUPPORTED_CLASSIFICATION_METRICS)}",
            case_sensitive=False,
        ),
    ] = None,
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute cluster name for training (optional override)."),
    ] = None,
    algorithms: Annotated[
        list[str] | None,
        typer.Option(
            "--algorithms",
            "-a",
            help="Restrict training algorithms (comma-separated or repeated). If not provided, AutoML uses its default algorithm set.",
        ),
    ] = None,
):
    """Submit an AutoML classification job for fraud detection."""
    settings = get_settings()
    ml_client = get_ml_client()

    resolved_metric = metric or settings.default_metric_automl_train
    training_data = settings.registered_train

    compute_target = compute or settings.training_compute_cluster_name
    ensure_training_compute(
        ml_client,
        name=compute_target,
        size=settings.training_compute_cluster_type,
        min_instances=0,
        max_instances=settings.training_compute_cluster_node_max_count,
        idle_time_before_scale_down=settings.compute_idle_time_before_scale_down,
    )

    automl_job_config = automl_job_builder(
        metric=resolved_metric,
        training_data=training_data,
        compute=compute_target,
        allowed_algorithms=algorithms,
    )

    job = create_automl_job(config=automl_job_config)
    job_name = submit_job(ml_client=ml_client, job=job)
    typer.echo(f"Submitted AutoML job: {job_name}")


@app.command()
def train_xgb(
    train_data: Annotated[str | None, typer.Option("--train_data", help="Path to the training data file")] = None,
    test_data: Annotated[str | None, typer.Option("--test_data", help="Path to the testing data file")] = None,
    metric: Annotated[str, typer.Option("--metric", help="Evaluation metric to use for training")] = "average_precision_score_macro",
    compute: Annotated[str | None, typer.Option("--compute", help="Compute target for Task")] = None,
    dry_run: Annotated[bool, typer.Option("--dry_run", help="If set, the training will not be executed")] = False,
):
    """Submit an XGBoost hyperparameter sweep job."""
    settings = get_settings()
    ml_client = get_ml_client()

    training_data = train_data or settings.registered_train
    testing_data= test_data or settings.registered_test
    compute_target = compute or settings.training_compute_cluster_name
    ensure_training_compute(
        ml_client,
        name=compute_target,
        size=settings.training_compute_cluster_type,
        min_instances=0,
        max_instances=settings.training_compute_cluster_node_max_count,
        idle_time_before_scale_down=settings.compute_idle_time_before_scale_down,
    )

    try:
        config = xgb_sweep_job_builder(
            training_data=training_data,
            test_data=testing_data,
            metric=metric,
            compute=compute_target,
            settings=settings,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if dry_run:
        environment = resolve_xgb_environment(ml_client, config)
        job = create_xgb_sweep_job(config, environment=environment)
        typer.echo(f"Built XGBoost sweep job: {job.experiment_name}")
        return

    job_name = submit_xgb_sweep_job(ml_client, config)
    typer.echo(f"Submitted XGBoost sweep job: {job_name}")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
