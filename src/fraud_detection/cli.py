"""Command line interface for the fraud detection toolkit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, cast

import typer

from fraud_detection.config import ROOT_DIR, get_settings
from fraud_detection.utils.logging import get_logger
from fraud_detection.azure.client import get_ml_client


app = typer.Typer(help="Utilities to orchestrate Azure ML jobs")
logger = get_logger(__name__)

DEFAULT_LABEL_COLUMN = "Class"
DEFAULT_CLASS_RATIO_BOUNDS = (0.0005, 0.03)
REQUIRED_COLUMNS = (
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
)

DEFAULT_PROVIDER_NAMES = (
    "Microsoft.MachineLearningServices",
    "Microsoft.ContainerRegistry",
    "Microsoft.Storage",
    "Microsoft.KeyVault",
    "Microsoft.ManagedIdentity",
    "Microsoft.Insights",
    "Microsoft.Network",
    "Microsoft.PolicyInsights",
    "Microsoft.Cdn",
)


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n"}


def _read_bool_file(path: str | None) -> bool | None:
    if not path:
        return None
    content = Path(path).read_text(encoding="utf-8").strip().lower()
    if content in _TRUE_VALUES:
        return True
    if content in _FALSE_VALUES:
        return False
    raise typer.BadParameter(f"Unexpected boolean value in {path}: {content!r}")


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


def _run_az(command: list[str]) -> dict[str, object]:
    az_path = os.environ.get("AZ_PATH") or shutil.which("az")
    if not az_path:
        raise typer.BadParameter(
            "Azure CLI 'az' not found. Install Azure CLI or set AZ_PATH to the az executable."
        )

    command = [az_path, *command[1:]]
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise typer.BadParameter(f"Azure CLI failed: {message or 'unknown error'}")

    output = (result.stdout or "").strip()
    if not output:
        return {}
    try:
        return cast(dict[str, object], json.loads(output))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter("Azure CLI returned non-JSON output.") from exc


def _get_provider_state(namespace: str, subscription_id: str) -> str | None:
    payload = _run_az(
        [
            "az",
            "provider",
            "show",
            "--namespace",
            namespace,
            "--subscription",
            subscription_id,
            "--query",
            "{namespace:namespace, registrationState:registrationState}",
            "--output",
            "json",
        ]
    )
    state = payload.get("registrationState") if isinstance(payload, dict) else None
    return str(state) if state is not None else None



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
    from fraud_detection.data.data_val.data_validate import (
        load_from_mltable,
        validate_data as validate_dataframe,
    )

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
    ] = 0.1,
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
    from fraud_detection.data.data_for_train.data_for_train import (
        fit_scalers,
        read_is_valid_flag,
        save_outputs,
        split_data,
        transform_with_scalers,
        write_validation_failure_metadata,
    )

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
    from fraud_detection.pipeline.data_pipeline import submit_data_pipeline_job

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

    typer.echo(f"Submitted pipeline job: {getattr(job, 'name', None)}")
    if wait:
        ml_client.jobs.stream(getattr(job, 'name'))



@app.command()
def train_automl(
    metric: Annotated[
        str | None,
        typer.Option(
            "--metric",
            "-m",
            help="Metric to optimize (see AutoML supported classification metrics).",
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
    from fraud_detection.training.automl import automl_job_builder, create_automl_job, submit_job
    from fraud_detection.utils.compute import ensure_training_compute

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
    train_data: Annotated[str | None, typer.Option("--train_data", help="Registered name of training data asset")] = None,
    metric: Annotated[
        str, typer.Option("--metric", help="Evaluation metric to use for sweep (average_precision_score_macro or AUC_macro)")
    ] = "average_precision_score_macro",
    val_size: Annotated[float, typer.Option("--val_size", help="Fraction of data used for validation split")] = 0.2,
    compute: Annotated[str | None, typer.Option("--compute", help="Compute target for task")] = None,
    run_suffix: Annotated[
        str | None,
        typer.Option("--run-suffix", help="Optional suffix to make the job name unique."),
    ] = None,
    force_new_run: Annotated[
        bool,
        typer.Option("--force-new-run/--no-force-new-run", help="Force a new sweep run even if a job exists."),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry_run", help="If set, the training will not be executed")] = False,
):
    """Submit an XGBoost hyperparameter sweep job (no test_data)."""
    from fraud_detection.training.xgb.submit_xgb import (
        create_xgb_sweep_job,
        resolve_xgb_environment,
        submit_xgb_sweep_job,
        xgb_sweep_job_builder,
    )
    from fraud_detection.utils.compute import ensure_training_compute

    settings = get_settings()
    ml_client = get_ml_client()

    training_data = train_data or settings.registered_train
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
            metric=metric,
            val_size=val_size,
            compute=compute_target,
            run_suffix=run_suffix,
            force_new_run=force_new_run,
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



@app.command()
def train_lgbm(
    train_data: Annotated[str | None, typer.Option("--train_data", help="Registered name of training data asset")] = None,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Sweep metric (average_precision_score_macro or AUC_macro)"),
    ] = "average_precision_score_macro",
    val_size: Annotated[float, typer.Option("--val_size", help="Fraction of data used for validation split")] = 0.2,
    compute: Annotated[str | None, typer.Option("--compute", help="Compute target for task")] = None,
    run_suffix: Annotated[
        str | None,
        typer.Option("--run-suffix", help="Optional suffix to make the job name unique."),
    ] = None,
    force_new_run: Annotated[
        bool,
        typer.Option("--force-new-run/--no-force-new-run", help="Force a new sweep run even if a job exists."),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry_run", help="If set, the training will not be executed")] = False,
) -> None:
    """Submit a LightGBM hyperparameter sweep job (no test_data)."""
    from fraud_detection.training.lgbm.submit_lgbm import (
        create_lgbm_sweep_job,
        lgbm_sweep_job_builder,
        resolve_lgbm_environment,
        submit_lgbm_sweep_job,
    )
    from fraud_detection.utils.compute import ensure_training_compute

    settings = get_settings()
    ml_client = get_ml_client()

    training_data = train_data or settings.registered_train
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
        config = lgbm_sweep_job_builder(
            training_data=training_data,
            metric=metric,
            val_size=val_size,
            compute=compute_target,
            run_suffix=run_suffix,
            force_new_run=force_new_run,
            settings=settings,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if dry_run:
        environment = resolve_lgbm_environment(ml_client, config)
        job = create_lgbm_sweep_job(config, environment=environment)
        typer.echo(f"Built LightGBM sweep job: {job.experiment_name}")
        return

    job_name = submit_lgbm_sweep_job(ml_client, config)
    typer.echo(f"Submitted LightGBM sweep job: {job_name}")



@app.command("promote-prod-model")
def promote_prod_model(
    compare_metric: Annotated[
        str | None,
        typer.Option(
            "--compare-metric",
            help="Metric to compare when selecting the production model.",
        ),
    ] = None,
    experiments: Annotated[
        str | None,
        typer.Option(
            "--experiments",
            help="Comma-separated experiment names to search for best runs.",
        ),
    ] = None,
    dry_run: Annotated[
        str | None,
        typer.Option(
            "--dry-run",
            help="Set to true to skip model registration (decision only).",
        ),
    ] = None,
    production_info: Annotated[
        str | None,
        typer.Option(
            "--production-info",
            help="Output path for promotion metadata JSON.",
        ),
    ] = None,
    new_promotion: Annotated[
        str | None,
        typer.Option(
            "--new-promotion",
            help="Output path for promotion flag ('true'/'false').",
        ),
    ] = None,
) -> None:
    """Compare experiment runs and promote the best model to production."""
    from fraud_detection.registry.prod_model import register_prod_model

    settings = get_settings()
    ml_client = get_ml_client()

    resolved_metric = compare_metric or settings.default_metric_serving
    dry_run_flag = _parse_bool(dry_run)

    try:
        result = register_prod_model(
            resolved_metric,
            experiments=experiments,
            dry_run=dry_run_flag,
            ml_client=ml_client,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("Production model promotion failed", extra={"metric": resolved_metric})
        raise typer.BadParameter(str(exc)) from exc

    if production_info:
        _write_json(Path(production_info), result.to_dict())
    if new_promotion:
        _write_text(Path(new_promotion), "true" if result.decision == "promote" else "false")

    typer.echo(f"Promotion decision: {result.decision}")



@app.command("run-deployment-pipeline")
def run_deployment_pipeline(
    compare_metric: Annotated[
        str | None,
        typer.Option(
            "--compare-metric",
            help="Metric to compare when selecting the production model.",
        ),
    ] = None,
    experiments: Annotated[
        str | None,
        typer.Option(
            "--experiments",
            help="Comma-separated experiment names to search for best runs.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="If set, evaluate promotion without registering a model.",
        ),
    ] = False,
    force_deploy: Annotated[
        bool,
        typer.Option(
            "--force-deploy/--no-force-deploy",
            help="Deploy even when new_promotion=false (useful for scoring env updates).",
        ),
    ] = False,
    skip_env: Annotated[
        bool,
        typer.Option(
            "--skip-env/--no-skip-env",
            help="Reuse existing scoring environment instead of registering a new one.",
        ),
    ] = False,
    component_version: Annotated[
        str | None,
        typer.Option(
            "--component-version",
            help="Specific component version to use (optional).",
        ),
    ] = None,
    serve_component_version: Annotated[
        str | None,
        typer.Option(
            "--serve-component-version",
            help="Specific serving component version to use (optional).",
        ),
    ] = None,
    scaler_asset_name: Annotated[
        str | None,
        typer.Option("--scaler-asset-name", help="Scaler asset name override."),
    ] = None,
    prod_model_name: Annotated[
        str | None,
        typer.Option("--prod-model-name", help="Registered production model name override."),
    ] = None,
    endpoint_name: Annotated[
        str | None,
        typer.Option("--endpoint-name", help="Endpoint name override."),
    ] = None,
    deployment_name: Annotated[
        str | None,
        typer.Option("--deployment-name", help="Deployment name override."),
    ] = None,
    max_alerts: Annotated[
        int | None,
        typer.Option("--max-alerts", help="Maximum number of alerts to return."),
    ] = None,
    traffic_percentage: Annotated[
        int | None,
        typer.Option("--traffic-percentage", help="Traffic percentage for the new deployment."),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option(
            "--experiment-name",
            help="Azure ML experiment name for the pipeline job.",
        ),
    ] = None,
    wait: Annotated[
        bool,
        typer.Option(
            "--wait/--no-wait",
            help="Stream pipeline job logs.",
        ),
    ] = False,
) -> None:
    """Submit the deployment pipeline job to promote and (optionally) serve the production model."""
    from fraud_detection.pipeline.deployment_pipeline import submit_deployment_pipeline_job

    settings = get_settings()
    ml_client = get_ml_client()

    try:
        job = submit_deployment_pipeline_job(
            compare_metric=compare_metric or settings.default_metric_serving,
            experiments=experiments,
            dry_run=dry_run,
            force_deploy=force_deploy,
            skip_env=skip_env,
            component_version=component_version,
            serve_component_version=serve_component_version,
            experiment_name=experiment_name,
            scaler_asset_name=scaler_asset_name,
            prod_model_name=prod_model_name,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            max_alerts=max_alerts,
            traffic_percentage=traffic_percentage,
            ml_client=ml_client,
        )
    except Exception as exc:
        logger.exception("Failed to submit deployment pipeline", extra={"metric": compare_metric})
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Submitted deployment pipeline job: {getattr(job, 'name', None)}")
    if wait:
        ml_client.jobs.stream(getattr(job, "name"))


@app.command("check-providers")
def check_providers(
    subscription_id: Annotated[
        str | None,
        typer.Option("--subscription-id", help="Azure subscription id to check."),
    ] = None,
    providers: Annotated[
        list[str] | None,
        typer.Option("--provider", help="Provider namespace. Provide multiple times for many providers."),
    ] = None,
) -> None:
    """Check Azure resource provider registration status for the subscription."""
    settings = get_settings()
    sub_id = (subscription_id or settings.subscription_id).strip()
    provider_list = providers or list(DEFAULT_PROVIDER_NAMES)

    if not sub_id:
        raise typer.BadParameter("subscription-id is required.")

    typer.echo(f"Subscription: {sub_id}")
    missing: list[str] = []

    for provider in provider_list:
        name = provider.strip()
        if not name:
            continue
        state = _get_provider_state(name, sub_id) or "Unknown"
        typer.echo(f"{name}: {state}")
        if state.lower() != "registered":
            missing.append(name)

    if missing:
        typer.echo("Providers not registered:")
        for name in missing:
            typer.echo(f"- {name}")
        typer.echo("Register them with:")
        for name in missing:
            typer.echo(f"az provider register --namespace {name} --subscription {sub_id}")
        raise typer.Exit(code=1)

    typer.echo("All providers are registered.")


@app.command("create-endpoint")
def create_endpoint(
    endpoint_name: Annotated[
        str | None,
        typer.Option("--endpoint-name", help="Endpoint name override."),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--description", help="Optional endpoint description."),
    ] = None,
    auth_mode: Annotated[
        str,
        typer.Option("--auth-mode", help="Authentication mode (key or aml_token)."),
    ] = "key",
) -> None:
    """Create (or update) an Azure ML online endpoint."""
    from fraud_detection.serving.managed_endpoint.endpoint_ops import create_endpoint as create_ml_endpoint

    settings = get_settings()
    ml_client = get_ml_client()
    resolved_name = (endpoint_name or settings.endpoint_name).strip()
    if not resolved_name:
        raise typer.BadParameter("endpoint-name is required.")

    create_ml_endpoint(
        ml_client,
        name=resolved_name,
        description=description,
        auth_mode=auth_mode,
        settings=settings,
    )
    typer.echo(f"Created/updated endpoint: {resolved_name}")


@app.command("delete-endpoint")
def delete_endpoint(
    endpoint_name: Annotated[
        str | None,
        typer.Option("--endpoint-name", help="Endpoint name override."),
    ] = None,
) -> None:
    """Delete an Azure ML online endpoint."""
    from fraud_detection.serving.managed_endpoint.endpoint_ops import delete_endpoint as delete_ml_endpoint

    settings = get_settings()
    ml_client = get_ml_client()
    resolved_name = (endpoint_name or settings.endpoint_name).strip()
    if not resolved_name:
        raise typer.BadParameter("endpoint-name is required.")

    delete_ml_endpoint(ml_client, name=resolved_name)
    typer.echo(f"Deleted endpoint: {resolved_name}")


@app.command("serve-prod-model")
def serve_prod_model(
    new_promotion: Annotated[
        str | None,
        typer.Option("--new-promotion", help="Path to new_promotion flag file from promote-prod-model."),
    ] = None,
    prod_model_name: Annotated[
        str | None,
        typer.Option("--prod-model-name", help="Registered production model name override."),
    ] = None,
    model_version: Annotated[
        str | None,
        typer.Option("--model-version", help="Specific model version to deploy (optional)."),
    ] = None,
    endpoint_name: Annotated[
        str | None,
        typer.Option("--endpoint-name", help="Endpoint name override."),
    ] = None,
    deployment_name: Annotated[
        str | None,
        typer.Option("--deployment-name", help="Deployment name override."),
    ] = None,
    scaler_dir: Annotated[
        str | None,
        typer.Option("--scaler-dir", help="Scaler directory (URI_FOLDER from data prep)."),
    ] = None,
    max_alerts: Annotated[
        int,
        typer.Option("--max-alerts", help="Maximum number of alerts to return.", min=0),
    ] = 100,
    traffic_percentage: Annotated[
        int | None,
        typer.Option("--traffic-percentage", help="Traffic percentage for the new deployment."),
    ] = 100,
    dry_run: Annotated[
        str | None,
        typer.Option("--dry-run", help="If true, skip deployment and only log intent."),
    ] = None,
    skip_env: Annotated[
        str | None,
        typer.Option(
            "--skip-env",
            help="If true, reuse existing scoring environment instead of registering a new one.",
        ),
    ] = None,
    force_deploy: Annotated[
        str | None,
        typer.Option(
            "--force-deploy",
            help="If true, deploy even when new_promotion=false (useful for scoring env updates).",
        ),
    ] = None,
    success_flag: Annotated[
        str | None,
        typer.Option("--success-flag", help="Path to write success flag (true/false)."),
    ] = None,
    endpoint_name_out: Annotated[
        str | None,
        typer.Option("--endpoint-name-out", help="Path to write resolved endpoint name."),
    ] = None,
    deployment_name_out: Annotated[
        str | None,
        typer.Option("--deployment-name-out", help="Path to write resolved deployment name."),
    ] = None,
    deployed_model_name: Annotated[
        str | None,
        typer.Option("--deployed-model-name", help="Path to write deployed model name."),
    ] = None,
    deployed_model_version: Annotated[
        str | None,
        typer.Option("--deployed-model-version", help="Path to write deployed model version."),
    ] = None,
    scoring_info: Annotated[
        str | None,
        typer.Option("--scoring-info", help="Path to write deployment metadata JSON."),
    ] = None,
    target: Annotated[
        str,
        typer.Option("--target", help="Deployment target (aml-endpoint or container-apps)."),
    ] = "aml-endpoint",
    container_app_name: Annotated[
        str | None,
        typer.Option("--container-app-name", help="Container Apps name override."),
    ] = None,
    container_app_environment: Annotated[
        str | None,
        typer.Option("--container-app-environment", help="Container Apps environment name override."),
    ] = None,
    container_registry_name: Annotated[
        str | None,
        typer.Option("--container-registry", help="ACR name for building the image."),
    ] = None,
    container_image: Annotated[
        str | None,
        typer.Option("--container-image", help="Full image reference to deploy (skips build)."),
    ] = None,
    container_image_name: Annotated[
        str | None,
        typer.Option("--container-image-name", help="Image repository name when building."),
    ] = None,
    container_image_tag: Annotated[
        str | None,
        typer.Option("--container-image-tag", help="Image tag when building."),
    ] = None,
    container_dockerfile: Annotated[
        str | None,
        typer.Option("--container-dockerfile", help="Dockerfile path for image build."),
    ] = None,
    container_source: Annotated[
        str | None,
        typer.Option("--container-source", help="Build context path for image build."),
    ] = None,
    use_managed_identity: Annotated[
        bool,
        typer.Option(
            "--use-managed-identity/--use-service-principal",
            help="Use managed identity for Container Apps (default).",
        ),
    ] = True,
    scaler_asset_name: Annotated[
        str | None,
        typer.Option("--scaler-asset-name", help="Scaler data asset name (Container Apps)."),
    ] = None,
    scaler_asset_version: Annotated[
        str | None,
        typer.Option("--scaler-asset-version", help="Scaler data asset version (optional)."),
    ] = None,
) -> None:
    """Deploy the production model to an online endpoint (component entrypoint)."""
    from fraud_detection.serving.managed_endpoint.serve_prod_model import (
        resolve_model as resolve_serving_model,
        serve_prod_model as deploy_prod_model,
    )
    from fraud_detection.serving.container_registry.container_apps import deploy_container_app

    settings = get_settings()
    ml_client = get_ml_client()

    should_deploy = _read_bool_file(new_promotion)
    force_deploy_flag = _parse_bool(force_deploy)
    if should_deploy is False and not force_deploy_flag:
        resolved_endpoint = (endpoint_name or settings.endpoint_name).strip()
        resolved_deployment = (deployment_name or settings.deployment_name).strip()
        resolved_model_name = (prod_model_name or settings.prod_model_name).strip()

        model_version_value = ""
        model_name_value = resolved_model_name
        try:
            resolved_model = resolve_serving_model(ml_client, model_name=resolved_model_name)
            model_version_value = resolved_model.version
            model_name_value = resolved_model.name
        except Exception as exc:
            logger.warning("Skipping deployment; failed to resolve model", extra={"error": str(exc)})

        info = {
            "action": "skipped",
            "reason": "new_promotion=false",
            "endpoint_name": resolved_endpoint,
            "deployment_name": resolved_deployment,
            "model_name": model_name_value,
            "model_version": model_version_value,
        }

        if success_flag:
            _write_text(Path(success_flag), "true")
        if endpoint_name_out:
            _write_text(Path(endpoint_name_out), resolved_endpoint)
        if deployment_name_out:
            _write_text(Path(deployment_name_out), resolved_deployment)
        if deployed_model_name:
            _write_text(Path(deployed_model_name), model_name_value)
        if deployed_model_version:
            _write_text(Path(deployed_model_version), model_version_value)
        if scoring_info:
            _write_json(Path(scoring_info), info)
        typer.echo("Deployment skipped (new_promotion=false).")
        return
    if should_deploy is False and force_deploy_flag:
        logger.info(
            "Forcing deployment despite new_promotion=false",
            extra={"endpoint_name": endpoint_name, "deployment_name": deployment_name},
        )

    resolved_target = (target or "").strip().lower()
    if resolved_target in {"aml", "aml-endpoint", "online-endpoint"}:
        if not scaler_dir:
            raise typer.BadParameter("--scaler-dir is required for AML endpoint deployment.")

        result = deploy_prod_model(
            ml_client,
            prod_model_name=prod_model_name,
            model_version=model_version,
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            scaler_dir=scaler_dir,
            max_alerts=max_alerts,
            traffic_percentage=traffic_percentage,
            dry_run=_parse_bool(dry_run),
            skip_env=_parse_bool(skip_env),
            settings=settings,
        )

        if success_flag:
            _write_text(Path(success_flag), "true")
        if endpoint_name_out:
            _write_text(Path(endpoint_name_out), result.endpoint_name)
        if deployment_name_out:
            _write_text(Path(deployment_name_out), result.deployment_name)
        if deployed_model_name:
            _write_text(Path(deployed_model_name), result.model_name)
        if deployed_model_version:
            _write_text(Path(deployed_model_version), result.model_version)
        if scoring_info:
            _write_json(Path(scoring_info), result.to_dict())
        return

    if resolved_target not in {"container-apps", "containerapp", "aca"}:
        raise typer.BadParameter(
            "Unsupported target. Choose 'aml-endpoint' or 'container-apps'."
        )

    resolved_app_name = (container_app_name or settings.container_app_name).strip()
    resolved_env_name = (container_app_environment or settings.container_app_environment).strip()
    if not resolved_app_name:
        raise typer.BadParameter("container-app-name is required for Container Apps deployment.")
    if not resolved_env_name:
        raise typer.BadParameter("container-app-environment is required for Container Apps deployment.")

    resolved_registry = (container_registry_name or settings.container_registry_name or "").strip() or None
    resolved_image_name = (container_image_name or settings.container_app_image_name or "").strip() or None
    resolved_dockerfile = container_dockerfile or str(ROOT_DIR / "deploy" / "container_apps" / "Dockerfile")
    resolved_source = container_source or str(ROOT_DIR)

    resolved_scaler_asset = (scaler_asset_name or settings.serving_scalers_name).strip()
    if not resolved_scaler_asset:
        raise typer.BadParameter("scaler-asset-name is required for Container Apps deployment.")

    container_result = deploy_container_app(
        ml_client,
        app_name=resolved_app_name,
        environment_name=resolved_env_name,
        model_name=prod_model_name or settings.prod_model_name,
        model_version=model_version,
        scaler_asset_name=resolved_scaler_asset,
        scaler_asset_version=scaler_asset_version,
        registry_name=resolved_registry,
        image_name=resolved_image_name,
        image_tag=container_image_tag,
        image=container_image,
        dockerfile=resolved_dockerfile,
        source_dir=resolved_source,
        target_port=settings.container_app_port,
        max_alerts=max_alerts,
        dry_run=_parse_bool(dry_run),
        include_sp_credentials=not use_managed_identity,
        settings=settings,
    )

    if success_flag:
        _write_text(Path(success_flag), "true")
    if endpoint_name_out:
        _write_text(Path(endpoint_name_out), container_result.app_name)
    if deployment_name_out:
        _write_text(Path(deployment_name_out), container_result.revision or container_result.app_name)
    if deployed_model_name:
        _write_text(Path(deployed_model_name), container_result.model_name)
    if deployed_model_version:
        _write_text(Path(deployed_model_version), container_result.model_version)
    if scoring_info:
        _write_json(Path(scoring_info), container_result.to_dict())


@app.command("evaluate-endpoint")
def evaluate_endpoint_cmd(
    test_data: Annotated[
        str | None,
        typer.Option("--test-data", help="Registered test data asset name or local path."),
    ] = None,
    data_label: Annotated[
        str | None,
        typer.Option("--data-label", help="Optional data asset label for test data."),
    ] = None,
    endpoint_name: Annotated[
        str | None,
        typer.Option("--endpoint-name", help="Endpoint name override."),
    ] = None,
    deployment_name: Annotated[
        str | None,
        typer.Option("--deployment-name", help="Deployment name override."),
    ] = None,
    scoring_uri: Annotated[
        str | None,
        typer.Option("--scoring-uri", help="Override scoring URI (skips endpoint lookup)."),
    ] = None,
    endpoint_key: Annotated[
        str | None,
        typer.Option("--endpoint-key", help="Endpoint key (falls back to env vars)."),
    ] = None,
    auth_mode: Annotated[
        str,
        typer.Option("--auth-mode", help="Auth mode for endpoint calls (key or aad)."),
    ] = "key",
    label_column: Annotated[
        str,
        typer.Option("--label-column", help="Label column name."),
    ] = DEFAULT_LABEL_COLUMN,
    out: Annotated[
        str | None,
        typer.Option("--out", help="Output directory for evaluation artifacts."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for endpoint calls.", min=1),
    ] = 200,
    max_retries: Annotated[
        int,
        typer.Option("--max-retries", help="Max retries for endpoint calls.", min=0),
    ] = 3,
    request_timeout: Annotated[
        float,
        typer.Option("--request-timeout", help="Request timeout (seconds).", min=1.0),
    ] = 30.0,
    max_alerts: Annotated[
        int | None,
        typer.Option("--max-alerts", help="Optional max alerts override (top-k)."),
    ] = None,
    sample_rows: Annotated[
        int | None,
        typer.Option("--sample-rows", help="Load only the first N rows (debug).", min=1),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option("--experiment-name", help="MLflow experiment name override."),
    ] = None,
    local: Annotated[
        bool,
        typer.Option("--local/--submit", help="Run locally instead of submitting an Azure ML job."),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Stream job logs when submitting."),
    ] = False,
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute target override."),
    ] = None,
) -> None:
    """Evaluate the deployed endpoint on test data."""
    from fraud_detection.monitoring.endpoint_client import EndpointConfig
    from fraud_detection.monitoring.evaluate_endpoint import EvaluationConfig, evaluate_endpoint
    from fraud_detection.monitoring.submit_monitor import build_monitor_job_config, submit_monitor_job
    from fraud_detection.registry.prod_model import mlflow_tracking_uri

    settings = get_settings()
    ml_client = get_ml_client()

    resolved_test = (test_data or settings.registered_test).strip()
    resolved_endpoint = _normalize_optional_str(endpoint_name) or settings.endpoint_name
    resolved_deployment = _normalize_optional_str(deployment_name) or settings.deployment_name
    resolved_scoring = _normalize_optional_str(scoring_uri)
    resolved_key = _normalize_optional_str(endpoint_key)
    resolved_out = Path(out) if out else (ROOT_DIR / "outputs" / "monitoring")
    resolved_max_alerts = _normalize_positive_int(max_alerts)

    if local:
        mlflow_tracking_uri(ml_client)
        endpoint_cfg = EndpointConfig(
            endpoint_name=resolved_endpoint,
            deployment_name=resolved_deployment,
            scoring_uri=resolved_scoring,
            auth_mode=auth_mode,
            endpoint_key=resolved_key,
            batch_size=batch_size,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        eval_cfg = EvaluationConfig(label_column=label_column, sample_rows=sample_rows)
        evaluate_endpoint(
            test_data=resolved_test,
            endpoint_config=endpoint_cfg,
            output_dir=resolved_out,
            config=eval_cfg,
            ml_client=ml_client,
            data_label=data_label,
            max_alerts=resolved_max_alerts,
            experiment_name=experiment_name,
        )
        typer.echo(f"Evaluation artifacts written to: {resolved_out}")
        return

    reference_data = settings.monitor_reference_data or settings.registered_dataset_name
    job_config = build_monitor_job_config(
        test_data=resolved_test,
        reference_data=reference_data,
        endpoint_name=resolved_endpoint,
        deployment_name=resolved_deployment,
        scoring_uri=resolved_scoring,
        auth_mode=auth_mode,
        endpoint_key=resolved_key,
        label_column=label_column,
        batch_size=batch_size,
        max_retries=max_retries,
        request_timeout=request_timeout,
        max_alerts=resolved_max_alerts,
        psi_bins=10,
        psi_threshold=0.2,
        ks_threshold=0.1,
        mode="evaluate",
        compute=compute,
        experiment_name=experiment_name,
        settings=settings,
    )
    job = submit_monitor_job(ml_client, job_config)
    typer.echo(f"Submitted evaluation job: {getattr(job, 'name', None)}")
    if wait:
        ml_client.jobs.stream(getattr(job, "name"))


@app.command("check-drift")
def check_drift_cmd(
    test_data: Annotated[
        str | None,
        typer.Option("--test-data", help="Registered test data asset name or local path."),
    ] = None,
    ref_data: Annotated[
        str | None,
        typer.Option("--ref-data", help="Reference data asset name or local path."),
    ] = None,
    data_label: Annotated[
        str | None,
        typer.Option("--data-label", help="Optional data asset label."),
    ] = None,
    label_column: Annotated[
        str,
        typer.Option("--label-column", help="Label column name."),
    ] = DEFAULT_LABEL_COLUMN,
    out: Annotated[
        str | None,
        typer.Option("--out", help="Output directory for drift artifacts."),
    ] = None,
    psi_bins: Annotated[
        int,
        typer.Option("--psi-bins", help="Number of PSI bins.", min=2),
    ] = 10,
    psi_threshold: Annotated[
        float,
        typer.Option("--psi-threshold", help="PSI drift threshold.", min=0.0),
    ] = 0.2,
    ks_threshold: Annotated[
        float,
        typer.Option("--ks-threshold", help="KS drift threshold.", min=0.0),
    ] = 0.1,
    sample_rows: Annotated[
        int | None,
        typer.Option("--sample-rows", help="Load only the first N rows (debug).", min=1),
    ] = None,
    experiment_name: Annotated[
        str | None,
        typer.Option("--experiment-name", help="MLflow experiment name override."),
    ] = None,
    local: Annotated[
        bool,
        typer.Option("--local/--submit", help="Run locally instead of submitting an Azure ML job."),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Stream job logs when submitting."),
    ] = False,
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute target override."),
    ] = None,
) -> None:
    """Check drift between reference and current datasets."""
    from fraud_detection.monitoring.drift import DriftConfig, run_drift_check
    from fraud_detection.monitoring.submit_monitor import build_monitor_job_config, submit_monitor_job
    from fraud_detection.registry.prod_model import mlflow_tracking_uri

    settings = get_settings()
    ml_client = get_ml_client()

    resolved_test = (test_data or settings.registered_test).strip()
    resolved_ref = (ref_data or settings.monitor_reference_data or settings.registered_dataset_name).strip()
    resolved_out = Path(out) if out else (ROOT_DIR / "outputs" / "monitoring")

    if local:
        mlflow_tracking_uri(ml_client)
        drift_cfg = DriftConfig(
            label_column=label_column,
            psi_bins=psi_bins,
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold,
            sample_rows=sample_rows,
        )
        run_drift_check(
            reference_data=resolved_ref,
            current_data=resolved_test,
            output_dir=resolved_out,
            config=drift_cfg,
            ml_client=ml_client,
            data_label=data_label,
            experiment_name=experiment_name,
        )
        typer.echo(f"Drift artifacts written to: {resolved_out}")
        return

    job_config = build_monitor_job_config(
        test_data=resolved_test,
        reference_data=resolved_ref,
        endpoint_name=settings.endpoint_name,
        deployment_name=settings.deployment_name,
        scoring_uri=None,
        auth_mode="key",
        endpoint_key=None,
        label_column=label_column,
        batch_size=200,
        max_retries=3,
        request_timeout=30.0,
        max_alerts=None,
        psi_bins=psi_bins,
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold,
        mode="drift",
        compute=compute,
        experiment_name=experiment_name,
        settings=settings,
    )
    job = submit_monitor_job(ml_client, job_config)
    typer.echo(f"Submitted drift job: {getattr(job, 'name', None)}")
    if wait:
        ml_client.jobs.stream(getattr(job, "name"))


@app.command("monitor")
def monitor_cmd(
    test_data: Annotated[
        str | None,
        typer.Option("--test-data", help="Registered test data asset name or local path."),
    ] = None,
    ref_data: Annotated[
        str | None,
        typer.Option("--ref-data", help="Reference data asset name or local path."),
    ] = None,
    data_label: Annotated[
        str | None,
        typer.Option("--data-label", help="Optional data asset label."),
    ] = None,
    endpoint_name: Annotated[
        str | None,
        typer.Option("--endpoint-name", help="Endpoint name override."),
    ] = None,
    deployment_name: Annotated[
        str | None,
        typer.Option("--deployment-name", help="Deployment name override."),
    ] = None,
    scoring_uri: Annotated[
        str | None,
        typer.Option("--scoring-uri", help="Override scoring URI (skips endpoint lookup)."),
    ] = None,
    endpoint_key: Annotated[
        str | None,
        typer.Option("--endpoint-key", help="Endpoint key (falls back to env vars)."),
    ] = None,
    auth_mode: Annotated[
        str,
        typer.Option("--auth-mode", help="Auth mode for endpoint calls (key or aad)."),
    ] = "key",
    label_column: Annotated[
        str,
        typer.Option("--label-column", help="Label column name."),
    ] = DEFAULT_LABEL_COLUMN,
    out: Annotated[
        str | None,
        typer.Option("--out", help="Output directory for monitoring artifacts."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for endpoint calls.", min=1),
    ] = 200,
    max_retries: Annotated[
        int,
        typer.Option("--max-retries", help="Max retries for endpoint calls.", min=0),
    ] = 3,
    request_timeout: Annotated[
        float,
        typer.Option("--request-timeout", help="Request timeout (seconds).", min=1.0),
    ] = 30.0,
    max_alerts: Annotated[
        int | None,
        typer.Option("--max-alerts", help="Optional max alerts override (top-k)."),
    ] = None,
    psi_bins: Annotated[
        int,
        typer.Option("--psi-bins", help="Number of PSI bins.", min=2),
    ] = 10,
    psi_threshold: Annotated[
        float,
        typer.Option("--psi-threshold", help="PSI drift threshold.", min=0.0),
    ] = 0.2,
    ks_threshold: Annotated[
        float,
        typer.Option("--ks-threshold", help="KS drift threshold.", min=0.0),
    ] = 0.1,
    sample_rows: Annotated[
        int | None,
        typer.Option("--sample-rows", help="Load only the first N rows (debug).", min=1),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="monitor, evaluate, or drift"),
    ] = "monitor",
    experiment_name: Annotated[
        str | None,
        typer.Option("--experiment-name", help="MLflow experiment name override."),
    ] = None,
    local: Annotated[
        bool,
        typer.Option("--local/--submit", help="Run locally instead of submitting an Azure ML job."),
    ] = False,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Stream job logs when submitting."),
    ] = False,
    compute: Annotated[
        str | None,
        typer.Option("--compute", help="Azure ML compute target override."),
    ] = None,
) -> None:
    """Run evaluation + drift monitoring."""
    from fraud_detection.monitoring.drift import DriftConfig
    from fraud_detection.monitoring.endpoint_client import EndpointConfig
    from fraud_detection.monitoring.evaluate_endpoint import EvaluationConfig
    from fraud_detection.monitoring.monitor import MonitorConfig, MonitorMode, run_monitoring
    from fraud_detection.monitoring.submit_monitor import build_monitor_job_config, submit_monitor_job
    from fraud_detection.registry.prod_model import mlflow_tracking_uri

    settings = get_settings()
    ml_client = get_ml_client()

    resolved_test = (test_data or settings.registered_test).strip()
    resolved_ref = (ref_data or settings.monitor_reference_data or settings.registered_dataset_name).strip()
    resolved_endpoint = _normalize_optional_str(endpoint_name) or settings.endpoint_name
    resolved_deployment = _normalize_optional_str(deployment_name) or settings.deployment_name
    resolved_scoring = _normalize_optional_str(scoring_uri)
    resolved_key = _normalize_optional_str(endpoint_key)
    resolved_out = Path(out) if out else (ROOT_DIR / "outputs" / "monitoring")
    resolved_max_alerts = _normalize_positive_int(max_alerts)

    mode_value = (mode or "monitor").strip().lower()
    if mode_value not in {"monitor", "evaluate", "drift"}:
        raise typer.BadParameter("--mode must be one of: monitor, evaluate, drift")

    if local:
        mlflow_tracking_uri(ml_client)
        endpoint_cfg = EndpointConfig(
            endpoint_name=resolved_endpoint,
            deployment_name=resolved_deployment,
            scoring_uri=resolved_scoring,
            auth_mode=auth_mode,
            endpoint_key=resolved_key,
            batch_size=batch_size,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
        eval_cfg = EvaluationConfig(label_column=label_column, sample_rows=sample_rows)
        drift_cfg = DriftConfig(
            label_column=label_column,
            psi_bins=psi_bins,
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold,
            sample_rows=sample_rows,
        )
        monitor_cfg = MonitorConfig(
            test_data=resolved_test,
            reference_data=resolved_ref,
            output_dir=resolved_out,
            endpoint_config=endpoint_cfg,
            max_alerts=resolved_max_alerts,
            evaluation=eval_cfg,
            drift=drift_cfg,
            mode=MonitorMode(mode_value),
            experiment_name=experiment_name,
        )
        run_monitoring(config=monitor_cfg, ml_client=ml_client, data_label=data_label)
        typer.echo(f"Monitoring artifacts written to: {resolved_out}")
        return

    job_config = build_monitor_job_config(
        test_data=resolved_test,
        reference_data=resolved_ref,
        endpoint_name=resolved_endpoint,
        deployment_name=resolved_deployment,
        scoring_uri=resolved_scoring,
        auth_mode=auth_mode,
        endpoint_key=resolved_key,
        label_column=label_column,
        batch_size=batch_size,
        max_retries=max_retries,
        request_timeout=request_timeout,
        max_alerts=resolved_max_alerts,
        psi_bins=psi_bins,
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold,
        mode=mode_value,
        compute=compute,
        experiment_name=experiment_name,
        settings=settings,
    )
    job = submit_monitor_job(ml_client, job_config)
    typer.echo(f"Submitted monitoring job: {getattr(job, 'name', None)}")
    if wait:
        ml_client.jobs.stream(getattr(job, "name"))


def run() -> None:
    app()


if __name__ == "__main__":
    run()
