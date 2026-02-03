"""Pipeline job for validating data and preparing training splits."""

from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline

from fraud_detection.azure.client import get_ml_client, resolve_azure_env_vars
from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

VALIDATION_COMPONENT_NAME = "data_validation"
PREP_COMPONENT_NAME = "prep_data_for_train"


def _get_component(ml_client: MLClient, name: str, version: str | None) -> object:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        if version:
            component = ml_client.components.get(name=name, version=version)
        else:
            component = ml_client.components.get(name=name)
    except ResourceNotFoundError as exc:
        raise ValueError(f"No component named '{name}' found.") from exc

    return component


def _apply_env_vars_to_jobs(pipeline_job: object, env_vars: dict[str, str]) -> None:
    if not env_vars:
        return
    jobs = getattr(pipeline_job, "jobs", None)
    if not isinstance(jobs, dict):
        return
    for job in jobs.values():
        existing = getattr(job, "environment_variables", None) or {}
        merged = {**existing, **env_vars}
        job.environment_variables = merged


def create_data_pipeline_job(
    *,
    data_name: str,
    label_col: str,
    test_ratio: float,
    seed: int,
    validation_component_version: str | None = None,
    prep_component_version: str | None = None,
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
) -> object:
    """Create a pipeline job that validates data and prepares training splits."""
    ml_client = ml_client or get_ml_client()
    settings = get_settings()

    validation_component = _get_component(ml_client, VALIDATION_COMPONENT_NAME, validation_component_version)
    prep_component = _get_component(ml_client, PREP_COMPONENT_NAME, prep_component_version)

    @pipeline(
        name="data_validation_and_preparation_pipeline",
        description="Pipeline for validating data and preparing training splits.",
        default_compute=settings.pipeline_compute_cluster_name,
        experiment_name=experiment_name or "data_pipeline",
    )
    def _data_pipeline(registered_data: str, label_col: str, test_ratio: float, seed: int):
        validation_job = validation_component(data=registered_data)
        prep_component(
            registered_data=registered_data,
            label_col=label_col,
            test_ratio=test_ratio,
            seed=seed,
            is_valid=validation_job.outputs.is_valid,
        )

    pipeline_job = _data_pipeline(
        registered_data=data_name,
        label_col=label_col,
        test_ratio=test_ratio,
        seed=seed,
    )

    azure_env_vars = resolve_azure_env_vars(settings=settings)
    _apply_env_vars_to_jobs(pipeline_job, azure_env_vars)

    return pipeline_job


def submit_data_pipeline_job(
    *,
    data_name: str,
    label_col: str,
    test_ratio: float,
    seed: int,
    validation_component_version: str | None = None,
    prep_component_version: str | None = None,
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
    ensure_compute: bool = True,
) -> object:
    """Submit the data pipeline job to Azure ML."""
    from fraud_detection.azure.client import get_ml_client
    from fraud_detection.utils.compute import ensure_pipeline_compute

    ml_client = ml_client or get_ml_client()
    if ensure_compute:
        ensure_pipeline_compute(ml_client)

    pipeline_job = create_data_pipeline_job(
        data_name=data_name,
        label_col=label_col,
        test_ratio=test_ratio,
        seed=seed,
        validation_component_version=validation_component_version,
        prep_component_version=prep_component_version,
        experiment_name=experiment_name,
        ml_client=ml_client,
    )
    return ml_client.jobs.create_or_update(pipeline_job)
