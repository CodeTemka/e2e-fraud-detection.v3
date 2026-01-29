"""Pipeline job for promoting the best model to production."""

from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline

from fraud_detection.azure.client import get_ml_client, resolve_azure_env_vars
from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

PROMOTE_COMPONENT_NAME = "promote_prod_model"


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
        setattr(job, "environment_variables", merged)


def _normalize_compare_metric(compare_metric: str | None) -> str:
    if compare_metric is None:
        return ""
    return compare_metric.strip()


def _normalize_experiments(experiments: str | None) -> str:
    if experiments is None:
        return ""
    return experiments.strip()


def create_deployment_pipeline_job(
    *,
    compare_metric: str | None = None,
    experiments: str | None = None,
    dry_run: bool = False,
    component_version: str | None = None,
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
) -> object:
    """Create a pipeline job that promotes the best model to production."""
    ml_client = ml_client or get_ml_client()
    settings = get_settings()

    promote_component = _get_component(ml_client, PROMOTE_COMPONENT_NAME, component_version)
    resolved_metric = _normalize_compare_metric(compare_metric) or settings.default_metric_serving
    resolved_experiments = _normalize_experiments(experiments)
    if not resolved_experiments:
        resolved_experiments = f"{settings.custom_train_exp},{settings.automl_train_exp}"

    @pipeline(
        name="deployment_pipeline",
        description="Pipeline for selecting and promoting the best model to production.",
        default_compute=settings.data_compute_cluster_name,
        experiment_name=experiment_name or "deployment_pipeline",
    )
    def _deployment_pipeline(compare_metric: str, experiments: str, dry_run: str):
        promote_job = promote_component(
            compare_metric=compare_metric,
            experiments=experiments,
            dry_run=dry_run,
        )
        return {
            "production_info": promote_job.outputs.production_info,
            "new_promotion": promote_job.outputs.new_promotion,
        }

    pipeline_job = _deployment_pipeline(
        compare_metric=resolved_metric,
        experiments=resolved_experiments,
        dry_run="true" if dry_run else "false",
    )

    azure_env_vars = resolve_azure_env_vars(settings=settings)
    _apply_env_vars_to_jobs(pipeline_job, azure_env_vars)
    return pipeline_job


def submit_deployment_pipeline_job(
    *,
    compare_metric: str | None = None,
    experiments: str | None = None,
    dry_run: bool = False,
    component_version: str | None = None,
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
    ensure_compute: bool = True,
) -> object:
    """Submit the deployment pipeline job to Azure ML."""
    from fraud_detection.utils.compute import ensure_data_compute

    ml_client = ml_client or get_ml_client()
    if ensure_compute:
        ensure_data_compute(ml_client)

    pipeline_job = create_deployment_pipeline_job(
        compare_metric=compare_metric,
        experiments=experiments,
        dry_run=dry_run,
        component_version=component_version,
        experiment_name=experiment_name,
        ml_client=ml_client,
    )
    return ml_client.jobs.create_or_update(pipeline_job)


__all__ = [
    "create_deployment_pipeline_job",
    "submit_deployment_pipeline_job",
]
