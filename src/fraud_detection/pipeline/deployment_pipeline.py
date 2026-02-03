"""Pipeline job for promoting the best model to production."""

from azure.ai.ml import Input, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from fraud_detection.azure.client import get_ml_client, resolve_azure_env_vars
from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

PROMOTE_COMPONENT_NAME = "promote_prod_model"
SERVE_COMPONENT_NAME = "serve_prod_model"


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


def _normalize_compare_metric(compare_metric: str | None) -> str:
    if compare_metric is None:
        return ""
    return compare_metric.strip()


def _normalize_experiments(experiments: str | None) -> str:
    if experiments is None:
        return ""
    return experiments.strip()


def _parse_version(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _version_sort_key(asset: object) -> tuple[int, tuple[int, ...] | str]:
    version_text = str(getattr(asset, "version", "") or "")
    parsed = _parse_version(version_text)
    if parsed is not None:
        return (1, parsed)
    return (0, version_text.lower())


def _resolve_scaler_asset_uri(
    ml_client: MLClient,
    *,
    scaler_asset_name: str | None,
    settings: object,
) -> str:
    name = str(scaler_asset_name or getattr(settings, "serving_scalers_name", "")).strip()
    if not name:
        raise ValueError("Scaler asset name must be provided.")

    if name.startswith("azureml:") or name.startswith("azureml://"):
        return name

    assets = list(ml_client.data.list(name=name))
    if not assets:
        raise ValueError(f"No scaler data assets found with name '{name}'.")

    latest = max(assets, key=_version_sort_key)
    version = str(getattr(latest, "version", "") or "")
    if not version:
        raise ValueError(f"Scaler asset '{name}' has no version.")

    return f"azureml:{name}:{version}"


def create_deployment_pipeline_job(
    *,
    compare_metric: str | None = None,
    experiments: str | None = None,
    dry_run: bool = False,
    force_deploy: bool = False,
    skip_env: bool = False,
    component_version: str | None = None,
    serve_component_version: str | None = None,
    experiment_name: str | None = None,
    scaler_asset_name: str | None = None,
    prod_model_name: str | None = None,
    endpoint_name: str | None = None,
    deployment_name: str | None = None,
    max_alerts: int | None = None,
    traffic_percentage: int | None = None,
    ml_client: MLClient | None = None,
) -> object:
    """Create a pipeline job that promotes and serves the best model to production."""
    ml_client = ml_client or get_ml_client()
    settings = get_settings()

    promote_component = _get_component(ml_client, PROMOTE_COMPONENT_NAME, component_version)
    serve_component = _get_component(ml_client, SERVE_COMPONENT_NAME, serve_component_version)
    resolved_metric = _normalize_compare_metric(compare_metric) or settings.default_metric_serving
    resolved_experiments = _normalize_experiments(experiments)
    if not resolved_experiments:
        resolved_experiments = f"{settings.custom_train_exp},{settings.automl_train_exp}"

    resolved_prod_model_name = (prod_model_name or settings.prod_model_name).strip()
    resolved_endpoint_name = (endpoint_name or settings.endpoint_name).strip()
    resolved_deployment_name = (deployment_name or settings.deployment_name).strip()
    resolved_scaler_uri = _resolve_scaler_asset_uri(
        ml_client,
        scaler_asset_name=scaler_asset_name,
        settings=settings,
    )
    resolved_max_alerts = int(max_alerts if max_alerts is not None else 100)
    resolved_traffic = int(traffic_percentage if traffic_percentage is not None else 100)

    @pipeline(
        name="deployment_pipeline",
        description="Pipeline for selecting, promoting, and serving the best model to production.",
        default_compute=settings.pipeline_compute_cluster_name,
        experiment_name=experiment_name or "deployment_pipeline",
    )
    def _deployment_pipeline(
        compare_metric: str,
        experiments: str,
        dry_run: str,
        force_deploy: str,
        skip_env: str,
        prod_model_name: str,
        endpoint_name: str,
        deployment_name: str,
        scaler_dir: str,
        max_alerts: int,
        traffic_percentage: int,
    ):
        promote_job = promote_component(
            compare_metric=compare_metric,
            experiments=experiments,
            dry_run=dry_run,
        )
        serve_inputs = {
            "new_promotion": promote_job.outputs.new_promotion,
            "prod_model_name": prod_model_name,
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "scaler_dir": scaler_dir,
            "max_alerts": max_alerts,
            "traffic_percentage": traffic_percentage,
            "dry_run": dry_run,
        }
        component_inputs = getattr(serve_component, "inputs", None)
        optional_flags = {
            "force_deploy": force_deploy,
            "skip_env": skip_env,
        }
        for key, value in optional_flags.items():
            if isinstance(component_inputs, dict) and key in component_inputs:
                serve_inputs[key] = value
            elif value == "true":
                raise ValueError(
                    f"{key} requested but serve_prod_model component lacks the input. " "Re-register the component."
                )
        serve_job = serve_component(**serve_inputs)
        return {
            "production_info": promote_job.outputs.production_info,
            "new_promotion": promote_job.outputs.new_promotion,
            "deployment_success": serve_job.outputs.success_flag,
            "deployment_endpoint_name": serve_job.outputs.endpoint_name,
            "deployment_name": serve_job.outputs.deployment_name,
            "deployed_model_name": serve_job.outputs.deployed_model_name,
            "deployed_model_version": serve_job.outputs.deployed_model_version,
            "deployment_scoring_info": serve_job.outputs.scoring_info,
        }

    pipeline_job = _deployment_pipeline(
        compare_metric=resolved_metric,
        experiments=resolved_experiments,
        dry_run="true" if dry_run else "false",
        force_deploy="true" if force_deploy else "false",
        skip_env="true" if skip_env else "false",
        prod_model_name=resolved_prod_model_name,
        endpoint_name=resolved_endpoint_name,
        deployment_name=resolved_deployment_name,
        scaler_dir=Input(type=AssetTypes.URI_FOLDER, path=resolved_scaler_uri),
        max_alerts=resolved_max_alerts,
        traffic_percentage=resolved_traffic,
    )

    azure_env_vars = resolve_azure_env_vars(settings=settings)
    _apply_env_vars_to_jobs(pipeline_job, azure_env_vars)
    return pipeline_job


def submit_deployment_pipeline_job(
    *,
    compare_metric: str | None = None,
    experiments: str | None = None,
    dry_run: bool = False,
    force_deploy: bool = False,
    skip_env: bool = False,
    component_version: str | None = None,
    serve_component_version: str | None = None,
    experiment_name: str | None = None,
    scaler_asset_name: str | None = None,
    prod_model_name: str | None = None,
    endpoint_name: str | None = None,
    deployment_name: str | None = None,
    max_alerts: int | None = None,
    traffic_percentage: int | None = None,
    ml_client: MLClient | None = None,
    ensure_compute: bool = True,
) -> object:
    """Submit the deployment pipeline job to Azure ML."""
    from fraud_detection.utils.compute import ensure_pipeline_compute

    ml_client = ml_client or get_ml_client()
    if ensure_compute:
        ensure_pipeline_compute(ml_client)

    pipeline_job = create_deployment_pipeline_job(
        compare_metric=compare_metric,
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
    return ml_client.jobs.create_or_update(pipeline_job)


__all__ = [
    "create_deployment_pipeline_job",
    "submit_deployment_pipeline_job",
]
