"""Azure ML component to deploy the production model."""

from __future__ import annotations

from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment

from fraud_detection.azure.client import get_ml_client
from fraud_detection.azure.client import resolve_azure_env_vars
from fraud_detection.config import ROOT_DIR, Settings, get_settings
from fraud_detection.utils.logging import get_logger
from fraud_detection.utils.versioning import (
    resolve_next_component_version,
    resolve_next_environment_version,
)

logger = get_logger(__name__)
DEFAULT_SERVING_ENV_NAME = "serve-prod-model-env"


def _resolve_serving_env_name(settings: Settings) -> str:
    return str(getattr(settings, "serving_env_name", None) or DEFAULT_SERVING_ENV_NAME)


def create_environment(ml_client: MLClient) -> Environment:
    settings = get_settings()
    env_name = _resolve_serving_env_name(settings)
    env_version = resolve_next_environment_version(ml_client, name=env_name)
    env = Environment(
        name=env_name,
        description="Environment for serving deployment component",
        conda_file=Path(__file__).resolve().parent / "serve_env.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        version=env_version,
    )
    created = ml_client.environments.create_or_update(env)
    logger.info(
        "Registered serving environment",
        extra={"environment": f"{created.name}:{created.version}"},
    )
    return created


def create_component() -> object:
    ml_client = get_ml_client()
    serving_env = create_environment(ml_client)
    component_version = resolve_next_component_version(ml_client, name="serve_prod_model")
    settings = get_settings()
    azure_env_vars = resolve_azure_env_vars(settings=settings)

    component = command(
        name="serve_prod_model",
        version=component_version,
        display_name="Serve Production Model",
        description="Deploy the latest production model to a managed online endpoint.",
        inputs={
            "new_promotion": Input(type=AssetTypes.URI_FILE),
            "prod_model_name": Input(type="string", default=""),
            "endpoint_name": Input(type="string", default=""),
            "deployment_name": Input(type="string", default=""),
            "scaler_dir": Input(type=AssetTypes.URI_FOLDER),
            "max_alerts": Input(type="integer"),
            "traffic_percentage": Input(type="integer", default=100),
            "dry_run": Input(type="string", default="false"),
            "force_deploy": Input(type="string", default="false"),
            "skip_env": Input(type="string", default="false"),
        },
        outputs={
            "success_flag": Output(type=AssetTypes.URI_FILE),
            "endpoint_name": Output(type=AssetTypes.URI_FILE),
            "deployment_name": Output(type=AssetTypes.URI_FILE),
            "deployed_model_name": Output(type=AssetTypes.URI_FILE),
            "deployed_model_version": Output(type=AssetTypes.URI_FILE),
            "scoring_info": Output(type=AssetTypes.URI_FILE),
        },
        environment=f"{serving_env.name}:{serving_env.version}",
        environment_variables=azure_env_vars,
        code=ROOT_DIR / "src",
        command=(
            "PYTHONPATH=. python -m fraud_detection.cli serve-prod-model "
            "--new-promotion ${{inputs.new_promotion}} "
            "--prod-model-name ${{inputs.prod_model_name}} "
            "--endpoint-name ${{inputs.endpoint_name}} "
            "--deployment-name ${{inputs.deployment_name}} "
            "--scaler-dir ${{inputs.scaler_dir}} "
            "--max-alerts ${{inputs.max_alerts}} "
            "--traffic-percentage ${{inputs.traffic_percentage}} "
            "--dry-run ${{inputs.dry_run}} "
            "--force-deploy ${{inputs.force_deploy}} "
            "--skip-env ${{inputs.skip_env}} "
            "--success-flag ${{outputs.success_flag}} "
            "--endpoint-name-out ${{outputs.endpoint_name}} "
            "--deployment-name-out ${{outputs.deployment_name}} "
            "--deployed-model-name ${{outputs.deployed_model_name}} "
            "--deployed-model-version ${{outputs.deployed_model_version}} "
            "--scoring-info ${{outputs.scoring_info}}"
        ),
    )

    logger.info(
        "Prepared serve-prod-model component",
        extra={"component": f"{component.name}:{component.version}"},
    )
    return component.component


def register_component() -> None:
    ml_client = get_ml_client()
    component = create_component()
    ml_client.components.create_or_update(component)
    logger.info("Registered serve-prod-model component")


__all__ = ["create_component", "create_environment", "register_component"]


if __name__ == "__main__":
    register_component()
