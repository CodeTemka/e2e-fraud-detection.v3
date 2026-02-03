"""Azure ML component for selecting and promoting the production model."""

from __future__ import annotations

from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment

from fraud_detection.azure.client import get_ml_client, resolve_azure_env_vars
from fraud_detection.config import ROOT_DIR, get_settings
from fraud_detection.utils.logging import get_logger
from fraud_detection.utils.versioning import (
    resolve_next_component_version,
    resolve_next_environment_version,
)

logger = get_logger(__name__)


def create_environment(ml_client: MLClient) -> Environment:
    env_version = resolve_next_environment_version(ml_client, name="prod-model-env")
    env = Environment(
        name="prod-model-env",
        description="Environment for production model selection component",
        conda_file=Path(__file__).resolve().parent / "prod_model_env.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        version=env_version,
    )
    created = ml_client.environments.create_or_update(env)
    logger.info(
        "Registered production model environment",
        extra={"environment": f"{created.name}:{created.version}"},
    )
    return created


def create_command():
    ml_client = get_ml_client()
    env = create_environment(ml_client)
    component_version = resolve_next_component_version(ml_client, name="promote_prod_model")
    settings = get_settings()
    azure_env_vars = resolve_azure_env_vars(settings=settings)

    prod_component = command(
        name="promote_prod_model",
        version=component_version,
        display_name="Promote Production Model",
        description="Compare experiment runs and promote the best model to production.",
        inputs={
            "compare_metric": Input(type="string", default=settings.default_metric_serving),
            "experiments": Input(
                type="string",
                default=f"{settings.custom_train_exp},{settings.automl_train_exp}",
            ),
            "dry_run": Input(type="string", default="false"),
        },
        outputs={
            "production_info": Output(type=AssetTypes.URI_FILE),
            "new_promotion": Output(type=AssetTypes.URI_FILE),
        },
        environment=f"{env.name}:{env.version}",
        environment_variables=azure_env_vars,
        code=ROOT_DIR / "src",
        command=(
            "PYTHONPATH=. python -m fraud_detection.cli promote-prod-model "
            "--compare-metric ${{inputs.compare_metric}} "
            "--experiments ${{inputs.experiments}} "
            "--dry-run ${{inputs.dry_run}} "
            "--production-info ${{outputs.production_info}} "
            "--new-promotion ${{outputs.new_promotion}}"
        ),
    )

    logger.info(
        "Prepared production model component",
        extra={"component": f"{prod_component.name}:{prod_component.version}"},
    )
    return prod_component.component


def register_component() -> None:
    ml_client = get_ml_client()
    component = create_command()
    ml_client.components.create_or_update(component)


if __name__ == "__main__":
    register_component()
