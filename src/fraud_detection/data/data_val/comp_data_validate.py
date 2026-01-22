"""Azure ML Reusable component for validating data"""

from __future__ import annotations

from pathlib import Path
from azure.ai.ml import command, Input, Output, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.versioning import (
    resolve_next_component_version,
    resolve_next_environment_version,
)
from fraud_detection.utils.logging import get_logger
from fraud_detection.config import ROOT_DIR


logger = get_logger(__name__)


def create_environment(ml_client: MLClient) -> Environment:
    env_version = resolve_next_environment_version(ml_client, name="data-validation-env")
    data_validation_env = Environment(
        name="data-validation-env",
        description="Environment for data validation component",
        conda_file=Path(__file__).resolve().parent / "data_validate_env.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        version=env_version,
    )
    created = ml_client.environments.create_or_update(data_validation_env)
    logger.info(
        "Registered data validation environment",
        extra={"environment": f"{created.name}:{created.version}"},
    )
    return created


def create_command():
    ml_client = get_ml_client()
    data_validation_env = create_environment(ml_client)
    component_version = resolve_next_component_version(ml_client, name="data_validation")
    data_validation = command(
        name="data_validation",
        version=component_version,
        display_name="Validate Data",
        description="Validate the given data for schema, missing values, and basic numeric sanity",
        inputs={
            "data": Input(type="string")
        },
        outputs={
            "validation_report": Output(type=AssetTypes.URI_FILE),
            "is_valid": Output(type=AssetTypes.URI_FILE),
        },
        environment=f"{data_validation_env.name}:{data_validation_env.version}",
        code=ROOT_DIR / "src",
        command=(
            "PYTHONPATH=. python -m fraud_detection.cli validate-data "
            "--dataset-name ${{inputs.data}} "
            "--validation-report ${{outputs.validation_report}} "
            "--is-valid ${{outputs.is_valid}}"
        ),
    )

    logger.info(
        "Prepared data validation component",
        extra={"component": f"{data_validation.name}:{data_validation.version}"},
    )
    return data_validation.component


def register_the_component():
    ml_client = get_ml_client()
    data_validation_component = create_command()
    ml_client.components.create_or_update(data_validation_component)


if __name__ == "__main__":
    register_the_component()
