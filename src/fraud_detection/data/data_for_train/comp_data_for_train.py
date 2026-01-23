"""Azure ML component for preparing data splits and scaling columns for training."""

from __future__ import annotations

from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment

from fraud_detection.azure.client import get_ml_client
from fraud_detection.utils.versioning import (
    resolve_next_component_version,
    resolve_next_environment_version,
)
from fraud_detection.config import ROOT_DIR
from fraud_detection.utils.logging import get_logger


logger = get_logger(__name__)

def create_environment(ml_client: MLClient) -> Environment:
    env_version = resolve_next_environment_version(ml_client, name="data-for-train-env")
    env = Environment(
        name="data-for-train-env",
        description="Environment for data preprocessing component",
        conda_file=Path(__file__).resolve().parent / "data_for_train_env.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        version=env_version,
    )
    created = ml_client.environments.create_or_update(env)
    logger.info(
        "Registered data prep environment",
        extra={"environment": f"{created.name}:{created.version}"},
    )
    return created


def create_command():
    ml_client = get_ml_client()
    prep_env = create_environment(ml_client)
    component_version = resolve_next_component_version(ml_client, name="prep_data_for_train")

    prep_component = command(
        name="prep_data_for_train",
        version=component_version,
        display_name="Prep Data For Train",
        description="Split data and scale Amount/Time for training.",
        inputs={
            "registered_data": Input(type="string"),
            "label_col": Input(type="string", default="Class"),
            "test_ratio": Input(type="number", default=0.2),
            "seed": Input(type="integer", default=42),
            "is_valid": Input(type=AssetTypes.URI_FILE),
        },
        outputs={
            "scalers": Output(type=AssetTypes.URI_FOLDER),
            "metadata": Output(type=AssetTypes.URI_FILE)
        },
        environment=f"{prep_env.name}:{prep_env.version}",
        code=ROOT_DIR / "src",
        command=(
            "PYTHONPATH=. python -m fraud_detection.cli prep-data-for-train "
            "--data-name ${{inputs.registered_data}} "
            "--label-col ${{inputs.label_col}} "
            "--test-ratio ${{inputs.test_ratio}} "
            "--seed ${{inputs.seed}} "
            "--is-valid ${{inputs.is_valid}} "
            "--scalers ${{outputs.scalers}} "
            "--metadata ${{outputs.metadata}}"
        ),
    )

    logger.info(
        "Created data prep component",
        extra={"component": f"{prep_component.name}:{prep_component.version}"},
    )

    return prep_component.component


def register_component() -> None:
    ml_client = get_ml_client()
    component = create_command()
    ml_client.components.create_or_update(component)


if __name__ == "__main__":
    register_component()
