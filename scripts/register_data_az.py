from __future__ import annotations

from pathlib import Path

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def register_local_data() -> None:
    settings = get_settings().require_az_id()
    ml_client = get_ml_client(settings=settings)
    
    dataset_name = settings.registered_dataset_name
    data_path = settings.local_data_path
    version = "1"
    try:
        ml_client.data.get(name=dataset_name, version=version)
        logger.info(f"Dataset {dataset_name} version {version} already registered.")
        return
    except ResourceNotFoundError:
        pass  # doesn't exist yet -> create

    
    data_asset = Data(
        name=dataset_name,
        version=version,
        path=str(data_path),
        type=AssetTypes.MLTABLE,
        description="Local CSV registered as MLTable",
    )

    ml_client.data.create_or_update(data_asset)
    logger.info(f"Registered dataset {dataset_name} version {version}.")


if __name__ == "__main__":
    register_local_data()