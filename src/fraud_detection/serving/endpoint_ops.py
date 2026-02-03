"""Endpoint creation and deletion helpers for Azure ML online endpoints."""

from __future__ import annotations

from typing import Mapping

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def create_endpoint(
    ml_client: MLClient,
    *,
    name: str,
    description: str | None = None,
    auth_mode: str = "key",
    tags: Mapping[str, str] | None = None,
    location: str | None = None,
    settings: Settings | None = None,
) -> ManagedOnlineEndpoint:
    cfg = settings or get_settings()
    endpoint = ManagedOnlineEndpoint(
        name=name,
        description=description or "Online endpoint for fraud detection model",
        auth_mode=auth_mode,
        tags=dict(tags) if tags else {"project": "fraud-detection"},
        location=location or cfg.location,
    )
    logger.info("Creating/updating endpoint", extra={"endpoint_name": name})
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def delete_endpoint(ml_client: MLClient, *, name: str) -> None:
    try:
        ml_client.online_endpoints.get(name)
    except ResourceNotFoundError:
        logger.info("Endpoint not found; skipping delete", extra={"endpoint_name": name})
        return

    logger.info("Deleting endpoint", extra={"endpoint_name": name})
    ml_client.online_endpoints.begin_delete(name).result()


__all__ = ["create_endpoint", "delete_endpoint"]
