"""Compare registered models and promote the best candidate to production."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _set_mlflow_tracking_uri(ml_client: MLClient) -> None:
    uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    if uri:
        mlflow.set_tracking_uri(uri)


@dataclass(frozen=True)
class CandidateScore:
    model: Model
    metric_value: float | None
    reason: str | None


@dataclass(frozen=True)
class PromotionOutcome:
    promoted: bool
    promoted_model: Model | None
    winner: Model | None
    winner_metric: float | None
    prod_model: Model | None
    prod_metric: float | None
    comparison_table: list[dict[str, Any]]


def _parse_version(version: str | None) -> int:
    try:
        return int(version or 0)
    except (TypeError, ValueError):
        return 0
    

def _latest_model_asset(ml_client: MLClient, name: str) -> Model | None:
    try:
        models = list(ml_client.models.list(name=name))
    except HttpResponseError as exc:
        logger.debug(f"Failed listing models for name {name}: {exc}")
        return None
    if not models:
        return None
    return max(models, key=lambda m: _parse_version(getattr(m, "version", None)))

