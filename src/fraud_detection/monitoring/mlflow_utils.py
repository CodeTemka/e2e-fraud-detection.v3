"""MLflow helpers for monitoring jobs."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)
_artifact_logging_disabled = False
_artifact_logging_disabled_reason: str | None = None


def _maybe_disable_artifact_logging(error: Exception) -> bool:
    global _artifact_logging_disabled, _artifact_logging_disabled_reason
    message = str(error)
    if "azureml_artifacts_builder()" in message and "tracking_uri" in message:
        _artifact_logging_disabled = True
        _artifact_logging_disabled_reason = message
        logger.warning(
            "Disabling MLflow artifact logging due to incompatible azureml-mlflow version",
            extra={"error": message},
        )
        return True
    return False


@contextmanager
def ensure_mlflow_run(
    *,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[object]:
    """Ensure there is an active MLflow run."""
    active = mlflow.active_run()
    if active is not None:
        yield active
        return

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_metrics(metrics: dict[str, float | None]) -> None:
    payload = {key: value for key, value in metrics.items() if value is not None}
    if not payload:
        return
    mlflow.log_metrics(payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
    log_artifact(path)


def log_artifact(path: Path) -> None:
    if _artifact_logging_disabled:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        if _maybe_disable_artifact_logging(exc):
            return
        logger.warning("Failed to log artifact to MLflow", extra={"path": str(path), "error": str(exc)})


__all__ = ["ensure_mlflow_run", "log_metrics", "log_json", "log_artifact", "write_json"]
