"""Compare experiment runs and promote the best candidate to production."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import mlflow
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceNotFoundError
from mlflow.tracking import MlflowClient

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

_MINIMIZE_HINTS = {"loss", "error", "rmse", "mae", "mse", "log_loss", "logloss"}


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    promotion_reason: str
    compare_metric: str
    metric_column: str
    maximize_metric: bool
    candidate_run_id: str
    candidate_experiment: str | None
    candidate_metric_value: float
    candidate_artifact_uri: str | None
    candidate_parent_run_id: str | None
    production_model_name: str
    production_model_version: str | None
    previous_model_version: str | None
    previous_model_metric: float | None
    previous_model_run_id: str | None
    model_source: str | None
    dry_run: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "promotion_reason": self.promotion_reason,
            "compare_metric": self.compare_metric,
            "metric_column": self.metric_column,
            "maximize_metric": self.maximize_metric,
            "candidate": {
                "run_id": self.candidate_run_id,
                "experiment": self.candidate_experiment,
                "metric_value": self.candidate_metric_value,
                "artifact_uri": self.candidate_artifact_uri,
                "parent_run_id": self.candidate_parent_run_id,
            },
            "production_model": {
                "name": self.production_model_name,
                "version": self.production_model_version,
            },
            "previous_production_model": {
                "version": self.previous_model_version,
                "metric_value": self.previous_model_metric,
                "run_id": self.previous_model_run_id,
            },
            "model_source": self.model_source,
            "dry_run": self.dry_run,
        }


def _set_mlflow_tracking_uri(ml_client: MLClient) -> None:
    uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    if uri:
        mlflow.set_tracking_uri(uri)


def _normalize_experiments(experiments: Sequence[str] | None) -> list[str] | None:
    if not experiments:
        return None
    cleaned: list[str] = []
    for item in experiments:
        if not item:
            continue
        for part in str(item).split(","):
            name = part.strip()
            if name:
                cleaned.append(name)
    return cleaned or None


def _filter_existing_experiments(experiments: Sequence[str]) -> list[str]:
    try:
        available = {
            exp.name.lower(): exp.name
            for exp in mlflow.search_experiments()
            if getattr(exp, "name", None)
        }
    except Exception as exc:
        logger.warning("Failed to list MLflow experiments; using provided names", extra={"error": str(exc)})
        return list(experiments)

    resolved: list[str] = []
    for name in experiments:
        key = name.strip().lower()
        if not key:
            continue
        resolved_name = available.get(key)
        if resolved_name:
            resolved.append(resolved_name)
        else:
            logger.warning("Experiment not found; skipping", extra={"experiment": name})
    return resolved


def _normalize_metric(metric: str) -> str:
    return metric.strip().replace("-", "_")


def _resolve_metric_column(metric: str, columns: Iterable[str]) -> str:
    metric_name = _normalize_metric(metric)
    candidate = metric_name if metric_name.startswith("metrics.") else f"metrics.{metric_name}"
    columns_list = list(columns)
    if candidate in columns_list:
        return candidate
    for column in columns_list:
        if column.lower() == candidate.lower():
            return column
    suffix_matches = [col for col in columns_list if col.lower().endswith(f".{metric_name.lower()}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    metric_columns = sorted(col for col in columns_list if col.startswith("metrics."))
    raise ValueError(
        "Metric not found in runs: "
        f"{metric_name}. Available metrics: {', '.join(metric_columns) if metric_columns else 'none'}"
    )


def _is_higher_better(metric_column: str) -> bool:
    metric_name = metric_column.split("metrics.", 1)[-1].lower()
    return not any(hint in metric_name for hint in _MINIMIZE_HINTS)


def _safe_str(value: object | None) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _select_best_run(runs: pd.DataFrame, metric_column: str, maximize: bool) -> tuple[pd.Series, float]:
    if "status" in runs.columns:
        runs = runs[runs["status"].astype(str).str.upper() == "FINISHED"]

    working = runs.copy()
    working["_metric"] = pd.to_numeric(working[metric_column], errors="coerce")
    working = working.dropna(subset=["_metric"])
    if working.empty:
        raise ValueError(f"No runs found with metric {metric_column}.")

    idx = working["_metric"].idxmax() if maximize else working["_metric"].idxmin()
    row = working.loc[idx]
    return row, float(row["_metric"])


def _resolve_experiment_name(experiment_id: str | None) -> str | None:
    if not experiment_id:
        return None
    try:
        experiment = mlflow.get_experiment(experiment_id)
    except Exception:
        return None
    return getattr(experiment, "name", None)


def _resolve_best_child_run(row: pd.Series, metric_name: str) -> tuple[str, float, str | None]:
    best_child_id = _safe_str(row.get("tags.automl_best_child_run_id"))
    if not best_child_id:
        run_id = _safe_str(row.get("run_id")) or ""
        metric_value = float(row.get("_metric"))
        return run_id, metric_value, None

    try:
        child_run = mlflow.get_run(best_child_id)
    except Exception as exc:
        logger.warning("Failed to fetch AutoML best child run; using parent", extra={"error": str(exc)})
        run_id = _safe_str(row.get("run_id")) or ""
        metric_value = float(row.get("_metric"))
        return run_id, metric_value, None

    child_metric = child_run.data.metrics.get(metric_name)
    if child_metric is None:
        run_id = _safe_str(row.get("run_id")) or ""
        metric_value = float(row.get("_metric"))
        return run_id, metric_value, None

    return best_child_id, float(child_metric), _safe_str(row.get("run_id"))


def _parse_version(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    parts = str(value).split(".")
    if any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _get_latest_model(ml_client: MLClient, name: str) -> Model | None:
    try:
        models = list(ml_client.models.list(name=name))
    except ResourceNotFoundError:
        return None

    if not models:
        return None

    best: Model | None = None
    best_version: tuple[int, ...] | None = None
    for model in models:
        version = _parse_version(getattr(model, "version", None))
        if version is None:
            continue
        if best_version is None or version > best_version:
            best_version = version
            best = model

    if best is not None:
        return best

    return max(models, key=lambda item: str(getattr(item, "version", "")))


def _resolve_job_output_uri(ml_client: MLClient, run_id: str) -> str | None:
    try:
        job = ml_client.jobs.get(run_id)
    except Exception:
        return None
    outputs = getattr(job, "outputs", None)
    if not isinstance(outputs, dict):
        return None

    output_names = ("model_output", "model", "best_model", "trained_model")
    for name in output_names:
        output = outputs.get(name)
        if output is None:
            continue
        uri = getattr(output, "uri", None) or getattr(output, "path", None)
        if not uri and isinstance(output, str):
            uri = output
        if uri:
            if name == "model_output":
                normalized = uri.rstrip("/")
                return uri if normalized.endswith("/model") else f"{uri}/model"
            return uri
    return None


def _download_model_artifact(run_id: str) -> Path | None:
    candidates = [
        "outputs/model_output/model",
        "outputs/model_output",
        "outputs/model",
        "outputs/mlflow-model",
        "outputs/trained_model",
        "model",
        "mlflow-model",
        "model_output/model",
    ]
    for artifact_path in candidates:
        local_path = _try_download_artifact(run_id, artifact_path)
        if local_path:
            return local_path

    discovered = _find_model_artifact_dir(run_id)
    if discovered:
        local_path = _try_download_artifact(run_id, discovered)
        if local_path:
            return local_path
    root_download = _download_artifact_root(run_id)
    if root_download:
        mlmodel_dir = _find_mlmodel_dir(root_download)
        if mlmodel_dir:
            return mlmodel_dir
    return None


def _try_download_artifact(run_id: str, artifact_path: str) -> Path | None:
    try:
        local_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path))
    except Exception:
        return None
    if local_path.exists():
        return local_path
    return None


def _download_artifact_root(run_id: str) -> Path | None:
    try:
        local_path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=""))
    except Exception:
        return None
    if local_path.exists():
        return local_path
    return None


def _find_mlmodel_dir(root: Path) -> Path | None:
    try:
        candidates = list(root.rglob("MLmodel"))
    except Exception:
        return None
    if not candidates:
        return None
    # Prefer the shortest path (closest to root) to avoid nested artifacts.
    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0].parent


def _find_model_artifact_dir(run_id: str, *, max_depth: int = 5) -> str | None:
    client = MlflowClient()

    def _contains_mlmodel(path: str) -> bool:
        try:
            for item in client.list_artifacts(run_id, path):
                if not item.is_dir and item.path.endswith("MLmodel"):
                    return True
        except Exception:
            return False
        return False

    queue: list[tuple[str, int]] = [("", 0)]
    seen: set[str] = set()

    while queue:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        try:
            artifacts = client.list_artifacts(run_id, current or None)
        except Exception:
            continue
        for item in artifacts:
            if item.is_dir:
                if item.path.endswith("model") or item.path.endswith("mlflow-model"):
                    if _contains_mlmodel(item.path):
                        return item.path
                if depth < max_depth:
                    queue.append((item.path, depth + 1))
            elif item.path.endswith("MLmodel"):
                return item.path.rsplit("/", 1)[0]
    return None


def _resolve_logged_model_path(run_id: str) -> str | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception:
        return None
    history = run.data.tags.get("mlflow.log-model.history")
    if not history:
        return None
    try:
        entries = json.loads(history)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict):
            path = entry.get("artifact_path")
            if isinstance(path, str) and path.strip():
                return path.strip().lstrip("/")
    return None


def _resolve_registered_model(run_id: str) -> tuple[str | None, str | None]:
    client = MlflowClient()
    try:
        for model in client.search_registered_models(f"tags.mlflow.runId = '{run_id}'"):
            name = getattr(model, "name", None)
            if name:
                return f"models:/{name}/latest", "mlflow_registry"
    except Exception:
        return None, None
    return None, None


def _resolve_parent_run_id(run_id: str) -> str | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception:
        return None
    tags = run.data.tags
    parent = _safe_str(tags.get("mlflow.parentRunId")) or _safe_str(tags.get("mlflow.rootRunId"))
    if parent == run_id:
        return None
    return parent


def _is_allowed_model_uri(value: str) -> bool:
    uri = value.strip()
    if not uri:
        return False
    if uri.startswith("runs:/"):
        return True
    if uri.startswith("azureml://jobs/"):
        return True
    if uri.startswith("azureml://datasets/"):
        return True
    if uri.startswith("azureml://subscriptions/") and "/datastores/" in uri and "/paths/" in uri:
        return True
    if uri.startswith("https://") and ".blob.core." in uri:
        return True
    return False


def _resolve_model_uri_from_tags(run_id: str) -> str | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception:
        return None
    tags = run.data.tags
    for key in ("model_artifact_uri", "artifact_uri", "model_uri"):
        value = _safe_str(tags.get(key))
        if value and _is_allowed_model_uri(value):
            return value
    return None


def _resolve_model_source(
    ml_client: MLClient,
    run_id: str,
    artifact_uri: str | None,
    parent_run_id: str | None = None,
) -> tuple[str | Path | None, str | None]:
    output_uri = _resolve_job_output_uri(ml_client, run_id)
    if output_uri:
        return output_uri, "job_output"

    tagged_uri = _resolve_model_uri_from_tags(run_id)
    if tagged_uri:
        return tagged_uri, "mlflow_tag"

    logged_model = _resolve_logged_model_path(run_id)
    if logged_model:
        return f"runs:/{run_id}/{logged_model}", "mlflow_logged_model"

    artifact_path = _find_model_artifact_dir(run_id)
    if artifact_path:
        return f"runs:/{run_id}/{artifact_path}", "mlflow_run"

    registered_model, registry_kind = _resolve_registered_model(run_id)
    if registered_model:
        return registered_model, registry_kind

    resolved_parent = parent_run_id or _resolve_parent_run_id(run_id)
    if resolved_parent:
        parent_output = _resolve_job_output_uri(ml_client, resolved_parent)
        if parent_output:
            return parent_output, "parent_job_output"

        parent_tagged = _resolve_model_uri_from_tags(resolved_parent)
        if parent_tagged:
            return parent_tagged, "mlflow_parent_tag"

        parent_logged = _resolve_logged_model_path(resolved_parent)
        if parent_logged:
            return f"runs:/{resolved_parent}/{parent_logged}", "mlflow_parent_logged_model"

        parent_artifact = _find_model_artifact_dir(resolved_parent)
        if parent_artifact:
            return f"runs:/{resolved_parent}/{parent_artifact}", "mlflow_parent_run"

        parent_registered, parent_registry_kind = _resolve_registered_model(resolved_parent)
        if parent_registered:
            return parent_registered, parent_registry_kind

    if artifact_uri and _is_allowed_model_uri(artifact_uri):
        return artifact_uri, "artifact_uri"

    local_path = _download_model_artifact(run_id)
    if local_path:
        return local_path, "mlflow_artifact"
    if resolved_parent:
        parent_local = _download_model_artifact(resolved_parent)
        if parent_local:
            return parent_local, "mlflow_parent_artifact"

    return None, None


def _coerce_metric_value(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _should_promote(
    *,
    candidate_metric: float,
    compare_metric: str,
    maximize: bool,
    current_model: Model | None,
    candidate_run_id: str,
) -> tuple[bool, str, float | None, str | None]:
    if current_model is None:
        return True, "no_existing_production_model", None, None

    tags = getattr(current_model, "tags", None) or {}
    existing_metric_name = _safe_str(tags.get("compare_metric"))
    existing_metric_value = _coerce_metric_value(_safe_str(tags.get("compare_metric_value")))
    existing_run_id = _safe_str(tags.get("source_run_id"))

    if existing_run_id and existing_run_id == candidate_run_id:
        return False, "candidate_already_promoted", existing_metric_value, existing_run_id

    if existing_metric_value is None or existing_metric_name is None:
        return True, "existing_model_missing_metric", existing_metric_value, existing_run_id

    if existing_metric_name != compare_metric:
        return True, "existing_model_metric_mismatch", existing_metric_value, existing_run_id

    if maximize and candidate_metric <= existing_metric_value:
        return False, "candidate_not_better", existing_metric_value, existing_run_id
    if not maximize and candidate_metric >= existing_metric_value:
        return False, "candidate_not_better", existing_metric_value, existing_run_id

    return True, "candidate_better_than_current", existing_metric_value, existing_run_id


def promote_best_model(
    *,
    compare_metric: str | None = None,
    experiments: Sequence[str] | None = None,
    dry_run: bool = False,
) -> PromotionResult:
    settings = get_settings()
    ml_client = get_ml_client()
    _set_mlflow_tracking_uri(ml_client)

    resolved_metric = compare_metric or settings.default_metric_serving
    resolved_metric = _normalize_metric(resolved_metric)
    resolved_experiments = _normalize_experiments(experiments) or [
        settings.custom_train_exp,
        settings.automl_train_exp,
    ]
    resolved_experiments = _filter_existing_experiments(resolved_experiments)
    if not resolved_experiments:
        raise ValueError("No matching experiments found.")

    runs = mlflow.search_runs(experiment_names=resolved_experiments)
    if runs.empty:
        raise ValueError(f"No runs found for experiments: {', '.join(resolved_experiments)}")

    metric_column = _resolve_metric_column(resolved_metric, runs.columns)
    maximize = _is_higher_better(metric_column)

    row, _ = _select_best_run(runs, metric_column, maximize)
    metric_name = metric_column.split("metrics.", 1)[-1]
    candidate_run_id, candidate_metric, parent_run_id = _resolve_best_child_run(row, metric_name)

    experiment_id = _safe_str(row.get("experiment_id"))
    run_info = None
    if candidate_run_id != _safe_str(row.get("run_id")):
        try:
            child_run = mlflow.get_run(candidate_run_id)
        except Exception:
            child_run = None
        if child_run is not None:
            experiment_id = child_run.info.experiment_id
            candidate_artifact_uri = _safe_str(child_run.info.artifact_uri)
        else:
            candidate_artifact_uri = _safe_str(row.get("artifact_uri"))
    else:
        candidate_artifact_uri = _safe_str(row.get("artifact_uri"))

    if not candidate_artifact_uri:
        try:
            run_info = mlflow.get_run(candidate_run_id)
        except Exception:
            run_info = None
        if run_info is not None:
            candidate_artifact_uri = _safe_str(run_info.info.artifact_uri)
            if experiment_id is None:
                experiment_id = run_info.info.experiment_id

    experiment_name = _resolve_experiment_name(experiment_id)

    current_model = _get_latest_model(ml_client, settings.prod_model_name)
    should_promote, reason, previous_metric, previous_run_id = _should_promote(
        candidate_metric=candidate_metric,
        compare_metric=metric_name,
        maximize=maximize,
        current_model=current_model,
        candidate_run_id=candidate_run_id,
    )

    model_source = None
    model_source_kind = None
    promoted_model = None
    if should_promote:
        model_source, model_source_kind = _resolve_model_source(
            ml_client,
            candidate_run_id,
            candidate_artifact_uri,
            parent_run_id=parent_run_id,
        )
        if model_source is None:
            raise ValueError(f"Unable to resolve model artifacts for run {candidate_run_id}")

        if not dry_run:
            tags = {
                "compare_metric": metric_name,
                "compare_metric_value": str(candidate_metric),
                "source_run_id": candidate_run_id,
                "source_experiment": experiment_name or "",
                "source_experiment_id": experiment_id or "",
                "source_artifact_uri": candidate_artifact_uri or "",
                "source_parent_run_id": parent_run_id or "",
                "model_source_kind": model_source_kind or "",
            }
            promoted_model = ml_client.models.create_or_update(
                Model(
                    name=settings.prod_model_name,
                    path=str(model_source),
                    type=AssetTypes.CUSTOM_MODEL,
                    tags=tags,
                )
            )
            logger.info(
                "Promoted new production model",
                extra={"model": f"{promoted_model.name}:{promoted_model.version}"},
            )

    promotion_applied = should_promote and not dry_run
    production_version = (
        str(getattr(promoted_model, "version", None))
        if promotion_applied
        else (_safe_str(getattr(current_model, "version", None)) if current_model else None)
    )

    return PromotionResult(
        promoted=promotion_applied,
        promotion_reason=reason,
        compare_metric=metric_name,
        metric_column=metric_column,
        maximize_metric=maximize,
        candidate_run_id=candidate_run_id,
        candidate_experiment=experiment_name,
        candidate_metric_value=candidate_metric,
        candidate_artifact_uri=candidate_artifact_uri,
        candidate_parent_run_id=parent_run_id,
        production_model_name=settings.prod_model_name,
        production_model_version=production_version,
        previous_model_version=_safe_str(getattr(current_model, "version", None)) if current_model else None,
        previous_model_metric=previous_metric,
        previous_model_run_id=previous_run_id,
        model_source=str(model_source) if model_source else None,
        dry_run=dry_run,
    )


__all__ = ["PromotionResult", "promote_best_model"]


if __name__ == "__main__":
    result = promote_best_model()
    print(json.dumps(result.as_dict(), indent=2))
