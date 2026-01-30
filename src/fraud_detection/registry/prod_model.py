"""Utilities to select and register the best model for production."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import (
    Settings,
    build_idempotency_key,
    get_git_sha,
    get_settings,
)
from fraud_detection.utils.logging import get_logger
from fraud_detection.utils.metric_keys import (
    ALIAS_TO_CANONICAL,
    CANONICAL_METRIC_KEYS,
    CANONICAL_METRICS_BARE,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class BestRun:
    experiment_name: str
    run_id: str
    metric_name: str
    metric_value: float


@dataclass(frozen=True)
class PromotionResult:
    metric: str
    metric_direction: str
    experiments: list[str]
    best_run: BestRun
    prod_metric: float | None
    prod_model_name: str | None
    prod_model_version: str | None
    prod_run_id: str | None
    decision: str
    promote: bool
    model_name: str | None
    model_version: str | None
    model_id: str | None
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "metric_direction": self.metric_direction,
            "experiments": list(self.experiments),
            "best_run_id": self.best_run.run_id,
            "best_experiment_name": self.best_run.experiment_name,
            "best_metric_name": self.best_run.metric_name,
            "best_metric_value": self.best_run.metric_value,
            "prod_metric_value": self.prod_metric,
            "prod_model_name": self.prod_model_name,
            "prod_model_version": self.prod_model_version,
            "prod_run_id": self.prod_run_id,
            "decision": self.decision,
            "promote": self.promote,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "dry_run": self.dry_run,
        }


def _normalize_metric(metric: str) -> tuple[str, str]:
    raw = (metric or "").strip()
    if not raw:
        raise ValueError("Metric name cannot be empty.")

    if raw.startswith("metrics."):
        bare = raw[len("metrics.") :]
    else:
        bare = raw

    alias_key = bare.lower()
    canonical_full = ALIAS_TO_CANONICAL.get(alias_key)
    if canonical_full:
        bare = canonical_full[len("metrics.") :]

    full = f"metrics.{bare}"
    if full not in CANONICAL_METRIC_KEYS:
        allowed = ", ".join(CANONICAL_METRICS_BARE)
        raise ValueError(f"Metric '{metric}' is not a recognized metric. Choose one of: {allowed}")
    return bare, full


def _normalize_experiments(experiments: Sequence[str] | str | None, *, settings: Settings) -> list[str]:
    if experiments is None:
        candidates = [settings.custom_train_exp, settings.automl_train_exp]
    elif isinstance(experiments, str):
        candidates = [part.strip() for part in experiments.split(",")]
    else:
        candidates = [str(part).strip() for part in experiments]

    resolved: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        resolved.append(name)

    if not resolved:
        raise ValueError("No experiment names were provided.")
    return resolved


def _metric_direction(metric: str) -> str:
    metric_lower = (metric or "").lower()
    if any(key in metric_lower for key in ("loss", "error", "rmse", "mae")):
        return "min"
    return "max"


def _resolve_thresholds(settings: Settings) -> tuple[float, float, float | None]:
    epsilon = float(getattr(settings, "promotion_metric_epsilon", 0.0) or 0.0)
    delta = float(getattr(settings, "promotion_metric_delta", 0.0) or 0.0)
    min_threshold = getattr(settings, "promotion_min_metric", None)
    if min_threshold is not None:
        min_threshold = float(min_threshold)
    return epsilon, delta, min_threshold


def _is_same(best: float, current: float, *, epsilon: float, delta: float) -> bool:
    tolerance = max(float(epsilon), float(delta))
    if tolerance <= 0:
        return best == current
    return math.isclose(best, current, rel_tol=0.0, abs_tol=tolerance)


def _is_better(best: float, current: float, *, direction: str, epsilon: float, delta: float) -> bool:
    required_delta = max(float(epsilon), float(delta))
    if direction == "min":
        return best < (current - required_delta)
    return best > (current + required_delta)


def _meets_min_threshold(metric_value: float, *, direction: str, min_threshold: float | None) -> bool:
    if min_threshold is None:
        return True
    if direction == "min":
        return metric_value <= min_threshold
    return metric_value >= min_threshold


def _parse_version(version: str | None) -> int:
    try:
        return int(version or 0)
    except (TypeError, ValueError):
        return 0


def _latest_model_asset(ml_client: MLClient, name: str) -> Model | None:
    """Return latest AzureML model asset (highest numeric version) or None."""
    try:
        models = list(ml_client.models.list(name=name))
    except HttpResponseError as exc:
        logger.debug("Failed listing models for name=%s: %s", name, exc)
        return None

    if not models:
        return None

    return max(models, key=lambda m: _parse_version(getattr(m, "version", None)))


def _extract_run_id_from_model(model: Model) -> str | None:
    tags = getattr(model, "tags", None) or {}
    for key in ("run_id", "mlflow_run_id", "best_run_id"):
        value = tags.get(key)
        if value:
            return str(value)
    return None


def job_artifact_uri(job_id: str, artifact_rel_path: str) -> str:
    """Build a job artifact URI for model registration."""
    ap = artifact_rel_path.strip().strip("/")
    # NOTE: This is the correct job-artifact URI pattern
    return f"azureml://jobs/{job_id}/outputs/artifacts/paths/{ap}"


def _default_model_name(run_id: str) -> str:
    return f"model_{run_id}"


def _shadow_model_name(run_id: str) -> str:
    """Name for non-production registrations (keeps prod model name stable)."""
    return f"model_{run_id}"


def _get_run_metric(run_id: str, metric: str) -> float:
    """Read a metric from an MLflow run by run_id."""
    metric_key, _ = _normalize_metric(metric)
    run = mlflow.get_run(run_id)
    if metric_key not in run.data.metrics:
        raise KeyError(f"Metric '{metric_key}' not found in run {run_id}.")
    return float(run.data.metrics[metric_key])


def _get_run_tag(run_id: str, tag_name: str) -> str | None:
    try:
        run = mlflow.get_run(run_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed reading run tag", extra={"run_id": run_id, "tag": tag_name, "error": str(exc)})
        return None
    return run.data.tags.get(tag_name)


def _dataset_version_for_run(run_id: str) -> str:
    for key in ("dataset_version", "train_dataset_version", "data_version"):
        value = _get_run_tag(run_id, key)
        if value:
            return value
    return "unknown"


def mlflow_tracking_uri(ml_client: MLClient) -> str | None:
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        mlflow.set_tracking_uri(env_uri)
        return env_uri

    workspace = None
    try:
        workspace = ml_client.workspaces.get(name=ml_client.workspace_name)
    except TypeError:
        try:
            workspace = ml_client.workspaces.get(ml_client.workspace_name)
        except Exception:
            workspace = None
    except Exception:
        workspace = None

    if workspace is None:
        try:
            workspace = ml_client.workspaces.get()
        except Exception as exc:
            logger.warning("Failed to resolve MLflow tracking URI", extra={"error": str(exc)})
            return None

    uri = getattr(workspace, "mlflow_tracking_uri", None)
    if not uri:
        logger.warning("Workspace did not provide mlflow_tracking_uri")
        return None

    mlflow.set_tracking_uri(uri)
    return uri


def best_run_by_metric(
    *,
    metric: str,
    direction: str | None = None,
    view_type: ViewType = ViewType.ACTIVE_ONLY,
    status: str | None = "FINISHED",
    experiments: Sequence[str] | str | None = None,
    settings: Settings | None = None,
) -> BestRun:
    """Find the best run (by a metric) across one or more experiments."""
    resolved_settings = settings or get_settings()
    resolved_experiments = _normalize_experiments(experiments, settings=resolved_settings)

    metric_name, metric_col = _normalize_metric(metric)
    resolved_direction = (direction or _metric_direction(metric_name)).lower()
    if resolved_direction not in {"max", "min"}:
        raise ValueError("direction must be 'max' or 'min'")

    df = mlflow.search_runs(
        experiment_names=resolved_experiments,
        run_view_type=view_type,
    )

    if df is None or df.empty:
        raise RuntimeError("No runs found (empty MLflow search result).")

    if metric_col not in df.columns:
        raise RuntimeError(f"Metric column not found in MLflow results: {metric_col}")

    if status:
        if "status" in df.columns:
            df = df[df["status"] == status]
        else:
            logger.warning("MLflow search results missing status column; skipping status filter.")

    df = df.dropna(subset=[metric_col]).sort_values(by=[metric_col], ascending=(resolved_direction == "min"))

    if df.empty:
        raise RuntimeError(f"No runs contained a non-null value for metric: {metric_col}")

    best = df.iloc[0]
    run_id = str(best["run_id"])
    exp_name = str(best.get("experiment_name") or "")

    if not exp_name:
        exp_id = best.get("experiment_id")
        if exp_id is not None:
            try:
                exp = mlflow.get_experiment(exp_id)
            except Exception:
                exp = None
            if exp is not None:
                exp_name = str(getattr(exp, "name", "") or "")

    value = float(best[metric_col])

    logger.info(
        "Selected best run",
        extra={
            "run_id": run_id,
            "experiment_name": exp_name,
            "metric": metric_name,
            "value": value,
            "direction": resolved_direction,
            "sorted_by": metric_col,
        },
    )
    return BestRun(
        experiment_name=exp_name,
        run_id=run_id,
        metric_name=metric_name,
        metric_value=value,
    )


def register_model_from_run(
    ml_client: MLClient,
    *,
    model_name: str,
    run_id: str,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    artifact_path: str | None = None,
) -> Model:
    """Register a model from an MLflow run.

    AutoML usually logs the MLflow model under: outputs/mlflow-model.
    This function tries common paths and falls back to custom model artifacts.
    """
    candidates: list[tuple[str, str]] = []
    if artifact_path:
        candidates.append((artifact_path, AssetTypes.MLFLOW_MODEL))

    candidates += [
        ("outputs/mlflow-model", AssetTypes.MLFLOW_MODEL),
        ("outputs/mlflow_model", AssetTypes.MLFLOW_MODEL),
        ("outputs/model", AssetTypes.MLFLOW_MODEL),
        ("model", AssetTypes.MLFLOW_MODEL),
        ("mlflow-model", AssetTypes.MLFLOW_MODEL),
        ("mlflow_model", AssetTypes.MLFLOW_MODEL),
        ("exported_model", AssetTypes.CUSTOM_MODEL),
    ]

    last_err: Exception | None = None

    for ap, model_type in candidates:
        try:
            registered_model = Model(
                path=job_artifact_uri(run_id, ap),
                name=model_name or _default_model_name(run_id),
                description=description,
                type=model_type,
                tags=tags or {},
            )
            return ml_client.models.create_or_update(registered_model)
        except HttpResponseError as exc:
            last_err = exc
            continue

    raise last_err if last_err else RuntimeError("Model registration failed for unknown reasons.")


def current_prod_model_metric(
    ml_client: MLClient,
    metric: str,
    *,
    settings: Settings | None = None,
) -> tuple[float | None, Model | None, str | None]:
    """Return (metric value, model asset, run_id) for the latest production model version."""
    cfg = settings or get_settings()
    prod_model_name = cfg.prod_model_name

    latest = _latest_model_asset(ml_client, prod_model_name)
    if latest is None:
        logger.info("No production model registered yet (name=%s).", prod_model_name)
        return None, None, None

    mlflow_tracking_uri(ml_client)

    run_id = _extract_run_id_from_model(latest)
    if not run_id:
        try:
            client = MlflowClient()
            mv = client.get_model_version(name=latest.name, version=str(latest.version))
            run_id = getattr(mv, "run_id", None)
        except Exception as exc:
            logger.debug("Failed resolving run_id via MLflow registry", extra={"error": str(exc)})
            run_id = None

    if not run_id:
        logger.warning(
            "Latest production model has no run_id (name=%s version=%s). Cannot fetch metrics.",
            latest.name,
            latest.version,
        )
        return None, latest, None

    try:
        return _get_run_metric(run_id, metric), latest, run_id
    except KeyError as exc:
        logger.warning("%s", exc)
        return None, latest, run_id


def register_prod_model(
    metric: str,
    *,
    ml_client: MLClient | None = None,
    settings: Settings | None = None,
    experiments: Sequence[str] | str | None = None,
    dry_run: bool = False,
) -> PromotionResult:
    """Compare best experiment run vs current production model and register accordingly."""
    cfg = settings or get_settings()
    ml_client = ml_client or get_ml_client(settings=cfg)

    mlflow_tracking_uri(ml_client)

    resolved_experiments = _normalize_experiments(experiments, settings=cfg)
    best = best_run_by_metric(metric=metric, experiments=resolved_experiments, settings=cfg)

    prod_metric, prod_model, prod_run_id = current_prod_model_metric(ml_client, metric, settings=cfg)

    direction = _metric_direction(best.metric_name)
    epsilon, delta, min_threshold = _resolve_thresholds(cfg)
    meets_min = _meets_min_threshold(best.metric_value, direction=direction, min_threshold=min_threshold)

    decision = "stage"
    should_promote = False
    reason = "worse_metric"

    if prod_metric is None:
        should_promote = meets_min
        decision = "promote" if should_promote else "stage"
        reason = "no_prod_model" if should_promote else "below_min_threshold"
    else:
        if _is_same(best.metric_value, prod_metric, epsilon=epsilon, delta=delta):
            decision = "skip"
            reason = "equal_metric"
            should_promote = False
        else:
            should_promote = meets_min and _is_better(
                best.metric_value,
                prod_metric,
                direction=direction,
                epsilon=epsilon,
                delta=delta,
            )
            if should_promote:
                decision = "promote"
                reason = "better_metric"
            else:
                decision = "stage"
                reason = "below_min_threshold" if not meets_min else "worse_metric"

    prod_model_name = getattr(prod_model, "name", None) if prod_model is not None else None
    prod_model_version = str(getattr(prod_model, "version", "") or "") if prod_model is not None else None

    if decision == "skip":
        model_name = None
    elif should_promote:
        model_name = cfg.prod_model_name
    else:
        model_name = _shadow_model_name(best.run_id)

    logger.info(
        "Model promotion decision",
        extra={
            "decision": decision,
            "reason": reason,
            "metric": best.metric_name,
            "direction": direction,
            "best_metric": best.metric_value,
            "prod_metric": prod_metric,
            "model_name": model_name,
            "run_id": best.run_id,
        },
    )

    model_version = None
    model_id = None

    if decision != "skip" and not dry_run:
        model_source = "automl" if best.experiment_name == cfg.automl_train_exp else "custom"
        idempotency_key = build_idempotency_key(best.experiment_name or "", best.metric_name, best.run_id)
        stage = "production" if should_promote else "staging"
        alias = "production" if should_promote else "staging"
        tags = {
            "experiment_name": best.experiment_name,
            "selected_metric": best.metric_name,
            "metric_name": best.metric_name,
            "metric_value": str(best.metric_value),
            "best_run_id": best.run_id,
            "run_id": best.run_id,
            "git_sha": get_git_sha(short=False),
            "idempotency_key": idempotency_key,
            "promotion": str(should_promote).lower(),
            "promotion_decision": decision,
            "promotion_reason": reason,
            "model_source": model_source,
            "dataset_version": _dataset_version_for_run(best.run_id),
            "stage": stage,
            "alias": alias,
        }

        registered = register_model_from_run(
            ml_client,
            model_name=model_name or _default_model_name(best.run_id),
            run_id=best.run_id,
            description="Classification model registered by metric comparison",
            tags=tags,
        )
        model_version = str(getattr(registered, "version", "") or "") or None
        model_id = getattr(registered, "id", None) or getattr(registered, "model_id", None)

    return PromotionResult(
        metric=best.metric_name,
        metric_direction=direction,
        experiments=resolved_experiments,
        best_run=best,
        prod_metric=prod_metric,
        prod_model_name=prod_model_name,
        prod_model_version=prod_model_version,
        prod_run_id=prod_run_id,
        decision=decision,
        promote=should_promote,
        model_name=model_name,
        model_version=model_version,
        model_id=model_id,
        dry_run=dry_run,
    )


__all__ = [
    "BestRun",
    "PromotionResult",
    "best_run_by_metric",
    "current_prod_model_metric",
    "mlflow_tracking_uri",
    "register_model_from_run",
    "register_prod_model",
]
