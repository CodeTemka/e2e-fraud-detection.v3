"""Deploy the production model to an Azure ML online endpoint."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from math import floor
from pathlib import Path
from typing import Any, Mapping

from azure.ai.ml import MLClient
from azure.ai.ml.entities import CodeConfiguration, Environment, ManagedOnlineDeployment, ManagedOnlineEndpoint, Model
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger
from fraud_detection.utils.versioning import resolve_next_environment_version

logger = get_logger(__name__)

DEFAULT_AUTH_MODE = "key"
DEFAULT_SCORING_ENV_NAME = "fraud-scoring-env"
DEFAULT_SCORING_ENV_FILE = Path(__file__).resolve().parent / "scoring_env.yaml"
DEFAULT_SCORING_ENV_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"


@dataclass(frozen=True)
class ResolvedModel:
    model: Model
    name: str
    version: str
    model_id: str | None
    tags: dict[str, str]
    path: str | None
    asset_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "id": self.model_id,
            "tags": self.tags,
            "path": self.path,
            "asset_type": self.asset_type,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }


@dataclass(frozen=True)
class ServeResult:
    model_name: str
    model_version: str
    endpoint_name: str
    deployment_name: str
    traffic: dict[str, int] | None
    scoring_env: str | None
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "endpoint_name": self.endpoint_name,
            "deployment_name": self.deployment_name,
            "traffic": self.traffic,
            "scoring_env": self.scoring_env,
            "action": self.action,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }


def _parse_version(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _version_sort_key(model: Model) -> tuple[int, tuple[int, ...] | str]:
    version_text = str(getattr(model, "version", "") or "")
    parsed = _parse_version(version_text)
    if parsed is not None:
        return (1, parsed)
    return (0, version_text.lower())


def _coerce_tags(tags: dict[str, Any] | None) -> dict[str, str]:
    if not tags:
        return {}
    return {str(key): str(value) for key, value in tags.items()}


def _build_resolved_model(model: Model) -> ResolvedModel:
    return ResolvedModel(
        model=model,
        name=str(getattr(model, "name", "")),
        version=str(getattr(model, "version", "")),
        model_id=getattr(model, "id", None),
        tags=_coerce_tags(getattr(model, "tags", None)),
        path=str(getattr(model, "path", None)) if getattr(model, "path", None) else None,
        asset_type=str(getattr(model, "type", None)) if getattr(model, "type", None) else None,
    )


def resolve_model(
    ml_client: MLClient,
    *,
    model_name: str,
    model_version: str | None = None,
) -> ResolvedModel:
    if not model_name or not str(model_name).strip():
        raise ValueError("model_name is required.")
    name = str(model_name).strip()

    if model_version:
        version = str(model_version).strip()
        if not version:
            raise ValueError("model_version cannot be empty.")
        model = ml_client.models.get(name=name, version=version)
        resolved = _build_resolved_model(model)
        logger.info(
            "Resolved model by version",
            extra={"model_name": resolved.name, "model_version": resolved.version},
        )
        return resolved

    models = list(ml_client.models.list(name=name))
    if not models:
        raise ResourceNotFoundError(f"No model found with name '{name}'.")

    model = max(models, key=_version_sort_key)
    resolved = _build_resolved_model(model)
    if not resolved.version:
        raise RuntimeError(f"Model '{resolved.name}' has no version.")

    logger.info(
        "Resolved latest model",
        extra={"model_name": resolved.name, "model_version": resolved.version},
    )
    return resolved


def _normalize_required(value: str | None, fallback: str, label: str) -> str:
    resolved = (value or "").strip() or fallback.strip()
    if not resolved:
        raise ValueError(f"{label} resolved to an empty value.")
    return resolved


def _validate_traffic_percentage(traffic_percentage: int | None) -> int | None:
    if traffic_percentage is None:
        return None
    traffic = int(traffic_percentage)
    if traffic < 1 or traffic > 100:
        raise ValueError("traffic_percentage must be between 1 and 100.")
    return traffic


def _merge_tags(base: dict[str, str], updates: dict[str, str | None]) -> dict[str, str]:
    merged = {str(key): str(value) for key, value in (base or {}).items()}
    for key, value in updates.items():
        if value is None:
            continue
        merged[str(key)] = str(value)
    return merged


def _resolve_scoring_env_name(settings: Settings) -> str:
    return str(getattr(settings, "scoring_env_name", None) or DEFAULT_SCORING_ENV_NAME)


def _resolve_scoring_env_file(settings: Settings) -> Path:
    raw = getattr(settings, "scoring_env_file", None) or DEFAULT_SCORING_ENV_FILE
    return Path(raw)


def resolve_scoring_environment(ml_client: MLClient, *, settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    env_name = _resolve_scoring_env_name(resolved_settings)
    env_version = getattr(resolved_settings, "scoring_env_version", None)
    env_path = _resolve_scoring_env_file(resolved_settings)

    if not env_path.exists():
        raise FileNotFoundError(f"Scoring environment file not found: {env_path}")

    if env_version:
        env_id = f"{env_name}:{env_version}"
        try:
            ml_client.environments.get(name=env_name, version=env_version)
            return env_id
        except ResourceNotFoundError:
            logger.warning("Scoring environment not found; creating", extra={"environment": env_id})

        env = Environment(
            name=env_name,
            description="Fraud detection scoring environment",
            conda_file=str(env_path),
            image=DEFAULT_SCORING_ENV_IMAGE,
            version=str(env_version),
        )
        created = ml_client.environments.create_or_update(env)
        logger.info(
            "Registered scoring environment",
            extra={"environment": f"{created.name}:{created.version}"},
        )
        return f"{created.name}:{created.version}"

    env_version = resolve_next_environment_version(ml_client, name=env_name)
    env = Environment(
        name=env_name,
        description="Fraud detection scoring environment",
        conda_file=str(env_path),
        image=DEFAULT_SCORING_ENV_IMAGE,
        version=env_version,
    )
    created = ml_client.environments.create_or_update(env)
    logger.info(
        "Registered scoring environment",
        extra={"environment": f"{created.name}:{created.version}"},
    )
    return f"{created.name}:{created.version}"


def prepare_code_directory(
    *,
    destination: Path,
    scoring_script: Path,
    scaler_dir: Path,
) -> Path:
    if not scoring_script.exists() or not scoring_script.is_file():
        raise FileNotFoundError(f"Scoring script not found: {scoring_script}")
    if not scaler_dir.exists() or not scaler_dir.is_dir():
        raise FileNotFoundError(f"Scaler directory not found: {scaler_dir}")

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scoring_script, destination / scoring_script.name)

    scalers_dest = destination / "scalers"
    shutil.copytree(scaler_dir, scalers_dest, dirs_exist_ok=True)
    return destination


def _normalize_instance_type(instance_type: str | None, settings: Settings) -> str:
    resolved = (instance_type or "").strip()
    return resolved if resolved else settings.deployment_instance_type


def _coerce_env_vars(env_vars: Mapping[str, str] | None) -> dict[str, str] | None:
    if not env_vars:
        return None
    return {str(key): str(value) for key, value in env_vars.items()}


def _coerce_tags_for_deployment(tags: Mapping[str, str] | None) -> dict[str, str]:
    resolved = {str(key): str(value) for key, value in (tags or {}).items()}
    resolved.setdefault("project", "fraud-detection")
    return resolved


def deploy_model(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str = "blue",
    model_name: str,
    model_version: str | None = None,
    instance_type: str | None = None,
    instance_count: int = 1,
    code_path: str | Path | None = None,
    scoring_script: str | Path | None = None,
    environment: str | Environment | None = None,
    environment_variables: Mapping[str, str] | None = None,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineDeployment:
    if instance_count < 1:
        raise ValueError("instance_count must be >= 1")

    settings = get_settings()
    resolved_instance_type = _normalize_instance_type(instance_type, settings)

    resolved_model = resolve_model(
        ml_client,
        model_name=model_name,
        model_version=model_version,
    )

    code_configuration = None
    if code_path and scoring_script:
        code_dir = Path(code_path)
        script_name = scoring_script.name if isinstance(scoring_script, Path) else str(scoring_script)
        if not code_dir.exists():
            raise FileNotFoundError(f"code_path not found: {code_dir}")
        if not (code_dir / script_name).exists():
            raise FileNotFoundError(f"scoring script not found in code_path: {script_name}")
        code_configuration = CodeConfiguration(code=str(code_dir), scoring_script=script_name)
    elif code_path or scoring_script:
        raise ValueError("Provide both code_path and scoring_script together.")

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=resolved_model.model,
        instance_type=resolved_instance_type,
        instance_count=instance_count,
        code_configuration=code_configuration,
        environment=environment,
        environment_variables=_coerce_env_vars(environment_variables),
        tags=_coerce_tags_for_deployment(tags),
    )

    logger.info(
        "Creating/updating deployment",
        extra={
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "model_name": resolved_model.name,
            "model_version": resolved_model.version,
            "instance_type": resolved_instance_type,
            "instance_count": instance_count,
            "scoring_script": scoring_script,
        },
    )

    return ml_client.online_deployments.begin_create_or_update(deployment).result()


def get_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    return ml_client.online_endpoints.get(endpoint_name)


def preflight_endpoint(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    description: str | None = None,
    auth_mode: str = DEFAULT_AUTH_MODE,
    tags: Mapping[str, str] | None = None,
) -> ManagedOnlineEndpoint:
    try:
        endpoint = get_endpoint(ml_client, endpoint_name)
    except ResourceNotFoundError as exc:
        raise RuntimeError(
            f"Endpoint '{endpoint_name}' does not exist. "
            "Create it first (CLI or Azure ML UI) before deploying."
        ) from exc

    state_raw = getattr(endpoint, "provisioning_state", None)
    state = str(state_raw).strip().lower() if state_raw is not None else None
    if state and state != "succeeded":
        raise RuntimeError(
            f"Endpoint '{endpoint_name}' is in provisioning_state='{state_raw}'. "
            "Wait for it to reach Succeeded, or delete/recreate the endpoint before deploying."
        )
    if not state:
        logger.info(
            "Endpoint provisioning_state unavailable; proceeding",
            extra={"endpoint_name": endpoint_name},
        )
    return endpoint


def update_traffic(ml_client: MLClient, *, endpoint_name: str, traffic: dict[str, int]) -> None:
    _validate_traffic(traffic)

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    previous = dict(endpoint.traffic or {})
    endpoint.traffic = traffic

    logger.info(
        "Updating endpoint traffic",
        extra={
            "endpoint_name": endpoint_name,
            "previous_traffic": previous,
            "traffic": traffic,
        },
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def set_initial_traffic(
    ml_client: MLClient,
    *,
    endpoint_name: str,
    deployment_name: str,
    low_traffic_percent: int = 10,
) -> dict[str, int]:
    if low_traffic_percent <= 0 or low_traffic_percent >= 100:
        raise ValueError("low_traffic_percent must be between 1 and 99")

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    current_traffic = dict(endpoint.traffic or {})
    traffic = _build_canary_traffic(
        current_traffic=current_traffic,
        deployment_name=deployment_name,
        deployment_percent=low_traffic_percent,
    )
    update_traffic(ml_client, endpoint_name=endpoint_name, traffic=traffic)
    return traffic


def _build_canary_traffic(
    *,
    current_traffic: Mapping[str, int],
    deployment_name: str,
    deployment_percent: int,
) -> dict[str, int]:
    if deployment_percent >= 100:
        return {deployment_name: 100}

    remaining = 100 - deployment_percent
    existing = {name: pct for name, pct in current_traffic.items() if name != deployment_name}
    if not existing:
        return {deployment_name: 100}

    total_existing = sum(existing.values())
    traffic = _allocate_remaining_traffic(
        existing=existing,
        remaining=remaining,
        total_existing=total_existing,
    )
    traffic[deployment_name] = deployment_percent
    return traffic


def _allocate_remaining_traffic(
    *,
    existing: Mapping[str, int],
    remaining: int,
    total_existing: int,
) -> dict[str, int]:
    if total_existing <= 0:
        base = remaining // len(existing)
        traffic = {name: base for name in existing}
        remainder = remaining - base * len(existing)
        for name in sorted(existing)[:remainder]:
            traffic[name] += 1
        return traffic

    allocations: list[tuple[str, int, float]] = []
    for name, pct in existing.items():
        raw = pct * remaining / total_existing
        base = floor(raw)
        allocations.append((name, base, raw - base))

    traffic = {name: base for name, base, _ in allocations}
    remainder = remaining - sum(base for _, base, _ in allocations)
    allocations.sort(key=lambda item: (-item[2], item[0]))
    for name, _, _ in allocations[:remainder]:
        traffic[name] += 1
    return traffic


def _validate_traffic(traffic: dict[str, int]) -> None:
    if not traffic:
        raise ValueError("traffic mapping is empty")

    for dep, pct in traffic.items():
        if not dep or not isinstance(dep, str):
            raise ValueError("traffic keys must be deployment names (non-empty strings)")
        if not isinstance(pct, int):
            raise ValueError(f"traffic percentage for '{dep}' must be an int")
        if pct < 0:
            raise ValueError(f"traffic percentage for '{dep}' must be >= 0")

    total = sum(traffic.values())
    if total != 100:
        raise ValueError(f"traffic must sum to 100, got {total}")


def serve_prod_model(
    ml_client: MLClient,
    *,
    prod_model_name: str | None = None,
    model_version: str | None = None,
    endpoint_name: str | None = None,
    deployment_name: str | None = None,
    scaler_dir: str | Path,
    max_alerts: int,
    traffic_percentage: int | None = 100,
    dry_run: bool = False,
    settings: Settings | None = None,
) -> ServeResult:
    cfg = settings or get_settings()

    resolved_model_name = _normalize_required(prod_model_name, cfg.prod_model_name, "prod_model_name")
    resolved_endpoint = _normalize_required(endpoint_name, cfg.endpoint_name, "endpoint_name")
    resolved_deployment = _normalize_required(deployment_name, cfg.deployment_name, "deployment_name")

    if max_alerts < 0:
        raise ValueError("max_alerts must be >= 0.")

    traffic = _validate_traffic_percentage(traffic_percentage)

    scaler_path = Path(scaler_dir).resolve()
    if not scaler_path.exists() or not scaler_path.is_dir():
        raise FileNotFoundError(f"scaler_dir must be an existing directory: {scaler_path}")

    resolved_model = resolve_model(
        ml_client,
        model_name=resolved_model_name,
        model_version=model_version,
    )

    if dry_run:
        logger.info(
            "Dry run enabled; skipping deployment",
            extra={
                "model_name": resolved_model.name,
                "model_version": resolved_model.version,
                "endpoint_name": resolved_endpoint,
                "deployment_name": resolved_deployment,
            },
        )
        return ServeResult(
            model_name=resolved_model.name,
            model_version=resolved_model.version,
            endpoint_name=resolved_endpoint,
            deployment_name=resolved_deployment,
            traffic=None,
            scoring_env=None,
            action="dry_run",
        )

    preflight_endpoint(
        ml_client,
        endpoint_name=resolved_endpoint,
    )

    scoring_env = resolve_scoring_environment(ml_client, settings=cfg)
    env_vars = {
        "MAX_ALERTS": str(max_alerts),
        "ALERT_CAP": str(max_alerts),
        "SCALER_DIR": "scalers",
    }

    scoring_script = Path(__file__).resolve().parent / "score.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        code_dir = prepare_code_directory(
            destination=Path(tmp_dir) / "code",
            scoring_script=scoring_script,
            scaler_dir=scaler_path,
        )

        tags = _merge_tags(
            {},
            {
                "source_model_name": resolved_model.name,
                "source_model_version": resolved_model.version,
                "source_model_id": resolved_model.model_id,
                "max_alerts": str(max_alerts),
            },
        )

        deploy_model(
            ml_client,
            endpoint_name=resolved_endpoint,
            deployment_name=resolved_deployment,
            model_name=resolved_model.name,
            model_version=resolved_model.version,
            instance_type=cfg.deployment_instance_type,
            instance_count=cfg.deployment_instance_count,
            code_path=code_dir,
            scoring_script=scoring_script.name,
            environment=scoring_env,
            environment_variables=env_vars,
            tags=tags,
        )

        traffic_map: dict[str, int] | None = None
        if traffic is not None:
            if traffic >= 100:
                traffic_map = {resolved_deployment: 100}
                update_traffic(ml_client, endpoint_name=resolved_endpoint, traffic=traffic_map)
            else:
                traffic_map = set_initial_traffic(
                    ml_client,
                    endpoint_name=resolved_endpoint,
                    deployment_name=resolved_deployment,
                    low_traffic_percent=traffic,
                )

        return ServeResult(
            model_name=resolved_model.name,
            model_version=resolved_model.version,
            endpoint_name=resolved_endpoint,
            deployment_name=resolved_deployment,
            traffic=traffic_map,
            scoring_env=scoring_env,
            action="deployed",
        )


__all__ = [
    "ResolvedModel",
    "ServeResult",
    "deploy_model",
    "resolve_model",
    "resolve_scoring_environment",
    "serve_prod_model",
    "prepare_code_directory",
]
