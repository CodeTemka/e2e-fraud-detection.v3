"""Submit endpoint evaluation + drift monitoring jobs to Azure ML."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import resolve_azure_env_vars
from fraud_detection.config import ROOT_DIR, Settings, build_idempotency_key, build_job_name, get_settings
from fraud_detection.utils.compute import ensure_pipeline_compute
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def _env_hash_from_file(env_file: Path) -> str:
    if not env_file.exists():
        return "missing"
    digest = hashlib.sha1(env_file.read_bytes()).hexdigest()
    return digest[:8]


def _parse_env_version(version: str | None) -> int:
    try:
        return int(version or 0)
    except (TypeError, ValueError):
        return 0


def _resolve_env_name(settings: Settings) -> str:
    return str(getattr(settings, "monitor_env_name", None) or "fraud-monitor-env")


@dataclass
class MonitorJobConfig:
    experiment_name: str
    test_data: str
    reference_data: str
    endpoint_name: str
    deployment_name: str | None
    scoring_uri: str | None
    auth_mode: str
    endpoint_key: str | None
    label_column: str
    batch_size: int
    max_retries: int
    request_timeout: float
    max_alerts: int | None
    alert_rate: float | None
    psi_bins: int
    psi_threshold: float
    ks_threshold: float
    mode: str
    compute: str | None

    environment_name: str = field(default_factory=lambda: _resolve_env_name(get_settings()))
    environment_version: str | None = None
    environment_file: Path = field(
        default_factory=lambda: ROOT_DIR / "src" / "fraud_detection" / "monitoring" / "monitor_env.yaml"
    )
    environment_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"

    job_name: str | None = None
    idempotency_key: str | None = None


def _resolve_data_asset_uri(
    ml_client: MLClient,
    *,
    data_ref: str,
    data_label: str | None,
) -> tuple[str, str]:
    """Resolve an Azure ML data asset reference to (uri, asset_type)."""
    if data_ref.startswith("azureml:"):
        parts = data_ref.split(":")
        if len(parts) >= 3:
            name = parts[1]
            version = parts[2]
            asset = ml_client.data.get(name=name, version=version)
            return f"azureml:{name}:{version}", str(asset.type)
        return data_ref, AssetTypes.URI_FOLDER

    if data_ref.startswith("azureml://"):
        return data_ref, AssetTypes.URI_FOLDER

    path = Path(data_ref)
    if path.exists():
        asset_type = AssetTypes.URI_FILE if path.is_file() else AssetTypes.URI_FOLDER
        return str(path), asset_type

    asset = ml_client.data.get(name=data_ref, label=data_label)
    uri = f"azureml:{asset.name}:{asset.version}"
    asset_type = str(getattr(asset, "type", None) or AssetTypes.URI_FOLDER)
    return uri, asset_type


def resolve_monitor_environment(ml_client: MLClient, config: MonitorJobConfig) -> str:
    if not config.environment_file.exists():
        raise FileNotFoundError(f"Environment file not found: {config.environment_file}")

    env_hash = _env_hash_from_file(config.environment_file)

    if config.environment_version:
        env_id = f"{config.environment_name}:{config.environment_version}"
        try:
            ml_client.environments.get(name=config.environment_name, version=config.environment_version)
            logger.info("Using existing monitoring environment", extra={"environment": env_id})
            return env_id
        except ResourceNotFoundError:
            logger.warning("Monitoring environment not found; creating", extra={"environment": env_id})

        job_env = Environment(
            name=config.environment_name,
            version=config.environment_version,
            description="Environment for endpoint evaluation + drift monitoring",
            conda_file=str(config.environment_file),
            image=config.environment_image,
            tags={"env_hash": env_hash},
        )
        job_env = ml_client.environments.create_or_update(job_env)
        env_id = f"{job_env.name}:{job_env.version}"
        logger.info("Created monitoring environment", extra={"environment": env_id})
        return env_id

    matching_version: str | None = None
    versions: list[str] = []
    try:
        envs = list(ml_client.environments.list(name=config.environment_name))
    except ResourceNotFoundError:
        envs = []

    for env in envs:
        version = str(getattr(env, "version", "") or "")
        versions.append(version)
        tags = getattr(env, "tags", None) or {}
        if tags.get("env_hash") == env_hash:
            if matching_version is None or _parse_env_version(version) > _parse_env_version(matching_version):
                matching_version = version

    if matching_version:
        env_id = f"{config.environment_name}:{matching_version}"
        logger.info("Using existing monitoring environment", extra={"environment": env_id})
        return env_id

    next_version = str(max((_parse_env_version(v) for v in versions), default=0) + 1)
    job_env = Environment(
        name=config.environment_name,
        version=next_version,
        description="Environment for endpoint evaluation + drift monitoring",
        conda_file=str(config.environment_file),
        image=config.environment_image,
        tags={"env_hash": env_hash},
    )
    job_env = ml_client.environments.create_or_update(job_env)
    env_id = f"{job_env.name}:{job_env.version}"
    logger.info("Created monitoring environment", extra={"environment": env_id})
    return env_id


def create_monitor_job(config: MonitorJobConfig, *, ml_client: MLClient) -> Any:
    """Create a single command job that runs evaluation + drift in one MLflow run."""
    import shlex

    azure_env_vars = resolve_azure_env_vars(settings=get_settings())
    environment = resolve_monitor_environment(ml_client, config)

    test_uri, test_type = _resolve_data_asset_uri(
        ml_client,
        data_ref=config.test_data,
        data_label=get_settings().registered_dataset_label,
    )
    ref_uri, ref_type = _resolve_data_asset_uri(
        ml_client,
        data_ref=config.reference_data,
        data_label=get_settings().registered_dataset_label,
    )

    cmd_parts = [
        "PYTHONPATH=. python -m fraud_detection.cli monitor",
        "--test-data ${{inputs.test_data}}",
        "--ref-data ${{inputs.reference_data}}",
        f"--endpoint-name {shlex.quote(config.endpoint_name)}",
        f"--auth-mode {shlex.quote(config.auth_mode or 'key')}",
        f"--label-column {shlex.quote(config.label_column)}",
        f"--batch-size {int(config.batch_size)}",
        f"--max-retries {int(config.max_retries)}",
        f"--request-timeout {float(config.request_timeout)}",
        f"--psi-bins {int(config.psi_bins)}",
        f"--psi-threshold {float(config.psi_threshold)}",
        f"--ks-threshold {float(config.ks_threshold)}",
        f"--mode {shlex.quote(config.mode)}",
        "--out ${{outputs.output_dir}}",
        "--local",
    ]

    if config.deployment_name:
        cmd_parts.append(f"--deployment-name {shlex.quote(config.deployment_name)}")
    if config.scoring_uri:
        cmd_parts.append(f"--scoring-uri {shlex.quote(config.scoring_uri)}")
    if config.endpoint_key:
        cmd_parts.append(f"--endpoint-key {shlex.quote(config.endpoint_key)}")
    if config.max_alerts is not None:
        cmd_parts.append(f"--max-alerts {int(config.max_alerts)}")
    if config.alert_rate is not None:
        cmd_parts.append(f"--alert-rate {float(config.alert_rate)}")

    job = command(
        code=str(ROOT_DIR / "src"),
        command=" ".join(cmd_parts),
        inputs={
            "test_data": Input(type=test_type, path=test_uri),
            "reference_data": Input(type=ref_type, path=ref_uri),
        },
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
        environment_variables=azure_env_vars,
        display_name="monitor-endpoint",
    )

    job.compute = config.compute
    job.experiment_name = config.experiment_name
    if config.job_name:
        job.name = config.job_name
    return job


def build_monitor_job_config(
    *,
    test_data: str,
    reference_data: str,
    endpoint_name: str,
    deployment_name: str | None,
    scoring_uri: str | None,
    auth_mode: str,
    endpoint_key: str | None,
    label_column: str,
    batch_size: int,
    max_retries: int,
    request_timeout: float,
    max_alerts: int | None,
    alert_rate: float | None,
    psi_bins: int,
    psi_threshold: float,
    ks_threshold: float,
    mode: str,
    compute: str | None,
    experiment_name: str | None = None,
    settings: Settings | None = None,
) -> MonitorJobConfig:
    resolved_settings = settings or get_settings()
    env_hash = _env_hash_from_file(ROOT_DIR / "src" / "fraud_detection" / "monitoring" / "monitor_env.yaml")
    key_parts: list[str] = [endpoint_name, mode, env_hash]
    run_id = (os.getenv("GITHUB_RUN_ID") or "").strip()
    run_attempt = (os.getenv("GITHUB_RUN_ATTEMPT") or "").strip()
    if run_id:
        key_parts.append(f"run-{run_id}")
    if run_attempt:
        key_parts.append(f"attempt-{run_attempt}")
    idempotency_key = build_idempotency_key(*key_parts)
    job_name = build_job_name("monitor", idempotency_key)

    return MonitorJobConfig(
        experiment_name=experiment_name or "monitoring",
        test_data=test_data,
        reference_data=reference_data,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        scoring_uri=scoring_uri,
        auth_mode=auth_mode,
        endpoint_key=endpoint_key,
        label_column=label_column,
        batch_size=batch_size,
        max_retries=max_retries,
        request_timeout=request_timeout,
        max_alerts=max_alerts,
        alert_rate=alert_rate,
        psi_bins=psi_bins,
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold,
        mode=mode,
        compute=compute or resolved_settings.pipeline_compute_cluster_name,
        job_name=job_name,
        idempotency_key=idempotency_key,
    )


def submit_monitor_job(ml_client: MLClient, config: MonitorJobConfig, *, ensure_compute: bool = True) -> Any:
    if ensure_compute:
        ensure_pipeline_compute(ml_client)

    job = create_monitor_job(config, ml_client=ml_client)
    returned = ml_client.jobs.create_or_update(job)
    logger.info("Submitted monitoring job", extra={"job_name": returned.name})
    return returned


__all__ = ["MonitorJobConfig", "build_monitor_job_config", "create_monitor_job", "submit_monitor_job"]
