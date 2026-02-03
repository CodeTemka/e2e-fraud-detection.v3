"""Submit the LightGBM training job to Azure ML.

Sweep ranges are tuned to recent best-performing runs to avoid wasting trials
on clearly underperforming regions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml.sweep import (
    Choice,
    MedianStoppingPolicy,
    RandomSamplingAlgorithm,
    SamplingAlgorithm,
    Uniform,
)
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import get_ml_client, resolve_azure_env_vars
from fraud_detection.config import (
    ROOT_DIR,
    Settings,
    build_idempotency_key,
    build_job_name,
    get_settings,
)
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


SUPPORTED_LGBM_METRICS = {"average_precision_score_macro", "AUC_macro"}

METRIC_ALIASES = {
    "metrics.average_precision_score_macro": "average_precision_score_macro",
    "average_precision_score_macro": "average_precision_score_macro",
    "average_precision_score": "average_precision_score_macro",
    "average_precision": "average_precision_score_macro",
    "metrics.auc_macro": "AUC_macro",
    "auc_macro": "AUC_macro",
}


def _metric_check(metric: str) -> str:
    metric_str = (metric or "").strip().replace("-", "_")
    metric_str = METRIC_ALIASES.get(metric_str.lower(), metric_str)
    if metric_str not in SUPPORTED_LGBM_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_LGBM_METRICS))
        raise ValueError(f"Unsupported metric '{metric_str}'. Choose one of: {allowed}")
    return metric_str


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


@dataclass
class LGBMSweepConfig:
    experiment_name: str
    training_data: str

    compute: str | None
    environment_name: str = "lgbm-env"
    environment_version: str | None = None
    environment_file: Path = field(
        default_factory=lambda: ROOT_DIR / "src" / "fraud_detection" / "training" / "lgbm" / "lgbm_env.yaml"
    )
    environment_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"

    label_column: str = "Class"
    random_state: int = 42
    val_size: float = 0.2
    primary_metric: str = "average_precision_score_macro"

    max_total_trials: int = 300
    max_concurrent_trials: int = 3
    timeout_minutes: int = 300
    sampling_algorithm: SamplingAlgorithm = RandomSamplingAlgorithm(seed=999)

    early_stopping_delay: int = 5
    early_stopping_interval: int = 2

    job_name: str | None = None
    idempotency_key: str | None = None
    run_suffix: str | None = None
    force_new_run: bool = False


def lgbm_sweep_job_builder(
    *,
    training_data: str,
    metric: str = "average_precision_score_macro",
    val_size: float = 0.2,
    compute: str | None = None,
    run_suffix: str | None = None,
    force_new_run: bool = False,
    settings: Settings | None = None,
) -> LGBMSweepConfig:
    resolved_metric = _metric_check(metric)
    resolved_settings = settings or get_settings()
    compute_target = compute or resolved_settings.training_compute_cluster_name
    env_file = ROOT_DIR / "src" / "fraud_detection" / "training" / "lgbm" / "lgbm_env.yaml"
    env_hash = _env_hash_from_file(env_file)
    idempotency_key = build_idempotency_key(resolved_settings.custom_train_exp, resolved_metric, env_hash)
    suffix = (run_suffix or "").strip()
    job_name = build_job_name("lgbm-sweep", idempotency_key)
    if suffix:
        job_name = f"{job_name}-{suffix}"
    if force_new_run:
        job_name = None
    return LGBMSweepConfig(
        experiment_name=resolved_settings.custom_train_exp,
        training_data=training_data,
        compute=compute_target,
        environment_file=env_file,
        label_column="Class",
        val_size=val_size,
        primary_metric=resolved_metric,
        job_name=job_name,
        idempotency_key=idempotency_key,
        run_suffix=suffix or None,
        force_new_run=force_new_run,
    )


def resolve_lgbm_environment(ml_client: MLClient, config: LGBMSweepConfig) -> str:
    """Resolve the LightGBM training environment."""
    if not config.environment_file.exists():
        raise FileNotFoundError(f"Environment file not found: {config.environment_file}")

    env_hash = _env_hash_from_file(config.environment_file)

    if config.environment_version:
        env_id = f"{config.environment_name}:{config.environment_version}"
        try:
            ml_client.environments.get(name=config.environment_name, version=config.environment_version)
        except ResourceNotFoundError:
            logger.warning("LightGBM environment not found; creating", extra={"environment": env_id})
        else:
            logger.info("Using existing LightGBM environment", extra={"environment": env_id})
            return env_id

        job_env = Environment(
            name=config.environment_name,
            version=config.environment_version,
            description="LightGBM training environment",
            conda_file=str(config.environment_file),
            image=config.environment_image,
            tags={"env_hash": env_hash},
        )
        job_env = ml_client.environments.create_or_update(job_env)
        env_id = f"{job_env.name}:{job_env.version}"
        logger.info("Created LightGBM environment", extra={"environment": env_id})
        return env_id

    matching_version: str | None = None
    versions: list[str] = []
    try:
        envs = list(ml_client.environments.list(name=config.environment_name))
    except ResourceNotFoundError:
        logger.info(
            "LightGBM environment not found; creating a new version",
            extra={"environment": config.environment_name},
        )
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
        logger.info("Using existing LightGBM environment", extra={"environment": env_id})
        return env_id

    next_version = str(max((_parse_env_version(v) for v in versions), default=0) + 1)
    job_env = Environment(
        name=config.environment_name,
        version=next_version,
        description="LightGBM training environment",
        conda_file=str(config.environment_file),
        image=config.environment_image,
        tags={"env_hash": env_hash},
    )
    job_env = ml_client.environments.create_or_update(job_env)
    env_id = f"{job_env.name}:{job_env.version}"
    logger.info("Created LightGBM environment", extra={"environment": env_id})
    return env_id


def create_lgbm_sweep_job(config: LGBMSweepConfig, *, environment: str) -> Any:
    """Create the LightGBM hyperparameter sweep job."""
    if not config.experiment_name:
        raise ValueError("Experiment name must be provided in the configuration.")

    azure_env_vars = resolve_azure_env_vars()

    base_job = command(
        code=str(ROOT_DIR / "src"),
        command=(
            "python -m fraud_detection.training.lgbm.train_lgbm "
            "--train_data ${{inputs.train_data}} "
            "--label_col ${{inputs.label_col}} "
            "--val_size ${{inputs.val_size}} "
            "--n_estimators ${{inputs.n_estimators}} "
            "--max_depth ${{inputs.max_depth}} "
            "--num_leaves ${{inputs.num_leaves}} "
            "--learning_rate ${{inputs.learning_rate}} "
            "--subsample ${{inputs.subsample}} "
            "--colsample_bytree ${{inputs.colsample_bytree}} "
            "--min_child_weight ${{inputs.min_child_weight}} "
            "--min_child_samples ${{inputs.min_child_samples}} "
            "--reg_alpha ${{inputs.reg_alpha}} "
            "--reg_lambda ${{inputs.reg_lambda}} "
            "--random_state ${{inputs.random_state}} "
            "--output_dir ${{outputs.model_output}}"
        ),
        inputs={
            "train_data": Input(type="string"),
            "label_col": Input(type="string"),
            "val_size": Input(type="number"),
            "n_estimators": Input(type="integer"),
            "max_depth": Input(type="integer"),
            "num_leaves": Input(type="integer"),
            "learning_rate": Input(type="number"),
            "subsample": Input(type="number"),
            "colsample_bytree": Input(type="number"),
            "min_child_weight": Input(type="number"),
            "min_child_samples": Input(type="integer"),
            "reg_alpha": Input(type="number"),
            "reg_lambda": Input(type="number"),
            "random_state": Input(type="integer"),
        },
        outputs={"model_output": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
        environment_variables=azure_env_vars,
        display_name="lgbm-train",
    )

    # Focused search space (avoid known poor extremes)
    job_for_sweep = base_job(
        train_data=config.training_data,
        label_col=config.label_column,
        val_size=config.val_size,
        learning_rate=Uniform(min_value=0.5, max_value=1.5),
        max_depth=Choice(values=[6, 7, 8, 9, 10]),
        num_leaves=Choice(values=[31, 63]),
        n_estimators=Choice(values=[800, 1000, 1200, 1400, 1600]),
        subsample=Uniform(min_value=0.6, max_value=0.85),
        colsample_bytree=Uniform(min_value=0.6, max_value=0.85),
        min_child_weight=Uniform(min_value=5.0, max_value=50.0),
        min_child_samples=Choice(values=[50, 75, 100, 150]),
        reg_alpha=Uniform(min_value=1.0, max_value=100.0),
        reg_lambda=Uniform(min_value=1e3, max_value=5e4),
        random_state=config.random_state,
    )

    sweep_job = job_for_sweep.sweep(
        compute=config.compute,
        sampling_algorithm=config.sampling_algorithm,
        primary_metric=config.primary_metric,
        goal="Maximize",
    )

    sweep_job.set_limits(
        max_total_trials=config.max_total_trials,
        max_concurrent_trials=config.max_concurrent_trials,
        timeout=config.timeout_minutes * 60,
    )
    sweep_job.early_termination = MedianStoppingPolicy(
        delay_evaluation=config.early_stopping_delay,
        evaluation_interval=config.early_stopping_interval,
    )

    sweep_job.display_name = "lgbm-sweep"
    sweep_job.experiment_name = config.experiment_name
    if config.job_name:
        sweep_job.name = config.job_name
    return sweep_job


def submit_lgbm_sweep_job(ml_client: MLClient, config: LGBMSweepConfig) -> str:
    if config.job_name:
        try:
            existing = ml_client.jobs.get(config.job_name)
        except ResourceNotFoundError:
            existing = None
        if existing:
            status = getattr(existing, "status", None)
            normalized = str(status or "").lower()
            if normalized in {"failed", "canceled", "cancelled", "error"}:
                logger.warning(
                    "Existing LightGBM sweep job failed; submitting a new run",
                    extra={"job_name": existing.name, "status": status},
                )
                config.job_name = None
            else:
                logger.info("Existing LightGBM sweep job reused", extra={"job_name": existing.name, "status": status})
                return existing.name

    environment = resolve_lgbm_environment(ml_client, config)
    sweep_job = create_lgbm_sweep_job(config, environment=environment)
    returned_job = ml_client.jobs.create_or_update(sweep_job)
    logger.info("Submitting LightGBM sweep job", extra={"job_name": returned_job.name})
    return returned_job.name


def main() -> None:
    settings = get_settings()
    ml_client = get_ml_client(settings=settings)

    training_data = settings.registered_train
    config = lgbm_sweep_job_builder(
        training_data=training_data,
        settings=settings,
    )
    job_name = submit_lgbm_sweep_job(ml_client, config)
    logger.info("LightGBM sweep job submitted", extra={"job_name": job_name})


__all__ = [
    "LGBMSweepConfig",
    "lgbm_sweep_job_builder",
    "resolve_lgbm_environment",
    "create_lgbm_sweep_job",
    "submit_lgbm_sweep_job",
]


if __name__ == "__main__":
    main()
