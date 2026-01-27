"""Submit the XGBoost training job to Azure ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
from azure.ai.ml.sweep import (
    Choice,
    LogUniform,
    MedianStoppingPolicy,
    RandomSamplingAlgorithm,
    SamplingAlgorithm,
    Uniform,
)
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import (
    ROOT_DIR,
    Settings,
    build_idempotency_key,
    build_job_name,
    get_settings,
)
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


SUPPORTED_XGB_METRICS = {"average_precision", "AUC_macro"}

METRIC_ALIASES = {
    "metrics.average_precision_score_macro": "average_precision",
    "average_precision_score_macro": "average_precision",
    "metrics.AUC_macro": "AUC_macro",
}

def _metric_check(metric: str) -> str:
    metric_str = (metric or "").strip().replace("-", "_")
    metric_str = METRIC_ALIASES.get(metric_str, metric_str)
    if metric_str not in SUPPORTED_XGB_METRICS:
        allowed = ", ".join(sorted(SUPPORTED_XGB_METRICS))
        raise ValueError(f"Unsupported metric '{metric_str}'. Choose one of: {allowed}")
    return metric_str


@dataclass
class XGBSweepConfig:
    experiment_name: str
    training_data: str
    test_data: str

    compute: str | None
    environment_name: str = "xgboost-env"
    environment_version: str = "1.0"
    environment_file: Path = field(
        default_factory=lambda: ROOT_DIR / "src" / "fraud_detection" / "training" / "xgb_env.yaml"
    )
    environment_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"

    label_column: str = "Class"
    primary_metric: str = "average_precision"

    max_total_trials: int = 30
    max_concurrent_trials: int = 2
    timeout_minutes: int = 180
    sampling_algorithm: SamplingAlgorithm = RandomSamplingAlgorithm(seed=999, rule="sobol", logbase="e")

    early_stopping_delay: int = 5
    early_stopping_interval: int = 2

    job_name: str | None = None
    idempotency_key: str | None = None


def xgb_sweep_job_builder(
    *,
    training_data: str,
    test_data: str,
    metric: str = "average_precision",
    compute: str | None = None,
    settings: Settings | None = None,
) -> XGBSweepConfig:
    resolved_metric = _metric_check(metric)
    resolved_settings = settings or get_settings()
    compute_target = compute or resolved_settings.training_compute_cluster_name
    idempotency_key = build_idempotency_key(resolved_settings.custom_train_exp, resolved_metric)
    return XGBSweepConfig(
        experiment_name=resolved_settings.custom_train_exp,
        training_data=training_data,
        test_data=test_data,
        compute=compute_target,
        label_column="Class",
        primary_metric=resolved_metric,
        job_name=build_job_name("xgb-sweep", idempotency_key),
        idempotency_key=idempotency_key,
    )


def resolve_xgb_environment(ml_client: MLClient, config: XGBSweepConfig) -> str:
    """Resolve the XGBoost training environment."""
    if config.environment_version:
        env_id = f"{config.environment_name}:{config.environment_version}"
        logger.info("Using existing XGBoost environment", extra={"environment": env_id})
        return env_id
    
    if not config.environment_file.exists():
        raise FileNotFoundError(f"Environment file not found: {config.environment_file}")
    
    job_env = Environment(
        name=config.environment_name,
        description="XGBoost training environment",
        conda_file=str(config.environment_file),
        image=config.environment_image,
    )
    job_env = ml_client.environments.create_or_update(job_env)
    env_id = f"{job_env.name}:{job_env.version}"
    logger.info("Created XGBoost environment", extra={"environment": env_id})
    return env_id


def create_xgb_sweep_job(config: XGBSweepConfig, *, environment: str) -> Any:
    """Create the XGBoost hyperparameter sweep job."""
    if not config.experiment_name:
        raise ValueError("Experiment name must be provided in the configuration.")
    
    base_job = command(
        code=str(ROOT_DIR / "src"),
        command=(
            "python fraud_detection.training.xgb.tmp_train.py "
                "--train_data ${{inputs.train_data}} "
                "--test_data ${{inputs.test_data}} "
                "--label_col ${{inputs.label_col}} "
                "--n_estimators ${{inputs.n_estimators}} "
                "--max_depth ${{inputs.max_depth}} "
                "--learning_rate ${{inputs.learning_rate}} "
                "--subsample ${{inputs.subsample}} "
                "--colsample_bytree ${{inputs.colsample_bytree}} "
                "--min_child_weight ${{inputs.min_child_weight}} "
                "--gamma ${{inputs.gamma}} "
                "--reg_lambda ${{inputs.reg_lambda}} "
                "--random_state ${{inputs.random_state}} "
                "--output_dir ${{outputs.model_output}}"
        ),
        inputs={
            "train_data": config.training_data,
            "test_data": config.test_data,
            "label_col": config.label_column,
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
        }
        outputs={"output_dir": Output(type=AssetTypes.URI_FOLDER)},
        environment=environment,
        compute=config.compute,
        experiment_name=config.experiment_name,
        display_name="xgb-train",
    )
        
    job_for_sweep = base_job(
        learning_rate=LogUniform(min_value=1e-3, max_value=0.2),
        max_depth=Choice(values=[3, 4, 5, 6, 7, 8]),
        n_estimators=Choice(values=[300, 600, 900, 1200]),
        subsample=Uniform(min_value=0.6, max_value=1.0),
        colsample_bytree=Uniform(min_value=0.6, max_value=1.0),
        min_child_weight=Choice(values=[1.0, 5.0, 10.0]),
        gamma=Uniform(min_value=0.0, max_value=5.0),
        reg_lambda=LogUniform(min_value=1e-3, max_value=10.0),
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

    sweep_job.display_name = "xgb-sweep"
    sweep_job.experiment_name = config.experiment_name
    if config.job_name:
        sweep_job.name = config.job_name
    return sweep_job


def submit_xgb_sweep_job(ml_client: MLClient, config: XGBSweepConfig) -> str:

    if config.job_name:
        try:
            existing = ml_client.jobs.get(config.job_name)
        except ResourceNotFoundError:
            existing = None
        if existing:
            status = getattr(existing, "status", None)
            logger.info("Existing XGBoost sweep job reused", extra={"job_name": existing.name, "status": status})
            return getattr(existing, "name")
        
    environment = resolve_xgb_environment(ml_client, config)
    sweep_job = create_xgb_sweep_job(config, environment=environment)
    returned_job = ml_client.jobs.create_or_update(sweep_job)
    logger.info("Submitting XGBoost sweep job", extra={"job_name": returned_job.name})
    return getattr(returned_job, "name")


def main() -> None:
    settings = get_settings()
    ml_client = get_ml_client(settings=settings)

    training_data = settings.registered_train
    test_data = settings.registered_test
    config = xgb_sweep_job_builder(
        training_data=training_data,
        test_data=test_data,
        settings=settings,
    )
    job_name = submit_xgb_sweep_job(ml_client, config)
    logger.info("XGBoost sweep job submitted", extra={"job_name": job_name})


__all__ = ["XGBSweepConfig", "xgb_sweep_job_builder", "resolve_xgb_environment", "create_xgb_sweep_job", "submit_xgb_sweep_job"]


if __name__ == "__main__":
    main()