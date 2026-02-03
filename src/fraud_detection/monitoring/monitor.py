"""Orchestrate evaluation + drift monitoring in a single run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from azure.ai.ml import MLClient

from fraud_detection.monitoring.data_loading import load_dataframe
from fraud_detection.monitoring.drift import DriftConfig, DriftResult, run_drift_check
from fraud_detection.monitoring.endpoint_client import EndpointConfig
from fraud_detection.monitoring.evaluate_endpoint import EvaluationConfig, EvaluationResult, evaluate_endpoint
from fraud_detection.monitoring.mlflow_utils import ensure_mlflow_run
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class MonitorMode(str, Enum):
    evaluate = "evaluate"
    drift = "drift"
    monitor = "monitor"


@dataclass
class MonitorConfig:
    test_data: str | Path
    reference_data: str | Path | None
    output_dir: Path
    endpoint_config: EndpointConfig
    max_alerts: int | None = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    mode: MonitorMode = MonitorMode.monitor
    experiment_name: str | None = None
    run_name: str | None = None


@dataclass
class MonitorResult:
    evaluation: EvaluationResult | None
    drift: DriftResult | None


def run_monitoring(
    *,
    config: MonitorConfig,
    ml_client: MLClient | None = None,
    data_label: str | None = None,
) -> MonitorResult:
    """Run evaluation and/or drift monitoring with a shared MLflow run."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_result: EvaluationResult | None = None
    drift_result: DriftResult | None = None

    test_df = load_dataframe(
        config.test_data,
        ml_client=ml_client,
        data_label=data_label,
        sample_rows=config.evaluation.sample_rows,
    )

    with ensure_mlflow_run(experiment_name=config.experiment_name, run_name=config.run_name):
        if config.mode in {MonitorMode.evaluate, MonitorMode.monitor}:
            evaluation_result = evaluate_endpoint(
                test_data=config.test_data,
                endpoint_config=config.endpoint_config,
                output_dir=config.output_dir,
                config=config.evaluation,
                ml_client=ml_client,
                data_label=data_label,
                max_alerts=config.max_alerts,
                dataframe=test_df,
            )

        if config.mode in {MonitorMode.drift, MonitorMode.monitor}:
            if not config.reference_data:
                raise ValueError("reference_data is required for drift monitoring.")

            ref_df = load_dataframe(
                config.reference_data,
                ml_client=ml_client,
                data_label=data_label,
                sample_rows=config.drift.sample_rows,
            )
            drift_result = run_drift_check(
                reference_data=config.reference_data,
                current_data=config.test_data,
                output_dir=config.output_dir,
                config=config.drift,
                ml_client=ml_client,
                data_label=data_label,
                reference_df=ref_df,
                current_df=test_df,
            )

    return MonitorResult(evaluation=evaluation_result, drift=drift_result)


__all__ = ["MonitorConfig", "MonitorMode", "MonitorResult", "run_monitoring"]
