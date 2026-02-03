"""Evaluate a deployed endpoint using registered test data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection.monitoring.data_loading import load_dataframe
from fraud_detection.monitoring.endpoint_client import EndpointConfig, invoke_endpoint_batches
from fraud_detection.monitoring.mlflow_utils import ensure_mlflow_run, log_artifact, log_metrics, write_json
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    label_column: str = "Class"
    id_columns: tuple[str, ...] = ("id", "transaction_id", "row_id")
    positive_label: int = 1
    sample_rows: int | None = None


@dataclass
class EvaluationResult:
    metrics: dict[str, float | None]
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    predictions: list[int]
    probabilities: list[float] | None
    threshold_used: float | None


def _coerce_label_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    return values.astype(int)


def _plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "metrics.accuracy": float(accuracy_score(y_true, y_pred)),
        "metrics.precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "metrics.recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "metrics.f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "metrics.alert_rate": float(np.mean(y_pred)) if y_pred.size else None,
        "metrics.alert_count": float(np.sum(y_pred)) if y_pred.size else None,
    }

    if probabilities is not None and probabilities.size == y_true.size:
        try:
            metrics["metrics.roc_auc"] = float(roc_auc_score(y_true, probabilities))
        except ValueError:
            metrics["metrics.roc_auc"] = None
        try:
            metrics["metrics.pr_auc"] = float(average_precision_score(y_true, probabilities))
        except ValueError:
            metrics["metrics.pr_auc"] = None
    else:
        metrics["metrics.roc_auc"] = None
        metrics["metrics.pr_auc"] = None

    return metrics


def evaluate_endpoint(
    *,
    test_data: str | Path,
    endpoint_config: EndpointConfig,
    output_dir: Path,
    config: EvaluationConfig | None = None,
    ml_client: MLClient | None = None,
    data_label: str | None = None,
    max_alerts: int | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    dataframe: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Evaluate a deployed endpoint and log metrics/artifacts to MLflow."""
    resolved_config = config or EvaluationConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = dataframe if dataframe is not None else load_dataframe(
        test_data,
        ml_client=ml_client,
        data_label=data_label,
        sample_rows=resolved_config.sample_rows,
    )

    if resolved_config.label_column not in df.columns:
        raise ValueError(f"Label column '{resolved_config.label_column}' not found in test data.")

    y_true = _coerce_label_series(df[resolved_config.label_column]).to_numpy()
    feature_df = df.drop(columns=[resolved_config.label_column], errors="ignore")
    feature_df = feature_df.drop(columns=list(resolved_config.id_columns), errors="ignore")

    if feature_df.empty:
        raise ValueError("No feature columns remain after dropping label/id columns.")

    invocation = invoke_endpoint_batches(
        feature_df,
        columns=list(feature_df.columns),
        config=endpoint_config,
        ml_client=ml_client,
        max_alerts=max_alerts,
    )

    y_pred = np.asarray(invocation.predictions, dtype=int)
    if y_pred.size != y_true.size:
        raise ValueError(
            "Prediction count does not match label count.",
        )

    probabilities = None
    if invocation.probabilities is not None and len(invocation.probabilities) == y_true.size:
        probabilities = np.asarray(invocation.probabilities, dtype=float)

    metrics = _build_metrics(y_true, y_pred, probabilities)
    if invocation.threshold_used is not None:
        metrics["metrics.threshold_used"] = float(invocation.threshold_used)
    if invocation.max_alerts is not None:
        metrics["metrics.max_alerts"] = float(invocation.max_alerts)
    if invocation.num_alerts is not None:
        metrics["metrics.num_alerts"] = float(invocation.num_alerts)

    labels = ["0", "1"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_json = {"labels": labels, "matrix": cm.tolist()}

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, {k: v for k, v in metrics.items() if v is not None})

    cm_png = output_dir / "confusion_matrix.png"
    cm_json_path = output_dir / "confusion_matrix.json"
    report_path = output_dir / "classification_report.json"

    _plot_confusion_matrix(cm, labels, cm_png)
    write_json(cm_json_path, cm_json)
    write_json(report_path, report)

    with ensure_mlflow_run(experiment_name=experiment_name, run_name=run_name):
        log_metrics(metrics)
        log_artifact(metrics_path)
        log_artifact(cm_png)
        log_artifact(cm_json_path)
        log_artifact(report_path)

    logger.info(
        "Evaluation summary",
        extra={
            "accuracy": metrics.get("metrics.accuracy"),
            "precision": metrics.get("metrics.precision"),
            "recall": metrics.get("metrics.recall"),
            "f1": metrics.get("metrics.f1"),
            "roc_auc": metrics.get("metrics.roc_auc"),
            "pr_auc": metrics.get("metrics.pr_auc"),
        },
    )

    return EvaluationResult(
        metrics=metrics,
        classification_report=report,
        confusion_matrix=cm.tolist(),
        predictions=invocation.predictions,
        probabilities=invocation.probabilities,
        threshold_used=invocation.threshold_used,
    )


__all__ = ["EvaluationConfig", "EvaluationResult", "evaluate_endpoint"]
