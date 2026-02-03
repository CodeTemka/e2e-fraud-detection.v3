"""Drift detection utilities (PSI/KS)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from pandas.api import types as ptypes

from fraud_detection.monitoring.data_loading import load_dataframe
from fraud_detection.monitoring.mlflow_utils import ensure_mlflow_run, log_artifact, log_metrics, write_json
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftConfig:
    label_column: str = "Class"
    psi_bins: int = 10
    psi_threshold: float = 0.2
    ks_threshold: float = 0.1
    sample_rows: int | None = None
    include_ks: bool = True


@dataclass
class DriftResult:
    report: pd.DataFrame
    summary: dict[str, Any]


def _psi(reference: np.ndarray, current: np.ndarray, bins: int) -> float:
    if bins < 2:
        raise ValueError("psi_bins must be >= 2")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if edges.size < 2:
        return float("nan")

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return float("nan")

    ref_dist = ref_counts / ref_counts.sum()
    cur_dist = cur_counts / cur_counts.sum()

    eps = 1e-6
    ref_dist = np.clip(ref_dist, eps, None)
    cur_dist = np.clip(cur_dist, eps, None)
    ref_dist /= ref_dist.sum()
    cur_dist /= cur_dist.sum()

    return float(np.sum((ref_dist - cur_dist) * np.log(ref_dist / cur_dist)))


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.sort(reference)
    current = np.sort(current)
    all_vals = np.concatenate([reference, current])
    ref_cdf = np.searchsorted(reference, all_vals, side="right") / reference.size
    cur_cdf = np.searchsorted(current, all_vals, side="right") / current.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def _numeric_features(df: pd.DataFrame, *, label_column: str, exclude: Iterable[str] | None) -> list[str]:
    exclude_set = {label_column, *list(exclude or [])}
    features: list[str] = []
    for col in df.columns:
        if col in exclude_set:
            continue
        if ptypes.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def _build_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    features: list[str],
    psi_bins: int,
    include_ks: bool,
    psi_threshold: float,
    ks_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        ref = pd.to_numeric(reference_df[feature], errors="coerce").to_numpy()
        cur = pd.to_numeric(current_df[feature], errors="coerce").to_numpy()
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]

        if ref.size == 0 or cur.size == 0:
            rows.append(
                {
                    "feature": feature,
                    "psi": np.nan,
                    "ks": np.nan if include_ks else None,
                    "psi_exceeds": False,
                    "ks_exceeds": False,
                }
            )
            continue

        psi_val = _psi(ref, cur, psi_bins)
        ks_val = _ks_statistic(ref, cur) if include_ks else None
        rows.append(
            {
                "feature": feature,
                "psi": psi_val,
                "ks": ks_val,
                "psi_exceeds": bool(psi_val >= psi_threshold) if np.isfinite(psi_val) else False,
                "ks_exceeds": bool(ks_val >= ks_threshold) if include_ks and ks_val is not None else False,
            }
        )

    return pd.DataFrame(rows)


def _summarize(report: pd.DataFrame, *, psi_threshold: float, ks_threshold: float, include_ks: bool) -> dict[str, Any]:
    if report.empty:
        return {
            "feature_count": 0,
            "psi_threshold": psi_threshold,
            "ks_threshold": ks_threshold,
            "psi_max": None,
            "psi_mean": None,
            "ks_max": None,
            "ks_mean": None,
            "psi_exceeded_count": 0,
            "ks_exceeded_count": 0,
            "drift_detected": False,
        }

    psi_values = report["psi"].dropna()
    ks_values = report["ks"].dropna() if include_ks and "ks" in report.columns else pd.Series(dtype=float)

    psi_exceeds = int(report["psi_exceeds"].sum()) if "psi_exceeds" in report.columns else 0
    ks_exceeds = int(report["ks_exceeds"].sum()) if include_ks and "ks_exceeds" in report.columns else 0
    drift_detected = psi_exceeds > 0 or (include_ks and ks_exceeds > 0)

    return {
        "feature_count": int(len(report)),
        "psi_threshold": psi_threshold,
        "ks_threshold": ks_threshold,
        "psi_max": float(psi_values.max()) if not psi_values.empty else None,
        "psi_mean": float(psi_values.mean()) if not psi_values.empty else None,
        "ks_max": float(ks_values.max()) if not ks_values.empty else None,
        "ks_mean": float(ks_values.mean()) if not ks_values.empty else None,
        "psi_exceeded_count": psi_exceeds,
        "ks_exceeded_count": ks_exceeds,
        "drift_detected": drift_detected,
    }


def run_drift_check(
    *,
    reference_data: str | Path,
    current_data: str | Path,
    output_dir: Path,
    config: DriftConfig | None = None,
    ml_client: MLClient | None = None,
    data_label: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    reference_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
    feature_columns: Iterable[str] | None = None,
) -> DriftResult:
    """Compute drift metrics and log artifacts to MLflow."""
    resolved = config or DriftConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_df = (
        reference_df
        if reference_df is not None
        else load_dataframe(
            reference_data,
            ml_client=ml_client,
            data_label=data_label,
            sample_rows=resolved.sample_rows,
        )
    )
    cur_df = (
        current_df
        if current_df is not None
        else load_dataframe(
            current_data,
            ml_client=ml_client,
            data_label=data_label,
            sample_rows=resolved.sample_rows,
        )
    )

    if feature_columns is None:
        features = _numeric_features(cur_df, label_column=resolved.label_column, exclude=None)
    else:
        features = [col for col in feature_columns if col in ref_df.columns and col in cur_df.columns]

    features = [col for col in features if col in ref_df.columns and col in cur_df.columns]

    if not features:
        raise ValueError("No numeric features found for drift computation.")

    report = _build_report(
        ref_df,
        cur_df,
        features=features,
        psi_bins=resolved.psi_bins,
        include_ks=resolved.include_ks,
        psi_threshold=resolved.psi_threshold,
        ks_threshold=resolved.ks_threshold,
    )
    summary = _summarize(
        report,
        psi_threshold=resolved.psi_threshold,
        ks_threshold=resolved.ks_threshold,
        include_ks=resolved.include_ks,
    )

    report_csv = output_dir / "drift_report.csv"
    report_json = output_dir / "drift_report.json"
    summary_json = output_dir / "drift_summary.json"

    report.to_csv(report_csv, index=False)
    write_json(summary_json, summary)
    write_json(report_json, report.to_dict(orient="records"))

    metrics: dict[str, float | None] = {
        "drift.psi_max": summary.get("psi_max"),
        "drift.psi_mean": summary.get("psi_mean"),
        "drift.psi_exceeded_count": float(summary.get("psi_exceeded_count", 0)),
        "drift.ks_max": summary.get("ks_max"),
        "drift.ks_mean": summary.get("ks_mean"),
        "drift.ks_exceeded_count": float(summary.get("ks_exceeded_count", 0)),
        "drift.detected": 1.0 if summary.get("drift_detected") else 0.0,
    }

    with ensure_mlflow_run(experiment_name=experiment_name, run_name=run_name):
        log_metrics(metrics)
        log_artifact(report_csv)
        log_artifact(report_json)
        log_artifact(summary_json)

    logger.info(
        "Drift summary",
        extra={
            "psi_max": summary.get("psi_max"),
            "psi_exceeded_count": summary.get("psi_exceeded_count"),
            "ks_max": summary.get("ks_max"),
            "ks_exceeded_count": summary.get("ks_exceeded_count"),
            "drift_detected": summary.get("drift_detected"),
        },
    )

    return DriftResult(report=report, summary=summary)


__all__ = ["DriftConfig", "DriftResult", "run_drift_check"]
