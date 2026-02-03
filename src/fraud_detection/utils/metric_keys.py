"""Canonical metric keys and logging helpers."""

from __future__ import annotations

CANONICAL_METRIC_KEYS: list[str] = [
    "metrics.weighted_accuracy",
    "metrics.AUC_micro",
    "metrics.precision_score_weighted",
    "metrics.f1_score_macro",
    "metrics.norm_macro_recall",
    "metrics.average_precision_score_weighted",
    "metrics.AUC_macro",
    "metrics.f1_score_micro",
    "metrics.recall_score_macro",
    "metrics.recall_score_micro",
    "metrics.precision_score_micro",
    "metrics.AUC_weighted",
    "metrics.log_loss",
    "metrics.average_precision_score_macro",
    "metrics.f1_score_weighted",
    "metrics.precision_score_macro",
    "metrics.average_precision_score_micro",
    "metrics.recall_score_weighted",
    "metrics.matthews_correlation",
    "metrics.accuracy",
    "metrics.balanced_accuracy",
]

CANONICAL_METRICS_BARE = sorted(
    {m[len("metrics.") :] for m in CANONICAL_METRIC_KEYS if m.startswith("metrics.")}
)

ALIAS_TO_CANONICAL = {
    "average_precision": "metrics.average_precision_score_macro",
    "average_precision_score": "metrics.average_precision_score_macro",
    "average_precision_score_macro": "metrics.average_precision_score_macro",
    "average_precision_score_weighted": "metrics.average_precision_score_weighted",
    "average_precision_score_micro": "metrics.average_precision_score_micro",
    "roc_auc": "metrics.AUC_macro",
    "auc": "metrics.AUC_macro",
    "auc_macro": "metrics.AUC_macro",
    "auc_weighted": "metrics.AUC_weighted",
    "auc_micro": "metrics.AUC_micro",
    "accuracy": "metrics.accuracy",
    "balanced_accuracy": "metrics.balanced_accuracy",
}


__all__ = [
    "CANONICAL_METRIC_KEYS",
    "CANONICAL_METRICS_BARE",
    "ALIAS_TO_CANONICAL",
]
