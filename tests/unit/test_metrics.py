import numpy as np

from fraud_detection.monitoring.evaluate_endpoint import _build_metrics


def test_build_metrics_includes_auc_when_probabilities_provided():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    metrics = _build_metrics(y_true, y_pred, probabilities)
    assert metrics["metrics.roc_auc"] is not None and metrics["metrics.pr_auc"] is not None


def test_build_metrics_sets_auc_none_without_probabilities():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    metrics = _build_metrics(y_true, y_pred, None)
    assert metrics["metrics.roc_auc"] is None and metrics["metrics.pr_auc"] is None
