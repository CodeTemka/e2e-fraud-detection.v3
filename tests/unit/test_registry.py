from types import SimpleNamespace

import pandas as pd

from fraud_detection.registry.prod_model import (
    _is_better,
    _is_same,
    _metric_direction,
    _normalize_experiments,
    _normalize_metric,
    best_run_by_metric,
)


def test_normalize_metric_accepts_alias():
    bare, full = _normalize_metric("auc")
    assert bare == "AUC_macro" and full == "metrics.AUC_macro"


def test_metric_direction_handles_loss():
    assert _metric_direction("log_loss") == "min"


def test_is_better_respects_delta():
    assert _is_better(0.91, 0.90, direction="max", epsilon=0.0, delta=0.005) is True


def test_normalize_experiments_deduplicates():
    settings = SimpleNamespace(custom_train_exp="exp-a", automl_train_exp="exp-b")
    resolved = _normalize_experiments("exp-a, exp-a, exp-b", settings=settings)
    assert resolved == ["exp-a", "exp-b"]


def test_best_run_by_metric_selects_top_run(monkeypatch):
    df = pd.DataFrame(
        {
            "run_id": ["run-1", "run-2"],
            "experiment_name": ["exp-1", "exp-1"],
            "status": ["FINISHED", "FINISHED"],
            "metrics.accuracy": [0.91, 0.95],
        }
    )

    monkeypatch.setattr("mlflow.search_runs", lambda **kwargs: df)

    settings = SimpleNamespace(custom_train_exp="exp-1", automl_train_exp="exp-2")
    best = best_run_by_metric(metric="accuracy", experiments=["exp-1"], settings=settings)
    assert best.run_id == "run-2"


def test_is_same_respects_epsilon():
    assert _is_same(0.900, 0.901, epsilon=0.005, delta=0.0) is True
