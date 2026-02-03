from types import SimpleNamespace

from fraud_detection.training.automl import automl_job_builder, resolve_latest_data_version
from fraud_detection.training.lgbm.submit_lgbm import lgbm_sweep_job_builder
from fraud_detection.training.xgb.submit_xgb import xgb_sweep_job_builder


def test_automl_job_builder_populates_tags(monkeypatch):
    settings = SimpleNamespace(
        training_compute_cluster_name="cpu",
        automl_train_exp="automl-exp",
        registered_train="train-data",
    )
    monkeypatch.setattr("fraud_detection.training.automl.get_settings", lambda: settings)
    monkeypatch.setattr("fraud_detection.training.automl.get_git_sha", lambda short=False: "deadbeef")

    config = automl_job_builder(metric="accuracy", training_data="train-data")
    assert config.job_name.startswith("automl-") and config.tags["git_sha"] == "deadbeef"


def test_xgb_sweep_job_builder_sets_job_name(monkeypatch):
    settings = SimpleNamespace(
        custom_train_exp="custom-exp",
        training_compute_cluster_name="cpu",
    )
    monkeypatch.setattr("fraud_detection.training.xgb.submit_xgb.get_settings", lambda: settings)

    config = xgb_sweep_job_builder(training_data="train-data", metric="average_precision_score_macro")
    assert config.job_name.startswith("xgb-sweep-")


def test_lgbm_sweep_job_builder_sets_job_name(monkeypatch):
    settings = SimpleNamespace(
        custom_train_exp="custom-exp",
        training_compute_cluster_name="cpu",
    )
    monkeypatch.setattr("fraud_detection.training.lgbm.submit_lgbm.get_settings", lambda: settings)

    config = lgbm_sweep_job_builder(training_data="train-data", metric="average_precision_score_macro")
    assert config.job_name.startswith("lgbm-sweep-")


def test_resolve_latest_data_version_selects_highest(fake_ml_client):
    fake_ml_client.data.list.return_value = [
        SimpleNamespace(version="1"),
        SimpleNamespace(version="3"),
    ]

    assert resolve_latest_data_version(fake_ml_client, "dataset") == "3"
