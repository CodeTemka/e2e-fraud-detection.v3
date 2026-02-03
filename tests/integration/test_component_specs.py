from types import SimpleNamespace

from fraud_detection.pipeline.deployment_pipeline import _apply_env_vars_to_jobs, _resolve_scaler_asset_uri


def test_resolve_scaler_asset_uri_uses_latest_version(fake_ml_client):
    fake_ml_client.data.list.return_value = [
        SimpleNamespace(version="1"),
        SimpleNamespace(version="3"),
    ]
    settings = SimpleNamespace(serving_scalers_name="fraud-scalers")

    uri = _resolve_scaler_asset_uri(fake_ml_client, scaler_asset_name=None, settings=settings)
    assert uri == "azureml:fraud-scalers:3"


def test_apply_env_vars_to_jobs_merges(monkeypatch):
    job_a = SimpleNamespace(environment_variables={"A": "1"})
    job_b = SimpleNamespace(environment_variables=None)
    pipeline_job = SimpleNamespace(jobs={"a": job_a, "b": job_b})

    _apply_env_vars_to_jobs(pipeline_job, {"B": "2"})

    assert pipeline_job.jobs["a"].environment_variables == {"A": "1", "B": "2"}
