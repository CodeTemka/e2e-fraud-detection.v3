import pandas as pd

from fraud_detection.monitoring.endpoint_client import EndpointConfig, EndpointInvocationResult
from fraud_detection.monitoring.evaluate_endpoint import evaluate_endpoint


def test_evaluate_endpoint_writes_artifacts(tmp_path, mock_mlflow, monkeypatch):
    df = pd.DataFrame(
        {
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1.0, 1.1, 1.2, 1.3],
            "Class": [0, 1, 0, 1],
        }
    )

    def fake_invoke(*args, **kwargs):
        return EndpointInvocationResult(
            predictions=[0, 1, 0, 1],
            probabilities=[0.1, 0.9, 0.2, 0.8],
            threshold_used=0.5,
            max_alerts=None,
            num_alerts=None,
        )

    monkeypatch.setattr("fraud_detection.monitoring.evaluate_endpoint.invoke_endpoint_batches", fake_invoke)

    config = EndpointConfig(endpoint_name="endpoint", scoring_uri="http://example", endpoint_key="fake")
    result = evaluate_endpoint(
        test_data="ignored",
        endpoint_config=config,
        output_dir=tmp_path,
        dataframe=df,
    )

    assert (tmp_path / "metrics.json").exists() and result.metrics["metrics.accuracy"] == 1.0
