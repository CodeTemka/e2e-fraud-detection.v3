import pandas as pd

from fraud_detection.monitoring.endpoint_client import (
    EndpointConfig,
    _normalize_auth_mode,
    extract_predictions,
    invoke_endpoint_batches,
)


def test_extract_predictions_handles_list_payload():
    payload = [
        {"predicted_label": 0, "probability": 0.2},
        {"predicted_label": 1, "probability": 0.8},
    ]
    result = extract_predictions(payload)
    assert result.predictions == [0, 1] and result.probabilities == [0.2, 0.8]


def test_extract_predictions_handles_nested_payload():
    payload = {
        "results": {"predictions": [1, 0], "probabilities": [0.9, 0.1], "threshold_used": 0.5}
    }
    result = extract_predictions(payload)
    assert result.threshold_used == 0.5 and result.predictions == [1, 0]


def test_invoke_endpoint_batches_aggregates_predictions(monkeypatch):
    calls = []

    def fake_post_json_with_retry(*, url, payload, headers, timeout, max_retries, backoff_base, backoff_max):
        calls.append(payload)
        batch_size = len(payload["data"])
        return {"predictions": [1] * batch_size, "probabilities": [0.9] * batch_size}

    monkeypatch.setattr(
        "fraud_detection.monitoring.endpoint_client._post_json_with_retry",
        fake_post_json_with_retry,
    )

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14]})
    config = EndpointConfig(
        endpoint_name="endpoint",
        scoring_uri="http://example",
        endpoint_key="fake",
        batch_size=2,
    )
    result = invoke_endpoint_batches(df, columns=["a", "b"], config=config)

    assert len(calls) == 3 and len(result.predictions) == 5


def test_normalize_auth_mode_accepts_aad_alias():
    assert _normalize_auth_mode("aml_token") == "aad"
