"""HTTP client utilities for invoking Azure ML online endpoints."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class EndpointConfig:
    endpoint_name: str
    deployment_name: str | None = None
    scoring_uri: str | None = None
    auth_mode: str = "key"
    endpoint_key: str | None = None
    batch_size: int = 200
    request_timeout: float = 30.0
    max_retries: int = 3
    backoff_base: float = 1.5
    backoff_max: float = 30.0
    alert_rate: float | None = None


@dataclass
class EndpointInvocationResult:
    predictions: list[int]
    probabilities: list[float] | None
    threshold_used: float | None
    max_alerts: int | None
    num_alerts: int | None


def _normalize_auth_mode(value: str | None) -> str:
    mode = (value or "key").strip().lower()
    if mode in {"key", "apikey", "api-key"}:
        return "key"
    if mode in {"aad", "token", "aml_token"}:
        return "aad"
    raise ValueError("auth_mode must be 'key' or 'aad'.")


def _resolve_scoring_uri(ml_client: MLClient | None, config: EndpointConfig) -> str:
    if config.scoring_uri:
        return config.scoring_uri
    if ml_client is None:
        raise ValueError("scoring_uri or MLClient must be provided to resolve endpoint URL.")
    endpoint = ml_client.online_endpoints.get(name=config.endpoint_name)
    scoring_uri = getattr(endpoint, "scoring_uri", None)
    if not scoring_uri:
        raise ValueError(f"Endpoint '{config.endpoint_name}' has no scoring URI.")
    return str(scoring_uri)


def _get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _resolve_endpoint_key(ml_client: MLClient | None, endpoint_name: str) -> str | None:
    env_key = _get_env_value("AZUREML_ENDPOINT_KEY", "ENDPOINT_KEY", "AML_ENDPOINT_KEY")
    if env_key:
        return env_key
    if ml_client is None:
        return None

    for fn_name in ("get_keys", "list_keys"):
        fn = getattr(ml_client.online_endpoints, fn_name, None)
        if not callable(fn):
            continue
        try:
            keys = fn(name=endpoint_name)
        except TypeError:
            keys = fn(endpoint_name)
        if isinstance(keys, dict):
            return keys.get("primary_key") or keys.get("primaryKey") or keys.get("primary") or keys.get("key")
        return getattr(keys, "primary_key", None) or getattr(keys, "primaryKey", None)
    return None


def _resolve_aad_token() -> str | None:
    env_token = _get_env_value("AZUREML_ENDPOINT_TOKEN", "ENDPOINT_TOKEN")
    if env_token:
        return env_token
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    token = credential.get_token("https://ml.azure.com/.default")
    return token.token if token else None


def _build_headers(
    *,
    auth_mode: str,
    token: str | None,
    deployment_name: str | None,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if deployment_name:
        headers["azureml-model-deployment"] = deployment_name
    return headers


def _sleep_with_jitter(delay: float) -> None:
    time.sleep(delay + random.uniform(0, max(delay * 0.1, 0.05)))


def _post_json_with_retry(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
) -> dict[str, Any] | list[Any]:
    session = requests.Session()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code < 300:
                try:
                    return response.json()
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Endpoint returned non-JSON response.") from exc

            if response.status_code in _RETRY_STATUS and attempt < max_retries:
                delay = min(backoff_base * (2**attempt), backoff_max)
                logger.warning(
                    "Endpoint call failed; retrying",
                    extra={"status": response.status_code, "attempt": attempt + 1, "delay": delay},
                )
                _sleep_with_jitter(delay)
                continue

            raise RuntimeError(f"Endpoint returned {response.status_code}: {response.text}")
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = min(backoff_base * (2**attempt), backoff_max)
            logger.warning(
                "Endpoint request error; retrying",
                extra={"error": str(exc), "attempt": attempt + 1, "delay": delay},
            )
            _sleep_with_jitter(delay)

    raise RuntimeError(f"Endpoint call failed after retries: {last_error}")


def _extract_values(payload: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _extract_from_nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    value = _extract_values(payload, keys)
    if value is not None:
        return value
    for nested_key in ("result", "results", "output", "outputs", "response"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            value = _extract_values(nested, keys)
            if value is not None:
                return value
    return None


def _get_first_non_none(payload: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def extract_predictions(payload: dict[str, Any] | list[Any]) -> EndpointInvocationResult:
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            preds = [_get_first_non_none(row, ("predicted_label", "prediction", "label")) for row in payload]
            probs = None
            if all(isinstance(row, dict) for row in payload):
                prob_values = [_get_first_non_none(row, ("probability", "score")) for row in payload]
                if all(value is not None for value in prob_values):
                    probs = [float(value) for value in prob_values]
            return EndpointInvocationResult(
                predictions=[int(value) for value in preds],
                probabilities=probs,
                threshold_used=None,
                max_alerts=None,
                num_alerts=None,
            )
        return EndpointInvocationResult(
            predictions=[int(value) for value in payload],
            probabilities=None,
            threshold_used=None,
            max_alerts=None,
            num_alerts=None,
        )

    predictions = _extract_from_nested(payload, ("predicted_label", "predictions", "labels"))
    probabilities = _extract_from_nested(payload, ("probability", "probabilities", "scores"))
    threshold = _extract_from_nested(payload, ("threshold_used", "threshold"))
    max_alerts = _extract_from_nested(payload, ("max_alerts", "alert_cap"))
    num_alerts = _extract_from_nested(payload, ("num_alerts", "alert_count"))

    if predictions is None:
        raise RuntimeError("Endpoint response did not contain predictions.")

    preds_list = [int(value) for value in list(predictions)]
    probs_list = None
    if probabilities is not None:
        probs_list = [float(value) for value in list(probabilities)]

    return EndpointInvocationResult(
        predictions=preds_list,
        probabilities=probs_list,
        threshold_used=float(threshold) if threshold is not None else None,
        max_alerts=int(max_alerts) if max_alerts is not None else None,
        num_alerts=int(num_alerts) if num_alerts is not None else None,
    )


def invoke_endpoint_batches(
    dataframe,
    *,
    columns: list[str],
    config: EndpointConfig,
    ml_client: MLClient | None = None,
    max_alerts: int | None = None,
) -> EndpointInvocationResult:
    """Invoke the endpoint in batches and return merged predictions."""
    import pandas as pd

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame.")

    resolved_mode = _normalize_auth_mode(config.auth_mode)
    scoring_uri = _resolve_scoring_uri(ml_client, config)

    token: str | None = None
    if resolved_mode == "key":
        token = config.endpoint_key or _resolve_endpoint_key(ml_client, config.endpoint_name)
        if not token:
            raise RuntimeError("Endpoint key not found; set ENDPOINT_KEY or provide endpoint_key.")
    else:
        token = _resolve_aad_token()
        if not token:
            raise RuntimeError("AAD token not found; set ENDPOINT_TOKEN or ensure DefaultAzureCredential works.")

    headers = _build_headers(auth_mode=resolved_mode, token=token, deployment_name=config.deployment_name)

    all_predictions: list[int] = []
    all_probabilities: list[float] = []
    probabilities_available = True
    threshold_used: float | None = None
    max_alerts_used: int | None = None
    num_alerts_used: int | None = None

    batch_size = max(int(config.batch_size), 1)
    total_rows = len(dataframe)
    request_batch = min(int(batch_size), max(1, total_rows))

    for start in range(0, total_rows, request_batch):
        batch_df = dataframe.iloc[start : start + request_batch]
        payload: dict[str, Any] = {
            "columns": columns,
            "data": batch_df[columns].to_numpy().tolist(),
        }
        if max_alerts is not None:
            payload["max_alerts"] = int(max_alerts)
        elif config.alert_rate is not None:
            if not (0.0 <= config.alert_rate <= 1.0):
                raise ValueError("alert_rate must be between 0.0 and 1.0.")
            batch_alerts = int(math.ceil(config.alert_rate * len(batch_df)))
            payload["max_alerts"] = batch_alerts

        response = _post_json_with_retry(
            url=scoring_uri,
            payload=payload,
            headers=headers,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
            backoff_base=config.backoff_base,
            backoff_max=config.backoff_max,
        )

        parsed = extract_predictions(response)
        all_predictions.extend(parsed.predictions)
        if parsed.probabilities is not None and probabilities_available:
            all_probabilities.extend(parsed.probabilities)
        elif parsed.probabilities is None:
            probabilities_available = False
            all_probabilities = []
        if threshold_used is None and parsed.threshold_used is not None:
            threshold_used = parsed.threshold_used
        if max_alerts_used is None and parsed.max_alerts is not None:
            max_alerts_used = parsed.max_alerts
        if num_alerts_used is None and parsed.num_alerts is not None:
            num_alerts_used = parsed.num_alerts

    probabilities = all_probabilities if probabilities_available and all_probabilities else None
    return EndpointInvocationResult(
        predictions=all_predictions,
        probabilities=probabilities,
        threshold_used=threshold_used,
        max_alerts=max_alerts_used,
        num_alerts=num_alerts_used,
    )


__all__ = [
    "EndpointConfig",
    "EndpointInvocationResult",
    "extract_predictions",
    "invoke_endpoint_batches",
]
