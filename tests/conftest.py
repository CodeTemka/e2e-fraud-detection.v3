import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")


@pytest.fixture
def fake_ml_client():
    client = MagicMock()
    client.data = MagicMock()
    client.environments = MagicMock()
    client.components = MagicMock()
    client.jobs = MagicMock()
    client.compute = MagicMock()
    client.online_endpoints = MagicMock()
    client.models = MagicMock()
    client.workspaces = MagicMock()
    return client


@pytest.fixture
def sample_train_data():
    rng = np.random.default_rng(42)
    size = 120
    return pd.DataFrame(
        {
            "Time": rng.integers(0, 10000, size=size),
            "Amount": np.abs(rng.normal(120, 35, size=size)),
            "V1": rng.normal(size=size),
            "V2": rng.normal(size=size),
            "Class": rng.integers(0, 2, size=size),
        }
    )


@pytest.fixture
def sample_test_data():
    rng = np.random.default_rng(7)
    size = 80
    return pd.DataFrame(
        {
            "Time": rng.integers(0, 10000, size=size),
            "Amount": np.abs(rng.normal(110, 30, size=size)),
            "V1": rng.normal(size=size),
            "V2": rng.normal(size=size),
            "Class": rng.integers(0, 2, size=size),
        }
    )


@pytest.fixture
def sample_credit_df():
    rng = np.random.default_rng(123)
    size = 200
    columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    data = {col: rng.normal(size=size) for col in columns}
    data["Time"] = np.abs(rng.normal(loc=1000, scale=250, size=size))
    data["Amount"] = np.abs(rng.normal(loc=75, scale=20, size=size))
    labels = np.zeros(size, dtype=int)
    labels[:2] = 1
    rng.shuffle(labels)
    data["Class"] = labels
    return pd.DataFrame(data)


@pytest.fixture
def fake_endpoint_response():
    return {
        "predictions": [0, 1, 0, 1],
        "probabilities": [0.1, 0.9, 0.2, 0.8],
        "threshold_used": 0.5,
        "max_alerts": 2,
        "num_alerts": 2,
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_mlflow(monkeypatch):
    import fraud_detection.monitoring.mlflow_utils as mlflow_utils

    @contextlib.contextmanager
    def _dummy_run(*args, **kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="dummy"))

    monkeypatch.setattr(mlflow_utils, "ensure_mlflow_run", lambda **_: _dummy_run())
    monkeypatch.setattr(mlflow_utils, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow_utils, "log_artifact", lambda *args, **kwargs: None)
    return True
