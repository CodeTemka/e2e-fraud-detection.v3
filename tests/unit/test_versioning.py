from types import SimpleNamespace

from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.utils.versioning import next_version, resolve_next_environment_version


def test_next_version_increments_patch():
    assert next_version(["1.0", "1.2", "2.0"]) == "2.1"


def test_resolve_next_environment_version_defaults_on_missing(fake_ml_client):
    fake_ml_client.environments.list.side_effect = ResourceNotFoundError("missing")
    assert resolve_next_environment_version(fake_ml_client, name="env") == "1.0"


def test_resolve_next_environment_version_uses_latest(fake_ml_client):
    fake_ml_client.environments.list.return_value = [
        SimpleNamespace(version="1.0"),
        SimpleNamespace(version="1.2"),
    ]
    assert resolve_next_environment_version(fake_ml_client, name="env") == "1.3"
