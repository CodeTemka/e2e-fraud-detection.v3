from types import SimpleNamespace

from typer.testing import CliRunner

import fraud_detection.cli as cli


def test_create_endpoint_invokes_helper(monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_create_endpoint(ml_client, *, name, description, auth_mode, settings):
        called["name"] = name

    monkeypatch.setattr(cli, "get_ml_client", lambda: object())
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(endpoint_name="default"))
    monkeypatch.setattr("fraud_detection.serving.endpoint_ops.create_endpoint", fake_create_endpoint)

    result = runner.invoke(cli.app, ["create-endpoint", "--endpoint-name", "test-endpoint"])
    assert result.exit_code == 0 and called["name"] == "test-endpoint"


def test_delete_endpoint_invokes_helper(monkeypatch):
    runner = CliRunner()
    called = {}

    def fake_delete_endpoint(ml_client, *, name):
        called["name"] = name

    monkeypatch.setattr(cli, "get_ml_client", lambda: object())
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(endpoint_name="default"))
    monkeypatch.setattr("fraud_detection.serving.endpoint_ops.delete_endpoint", fake_delete_endpoint)

    result = runner.invoke(cli.app, ["delete-endpoint", "--endpoint-name", "old-endpoint"])
    assert result.exit_code == 0 and called["name"] == "old-endpoint"
