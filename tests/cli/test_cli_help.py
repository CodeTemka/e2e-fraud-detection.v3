from typer.testing import CliRunner

from fraud_detection.cli import app


def test_cli_help_shows_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0 and "Utilities to orchestrate Azure ML jobs" in result.output


def test_cli_create_endpoint_help():
    runner = CliRunner()
    result = runner.invoke(app, ["create-endpoint", "--help"])
    assert result.exit_code == 0 and "endpoint" in result.output.lower()
