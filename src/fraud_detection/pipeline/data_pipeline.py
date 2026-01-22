"""Pipeline job for validating data and preparing training splits."""

import argparse
import json
import sys
from pathlib import Path
from azure.ai.ml import MLClient

from fraud_detection.config import ROOT_DIR, get_settings
from fraud_detection.utils.logging import get_logger


logger = get_logger(__name__)

VALIDATION_COMPONENT_NAME = "data_validation"
PREP_COMPONENT_NAME = "prep_data_for_train"


def _parse_version(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _resolve_latest_component_version(ml_client: MLClient, name: str) -> str:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        items = ml_client.components.list(name=name)
    except ResourceNotFoundError as exc:
        raise ValueError(f"No component named '{name}' found.") from exc

    versions: list[tuple[tuple[int, ...], str]] = []
    for item in items:
        parsed = _parse_version(getattr(item, "version", ""))
        if parsed is not None:
            versions.append((parsed, str(item.version)))

    if not versions:
        raise ValueError(f"No numeric versions found for component '{name}'.")

    return max(versions, key=lambda entry: entry[0])[1]


def _get_component(ml_client: MLClient, name: str, version: str | None) -> object:
    resolved_version = version or _resolve_latest_component_version(ml_client, name)
    return ml_client.components.get(name=name, version=resolved_version)


def _normalize_bool(text: str) -> bool:
    value = text.strip().lower()
    return value in {"true", "1", "yes", "y"}


def _read_is_valid_file(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return _normalize_bool(content)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _cli_validate_or_fail(args: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--is-valid-file", required=True)
    parser.add_argument("--output", required=True)
    parsed = parser.parse_args(args)

    is_valid = _read_is_valid_file(Path(parsed.is_valid_file))
    report_path = Path(parsed.validation_report)
    report_payload: dict[str, object] | str = "missing"
    if report_path.exists():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report_payload = report_path.read_text(encoding="utf-8")

    if not is_valid:
        logger.error(
            "Data validation failed; skipping data preparation.",
            extra={"is_valid": is_valid, "report": report_payload},
        )
        return 1

    _write_text(Path(parsed.output), "true")
    return 0


def _build_validation_gate_component(environment_ref: object) -> object:
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes

    return command(
        name="validation_gate",
        display_name="Validation Gate",
        inputs={
            "validation_report": Input(type=AssetTypes.URI_FILE),
            "is_valid_file": Input(type=AssetTypes.URI_FILE),
        },
        outputs={
            "validation_gate": Output(type=AssetTypes.URI_FILE),
        },
        environment=environment_ref,
        code=ROOT_DIR / "src",
        command=(
            "python -m fraud_detection.pipeline.data_pipeline validate-or-fail "
            "--validation-report ${{inputs.validation_report}} "
            "--is-valid-file ${{inputs.is_valid_file}} "
            "--output ${{outputs.validation_gate}}"
        ),
    )


def create_data_pipeline_job(
    *,
    data_name: str,
    label_col: str,
    test_ratio: float,
    seed: int,
    validation_component_version: str | None = None,
    prep_component_version: str | None = None,
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
) -> object:
    """Create a pipeline job that validates data and prepares training splits."""
    from azure.ai.ml import dsl
    from azure.ai.ml.entities import Environment

    from fraud_detection.azure.client import get_ml_client

    ml_client = ml_client or get_ml_client()
    settings = get_settings()

    validation_component = _get_component(ml_client, VALIDATION_COMPONENT_NAME, validation_component_version)
    prep_component = _get_component(ml_client, PREP_COMPONENT_NAME, prep_component_version)

    environment_ref = getattr(validation_component, "environment", None)
    if environment_ref is None:
        environment_ref = Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            name="pipeline-lite-env",
        )

    validation_gate_component = _build_validation_gate_component(environment_ref)

    @dsl.pipeline(
        name="fraud-data-validation-and-prep",
        description="Validate registered data and prepare train/test splits.",
    )
    def _pipeline(registered_data: str, label_col: str, test_ratio: float, seed: int):
        validation_job = validation_component(data=registered_data)
        gate_job = validation_gate_component(
            validation_report=validation_job.outputs.validation_report,
            is_valid_file=validation_job.outputs.is_valid,
        )
        prep_component(
            registered_data=registered_data,
            label_col=label_col,
            test_ratio=test_ratio,
            seed=seed,
            validation_gate=gate_job.outputs.validation_gate,
        )

    pipeline_job = _pipeline(
        registered_data=data_name,
        label_col=label_col,
        test_ratio=test_ratio,
        seed=seed,
    )
    pipeline_job.settings.default_compute = settings.data_compute_cluster_name
    pipeline_job.experiment_name = experiment_name or "data-pipeline"
    return pipeline_job


def submit_data_pipeline_job(
    *,
    data_name: str,
    label_col: str,
    test_ratio: float,
    seed: int,
    validation_component_version: str | None = None,
    prep_component_version: str | None = None,
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
    ensure_compute: bool = True,
) -> object:
    """Submit the data pipeline job to Azure ML."""
    from fraud_detection.azure.client import get_ml_client
    from fraud_detection.utils.compute import ensure_data_compute

    ml_client = ml_client or get_ml_client()
    if ensure_compute:
        ensure_data_compute(ml_client)

    pipeline_job = create_data_pipeline_job(
        data_name=data_name,
        label_col=label_col,
        test_ratio=test_ratio,
        seed=seed,
        validation_component_version=validation_component_version,
        prep_component_version=prep_component_version,
        experiment_name=experiment_name,
        ml_client=ml_client,
    )
    return ml_client.jobs.create_or_update(pipeline_job)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Expected subcommand: validate-or-fail")

    subcommand = sys.argv[1]
    args = sys.argv[2:]
    if subcommand == "validate-or-fail":
        raise SystemExit(_cli_validate_or_fail(args))

    raise SystemExit(f"Unknown subcommand: {subcommand}")


if __name__ == "__main__":
    main()
