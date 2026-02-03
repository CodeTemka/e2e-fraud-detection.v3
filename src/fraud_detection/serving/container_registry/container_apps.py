"""Deploy fraud detection model to Azure Container Apps."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import MLClient
from fraud_detection.azure.client import resolve_azure_env_vars
from fraud_detection.config import ROOT_DIR, Settings, get_settings
from fraud_detection.serving.managed_endpoint.serve_prod_model import resolve_model
from fraud_detection.training.automl import resolve_latest_data_version
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from datetime import UTC as _UTC  # type: ignore
except ImportError:  # pragma: no cover - Python < 3.11
    _UTC = timezone.utc

UTC = _UTC


@dataclass(frozen=True)
class ContainerAppDeployResult:
    model_name: str
    model_version: str
    app_name: str
    image: str
    fqdn: str | None
    revision: str | None
    scaler_asset_name: str
    scaler_asset_version: str
    action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": "container-apps",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "app_name": self.app_name,
            "image": self.image,
            "fqdn": self.fqdn,
            "revision": self.revision,
            "scaler_asset_name": self.scaler_asset_name,
            "scaler_asset_version": self.scaler_asset_version,
            "action": self.action,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }


def _resolve_az_path() -> str:
    az_path = os.environ.get("AZ_PATH") or shutil.which("az")
    if not az_path:
        raise RuntimeError("Azure CLI 'az' not found. Install Azure CLI or set AZ_PATH.")
    return az_path


def _run_az(command: list[str], *, expect_json: bool = True) -> dict[str, object] | str:
    az_path = _resolve_az_path()
    result = subprocess.run([az_path, *command], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Azure CLI failed: {message or 'unknown error'}")

    output = (result.stdout or "").strip()
    if not expect_json:
        return output
    if not output:
        return {}
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Azure CLI returned non-JSON output.") from exc


def _run_az_tsv(command: list[str]) -> str:
    az_path = _resolve_az_path()
    result = subprocess.run([az_path, *command], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Azure CLI failed: {message or 'unknown error'}")
    return (result.stdout or "").strip()


def _resource_exists(command: list[str]) -> bool:
    try:
        _run_az(command, expect_json=True)
        return True
    except RuntimeError as exc:
        message = str(exc).lower()
        if "not found" in message or "was not found" in message or "resource not found" in message:
            return False
        raise


def _ensure_container_env(
    *,
    name: str,
    resource_group: str,
    location: str,
    subscription_id: str | None,
) -> None:
    exists = _resource_exists(
        [
            "containerapp",
            "env",
            "show",
            "--name",
            name,
            "--resource-group",
            resource_group,
            "--output",
            "json",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ]
    )
    if exists:
        return

    logger.info("Creating Container Apps environment", extra={"env_name": name})
    _run_az(
        [
            "containerapp",
            "env",
            "create",
            "--name",
            name,
            "--resource-group",
            resource_group,
            "--location",
            location,
            "--output",
            "json",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ],
        expect_json=True,
    )


def _get_acr_login_server(registry_name: str, subscription_id: str | None) -> str:
    return _run_az_tsv(
        [
            "acr",
            "show",
            "--name",
            registry_name,
            "--query",
            "loginServer",
            "--output",
            "tsv",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ]
    )


def _get_acr_resource_id(registry_name: str, subscription_id: str | None) -> str:
    return _run_az_tsv(
        [
            "acr",
            "show",
            "--name",
            registry_name,
            "--query",
            "id",
            "--output",
            "tsv",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ]
    )


def _get_containerapp_principal_id(
    app_name: str, resource_group: str, subscription_id: str | None
) -> str:
    return _run_az_tsv(
        [
            "containerapp",
            "show",
            "--name",
            app_name,
            "--resource-group",
            resource_group,
            "--query",
            "identity.principalId",
            "--output",
            "tsv",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ]
    )


def _ensure_system_identity(
    app_name: str, resource_group: str, subscription_id: str | None
) -> None:
    try:
        _run_az(
            [
                "containerapp",
                "identity",
                "assign",
                "--name",
                app_name,
                "--resource-group",
                resource_group,
                "--system-assigned",
                "--output",
                "json",
                *(
                    ["--subscription", subscription_id]
                    if subscription_id
                    else []
                ),
            ],
            expect_json=True,
        )
        return
    except RuntimeError as exc:
        message = str(exc).lower()
        if "unrecognized" not in message and "not found" not in message:
            raise

    _run_az(
        [
            "containerapp",
            "update",
            "--name",
            app_name,
            "--resource-group",
            resource_group,
            "--system-assigned",
            "--output",
            "json",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ],
        expect_json=True,
    )


def _ensure_acr_pull_role(
    principal_id: str, acr_id: str, subscription_id: str | None
) -> None:
    try:
        _run_az(
            [
                "role",
                "assignment",
                "create",
                "--assignee",
                principal_id,
                "--role",
                "AcrPull",
                "--scope",
                acr_id,
                "--output",
                "json",
                *(
                    ["--subscription", subscription_id]
                    if subscription_id
                    else []
                ),
            ],
            expect_json=True,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "roleassignmentexists" in message or "already exists" in message:
            return
        raise


def _set_registry_identity(
    app_name: str,
    resource_group: str,
    login_server: str,
    subscription_id: str | None,
) -> None:
    _run_az(
        [
            "containerapp",
            "registry",
            "set",
            "--name",
            app_name,
            "--resource-group",
            resource_group,
            "--server",
            login_server,
            "--identity",
            "system",
            "--output",
            "json",
            *(
                ["--subscription", subscription_id]
                if subscription_id
                else []
            ),
        ],
        expect_json=True,
    )


def _build_env_vars(
    settings: Settings,
    *,
    model_name: str,
    model_version: str,
    scaler_asset_name: str,
    scaler_asset_version: str,
    max_alerts: int,
    include_sp_credentials: bool,
) -> dict[str, str]:
    env_vars: dict[str, str] = {
        "SUBSCRIPTION_ID": str(settings.subscription_id or "").strip(),
        "RESOURCE_GROUP": str(settings.resource_group),
        "WORKSPACE_NAME": str(settings.workspace_name),
        "MODEL_NAME": model_name,
        "MODEL_VERSION": model_version,
        "SCALER_ASSET_NAME": scaler_asset_name,
        "SCALER_ASSET_VERSION": scaler_asset_version,
        "MODEL_DIR": "/app/assets/model",
        "SCALER_DIR": "/app/assets/scalers",
        "AZUREML_MODEL_DIR": "/app/assets/model",
        "MAX_ALERTS": str(max_alerts),
        "ALERT_CAP": str(max_alerts),
    }
    if not include_sp_credentials:
        env_vars["USE_MANAGED_IDENTITY"] = "true"
    resolved_env = resolve_azure_env_vars(settings=settings)
    if not include_sp_credentials:
        for key in ("AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"):
            resolved_env.pop(key, None)
    env_vars.update(resolved_env)
    return {key: value for key, value in env_vars.items() if value}


def deploy_container_app(
    ml_client: MLClient,
    *,
    app_name: str,
    environment_name: str,
    model_name: str,
    model_version: str | None = None,
    scaler_asset_name: str,
    scaler_asset_version: str | None = None,
    registry_name: str | None = None,
    image_name: str | None = None,
    image_tag: str | None = None,
    image: str | None = None,
    dockerfile: str | Path | None = None,
    source_dir: str | Path | None = None,
    target_port: int = 8000,
    max_alerts: int = 100,
    dry_run: bool = False,
    include_sp_credentials: bool = False,
    settings: Settings | None = None,
) -> ContainerAppDeployResult:
    cfg = settings or get_settings()

    if not cfg.subscription_id:
        raise ValueError("subscription_id is required to deploy Container Apps.")

    resolved_model = resolve_model(
        ml_client,
        model_name=model_name,
        model_version=model_version,
    )

    resolved_scaler_version = (
        scaler_asset_version
        if scaler_asset_version
        else resolve_latest_data_version(ml_client, scaler_asset_name)
    )

    resolved_image = image
    if not resolved_image:
        if not registry_name:
            raise ValueError("registry_name is required when image is not provided.")
        resolved_image_name = image_name or app_name
        resolved_tag = image_tag or f"model-{resolved_model.version}"
        resolved_image = f"{registry_name}.azurecr.io/{resolved_image_name}:{resolved_tag}"

    env_vars = _build_env_vars(
        cfg,
        model_name=resolved_model.name,
        model_version=resolved_model.version,
        scaler_asset_name=scaler_asset_name,
        scaler_asset_version=resolved_scaler_version,
        max_alerts=max_alerts,
        include_sp_credentials=include_sp_credentials,
    )

    if dry_run:
        return ContainerAppDeployResult(
            model_name=resolved_model.name,
            model_version=resolved_model.version,
            app_name=app_name,
            image=resolved_image,
            fqdn=None,
            revision=None,
            scaler_asset_name=scaler_asset_name,
            scaler_asset_version=resolved_scaler_version,
            action="dry_run",
        )

    if not image and registry_name:
        dockerfile_path = Path(dockerfile or ROOT_DIR / "deploy" / "container_apps" / "Dockerfile")
        source_path = Path(source_dir or ROOT_DIR)
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        if not source_path.exists():
            raise FileNotFoundError(f"Source dir not found: {source_path}")

        logger.info(
            "Building container image",
            extra={"registry": registry_name, "image": resolved_image, "dockerfile": str(dockerfile_path)},
        )
        _run_az(
            [
                "acr",
                "build",
                "--registry",
                registry_name,
                "--image",
                resolved_image.split("/", maxsplit=1)[-1],
                "--file",
                str(dockerfile_path),
                str(source_path),
                "--output",
                "none",
                "--subscription",
                cfg.subscription_id,
            ],
            expect_json=False,
        )

    _ensure_container_env(
        name=environment_name,
        resource_group=cfg.resource_group,
        location=cfg.location,
        subscription_id=cfg.subscription_id,
    )

    login_server = None
    if registry_name:
        login_server = _get_acr_login_server(registry_name, cfg.subscription_id)

    env_pairs = [f"{key}={value}" for key, value in env_vars.items()]
    remove_envs: list[str] = []
    if not include_sp_credentials:
        remove_envs = ["AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID"]

    app_exists = _resource_exists(
        [
            "containerapp",
            "show",
            "--name",
            app_name,
            "--resource-group",
            cfg.resource_group,
            "--output",
            "json",
            "--subscription",
            cfg.subscription_id,
        ]
    )

    if not app_exists:
        logger.info("Creating Container App", extra={"app_name": app_name})
        create_cmd = [
            "containerapp",
            "create",
            "--name",
            app_name,
            "--resource-group",
            cfg.resource_group,
            "--environment",
            environment_name,
            "--image",
            resolved_image,
            "--ingress",
            "external",
            "--target-port",
            str(target_port),
            "--output",
            "json",
            "--subscription",
            cfg.subscription_id,
        ]
        if login_server:
            create_cmd += [
                "--registry-server",
                login_server,
                "--registry-identity",
                "system",
                "--system-assigned",
            ]
        if env_pairs:
            create_cmd += ["--env-vars", *env_pairs]
        response = _run_az(create_cmd, expect_json=True)
    else:
        logger.info("Updating Container App", extra={"app_name": app_name})
        update_cmd = [
            "containerapp",
            "update",
            "--name",
            app_name,
            "--resource-group",
            cfg.resource_group,
            "--image",
            resolved_image,
            "--output",
            "json",
            "--subscription",
            cfg.subscription_id,
        ]
        # containerapp update uses --set-env-vars
        if env_pairs:
            update_cmd += ["--set-env-vars", *env_pairs]
        if remove_envs:
            update_cmd += ["--remove-env-vars", *remove_envs]
        try:
            response = _run_az(update_cmd, expect_json=True)
        except RuntimeError as exc:
            message = str(exc).lower()
            if remove_envs and "unrecognized arguments" in message and "--remove-env-vars" in message:
                fallback_cmd = [arg for arg in update_cmd if arg != "--remove-env-vars"]
                for key in remove_envs:
                    if key in fallback_cmd:
                        fallback_cmd.remove(key)
                response = _run_az(fallback_cmd, expect_json=True)
            else:
                raise

    if login_server and registry_name or not include_sp_credentials:
        _ensure_system_identity(
            app_name,
            cfg.resource_group,
            cfg.subscription_id,
        )

    if login_server and registry_name:
        principal_id = _get_containerapp_principal_id(
            app_name,
            cfg.resource_group,
            cfg.subscription_id,
        )
        acr_id = _get_acr_resource_id(registry_name, cfg.subscription_id)
        _ensure_acr_pull_role(principal_id, acr_id, cfg.subscription_id)
        _set_registry_identity(
            app_name,
            cfg.resource_group,
            login_server,
            cfg.subscription_id,
        )

    fqdn = None
    revision = None
    if isinstance(response, dict):
        fqdn = (
            response.get("properties", {})
            .get("configuration", {})
            .get("ingress", {})
            .get("fqdn")
        )
        revision = response.get("properties", {}).get("latestRevisionName")

    return ContainerAppDeployResult(
        model_name=resolved_model.name,
        model_version=resolved_model.version,
        app_name=app_name,
        image=resolved_image,
        fqdn=fqdn,
        revision=revision,
        scaler_asset_name=scaler_asset_name,
        scaler_asset_version=resolved_scaler_version,
        action="deployed",
    )


__all__ = ["ContainerAppDeployResult", "deploy_container_app"]
