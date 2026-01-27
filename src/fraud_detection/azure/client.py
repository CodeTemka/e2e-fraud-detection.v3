"""Helpers for working with Azure ML clients."""

from __future__ import annotations

import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from fraud_detection.config import ROOT_DIR, Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

def build_default_credential() -> DefaultAzureCredential:
    """Create a default azure credential suitable for CI and local use."""

    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def _load_sp_credentials(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    tenant_id = data.get("tenant") or data.get("tenantId")
    client_id = data.get("appId") or data.get("clientId")
    client_secret = data.get("password") or data.get("clientSecret")
    if not all((tenant_id, client_id, client_secret)):
        return None
    return {
        "AZURE_TENANT_ID": str(tenant_id),
        "AZURE_CLIENT_ID": str(client_id),
        "AZURE_CLIENT_SECRET": str(client_secret),
    }


def resolve_azure_env_vars(*, settings: Settings | None = None) -> dict[str, str]:
    resolved = settings or get_settings()
    env_vars: dict[str, str] = {}
    if resolved.subscription_id:
        env_vars["SUBSCRIPTION_ID"] = str(resolved.subscription_id)
    sp_path = ROOT_DIR / "scripts" / "sp_credentials.json"
    env_vars.update(_load_sp_credentials(sp_path) or {})
    return env_vars


def get_ml_client(*, settings: Settings | None = None, credential: DefaultAzureCredential | None = None) -> MLClient:
    """Get an azure ML client.

    Args:
        settings: The application settings. If not provided, the default settings will be used.
        credential: The Azure credentials. If not provided, the default credentials wil be used.

    Returns:
        An Azure ML client.
    """

    resolved_settings = settings or get_settings()
    resolved_credential = credential or build_default_credential()

    logger.info(
        "Connecting to Azure ML workspace",
        extra={
            "subscription_id": resolved_settings.subscription_id,
            "resource_group": resolved_settings.resource_group,
            "workspace_name": resolved_settings.workspace_name,
        },
    )

    return MLClient(
        credential=resolved_credential,
        subscription_id=resolved_settings.subscription_id,
        resource_group_name=resolved_settings.resource_group,
        workspace_name=resolved_settings.workspace_name,
    )


__all__ = ["get_ml_client", "build_default_credential", "resolve_azure_env_vars"]
