from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from nacl import encoding, public

from fraud_detection.config import Settings, get_settings

DEFAULT_SP_PATH = Path(__file__).resolve().parent / "sp_credentials.json"


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "e2e-fraud-detection-secret-uploader",
    }


def _get_public_key(owner: str, repo: str, token: str) -> tuple[str, str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key"
    response = requests.get(url, headers=_headers(token), timeout=30)

    if response.status_code in (401, 403):
        raise PermissionError("GitHub token is not authorized to manage secrets for this repo.")
    if response.status_code == 404:
        raise RuntimeError(
            f"GitHub repo not found or token lacks access: {owner}/{repo} "
            f"({response.status_code} - {response.text})"
        )
    if response.status_code != 200:
        raise RuntimeError(f"Failed to get public key: {response.status_code} - {response.text}")

    data = response.json()
    return data["key_id"], data["key"]


def _encrypt(public_key_b64: str, plaintext: str) -> str:
    key = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed = public.SealedBox(key)
    encrypted_bytes = sealed.encrypt(plaintext.encode("utf-8"))
    return base64.b64encode(encrypted_bytes).decode("utf-8")


def set_github_secret(
    name: str,
    value: str | dict[str, Any],
    *,
    settings: Settings | None = None,
) -> None:
    cfg = settings or get_settings()

    token = (cfg.github_token or "").strip()
    owner = (cfg.github_owner or "").strip()
    repo = (cfg.github_repo or "").strip()

    if not owner or not repo:
        env_repo = os.getenv("GITHUB_REPOSITORY", "").strip()
        if env_repo and "/" in env_repo:
            owner, repo = env_repo.split("/", 1)

    if not owner or not repo:
        raise ValueError(
            "Missing GitHub repository. Set GITHUB_OWNER and GITHUB_REPO " "or GITHUB_REPOSITORY=owner/repo."
        )
    if not token:
        raise ValueError("Missing GitHub token. Set GITHUB_TOKEN in .env or the environment.")

    if isinstance(value, dict):
        value = json.dumps(value)

    key_id, public_key = _get_public_key(owner, repo, token)
    encrypted_value = _encrypt(public_key, value)

    url = f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{name}"
    payload = {"encrypted_value": encrypted_value, "key_id": key_id}
    response = requests.put(url, headers=_headers(token), json=payload, timeout=30)

    if response.status_code not in (201, 204):
        raise RuntimeError(f"Failed to upload secret '{name}': {response.status_code} - {response.text}")

    print(f"OK: secret '{name}' uploaded.")


def update_azure_credentials_secret(
    sp_path: Path = DEFAULT_SP_PATH,
) -> None:
    sp = json.loads(sp_path.read_text(encoding="utf-8"))

    client_id = sp.get("clientId") or sp.get("appId")
    tenant_id = sp.get("tenantId") or sp.get("tenant")
    client_secret = sp.get("clientSecret") or sp.get("password")
    settings = get_settings()
    subscription_id = sp.get("subscriptionId") or sp.get("subscription") or settings.subscription_id

    missing = [
        key
        for key, value in {
            "clientId/appId": client_id,
            "tenantId": tenant_id,
            "clientSecret/password": client_secret,
            "subscriptionId": subscription_id,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing fields in {sp_path}: {', '.join(missing)}")

    azure_credentials = {
        "clientId": client_id,
        "clientSecret": client_secret,
        "tenantId": tenant_id,
        "subscriptionId": subscription_id,
    }

    set_github_secret("AZURE_CREDENTIALS", azure_credentials)


if __name__ == "__main__":
    update_azure_credentials_secret()
