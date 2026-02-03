"""Runtime helpers for Container Apps serving."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import unquote
import re
import inspect
from pathlib import Path
from typing import Iterable

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.serving.managed_endpoint.serve_prod_model import resolve_model
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_ARTIFACT_FILES = {
    "MLmodel",
    "model.json",
    "model.ubj",
    "model.txt",
    "model.joblib",
    "model.pkl",
    "model.pickle",
}

SCALER_HINT_FILES = {
    "amount_scaler.joblib",
    "time_scaler.joblib",
    "amount_scaler.pkl",
    "time_scaler.pkl",
}


@dataclass(frozen=True)
class RuntimeAssets:
    model_name: str
    model_version: str
    model_dir: Path
    scaler_asset_name: str
    scaler_asset_version: str
    scaler_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_dir": str(self.model_dir),
            "scaler_asset_name": self.scaler_asset_name,
            "scaler_asset_version": self.scaler_asset_version,
            "scaler_dir": str(self.scaler_dir),
        }


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, "").strip()
    return value or default


def _use_managed_identity() -> bool:
    return os.getenv("USE_MANAGED_IDENTITY", "").strip().lower() in {"1", "true", "yes", "y"}


def _build_runtime_credential():
    if _use_managed_identity():
        return ManagedIdentityCredential()
    return DefaultAzureCredential(exclude_interactive_browser_credential=True)


def _parse_version(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _resolve_latest_data_version(ml_client: MLClient, name: str) -> str:
    try:
        items = ml_client.data.list(name=name)
    except ResourceNotFoundError as exc:
        raise ValueError(f"No data asset named '{name}' found.") from exc

    versions: list[tuple[tuple[int, ...], str]] = []
    for item in items:
        parsed = _parse_version(getattr(item, "version", ""))
        if parsed is not None:
            versions.append((parsed, str(item.version)))

    if not versions:
        raise ValueError(f"No numeric versions found for data asset '{name}'.")

    return max(versions, key=lambda entry: entry[0])[1]


def _find_artifact_dir(root: Path, filenames: Iterable[str]) -> Path | None:
    if root.is_dir():
        for filename in filenames:
            if (root / filename).exists():
                return root

    matches: list[Path] = []
    for filename in filenames:
        matches.extend(root.rglob(filename))

    if not matches:
        return None

    matches.sort(key=lambda path: len(path.parts))
    return matches[0].parent


def _has_any_files(root: Path, filenames: Iterable[str]) -> bool:
    return _find_artifact_dir(root, filenames) is not None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_model_dir(download_root: Path) -> Path:
    found = _find_artifact_dir(download_root, MODEL_ARTIFACT_FILES)
    return found or download_root


def _resolve_scaler_dir(download_root: Path) -> Path:
    found = _find_artifact_dir(download_root, SCALER_HINT_FILES)
    if found is not None:
        return found
    # Fall back to first directory containing joblib/pkl files
    for candidate in [download_root, *download_root.glob("*")]:
        if not candidate.is_dir():
            continue
        if any(p.suffix in {".joblib", ".pkl"} for p in candidate.iterdir() if p.is_file()):
            return candidate
    return download_root


def _invoke_download_function(
    fn: object,
    *,
    uri: str,
    download_path: Path,
    ml_client: MLClient,
) -> bool:
    if not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        try:
            fn(uri, str(download_path))
            return True
        except TypeError:
            return False

    params = sig.parameters
    args: list[object] = []
    kwargs: dict[str, object] = {}

    if "operation_scope" in params:
        kwargs["operation_scope"] = getattr(ml_client, "_operation_scope", None)

    if "azureml_uri" in params:
        kwargs["azureml_uri"] = uri
    elif "artifact_uri" in params:
        kwargs["artifact_uri"] = uri
    elif "uri" in params:
        kwargs["uri"] = uri
    elif "path" in params:
        kwargs["path"] = uri
    else:
        args.append(uri)

    if "download_path" in params:
        kwargs["download_path"] = str(download_path)
    elif "destination" in params:
        kwargs["destination"] = str(download_path)
    elif "dest" in params:
        kwargs["dest"] = str(download_path)
    elif "output_path" in params:
        kwargs["output_path"] = str(download_path)
    else:
        args.append(str(download_path))

    try:
        fn(*args, **kwargs)
        return True
    except TypeError:
        return False


def _download_data_asset(
    ml_client: MLClient,
    *,
    name: str,
    version: str,
    download_path: Path,
) -> None:
    """Download a data asset to a local path with SDK-version fallbacks."""
    download_fn = getattr(ml_client.data, "download", None)
    if callable(download_fn):
        download_fn(name=name, version=version, download_path=str(download_path))
        return

    asset = ml_client.data.get(name=name, version=version)
    uri = getattr(asset, "path", None) or getattr(asset, "uri", None)
    if not uri:
        raise ValueError(f"Data asset '{name}:{version}' has no path/uri.")

    artifact_ops = getattr(ml_client, "_artifact_operations", None) or getattr(
        ml_client, "artifact_operations", None
    )
    if artifact_ops is not None:
        download_method = getattr(artifact_ops, "download", None)
        if _invoke_download_function(
            download_method, uri=uri, download_path=download_path, ml_client=ml_client
        ):
            return

    try:
        import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
    except Exception:
        artifact_utils = None

    if artifact_utils is not None:
        candidates = [
            "download_artifact_from_azureml_uri",
            "_download_artifact_from_azureml_uri",
            "download_artifact_from_uri",
            "download_artifact",
        ]
        for name in candidates:
            fn = getattr(artifact_utils, name, None)
            if _invoke_download_function(
                fn, uri=uri, download_path=download_path, ml_client=ml_client
            ):
                return

        for name in dir(artifact_utils):
            if "download" not in name:
                continue
            fn = getattr(artifact_utils, name, None)
            if _invoke_download_function(
                fn, uri=uri, download_path=download_path, ml_client=ml_client
            ):
                return

    if uri.startswith("azureml://"):
        datastore_match = re.search(r"/datastores/([^/]+)/paths/(.+)$", uri)
        if datastore_match:
            datastore_name = datastore_match.group(1)
            blob_path = unquote(datastore_match.group(2))
            _download_from_datastore(
                ml_client,
                datastore_name=datastore_name,
                blob_path=blob_path,
                download_path=download_path,
            )
            return

    raise RuntimeError(
        "Data download not supported by this Azure ML SDK. "
        "Upgrade azure-ai-ml or provide a compatible download utility."
    )


def _download_from_datastore(
    ml_client: MLClient,
    *,
    datastore_name: str,
    blob_path: str,
    download_path: Path,
) -> None:
    try:
        from azure.storage.blob import BlobServiceClient
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("azure-storage-blob is required for datastore downloads.") from exc

    datastore = ml_client.datastores.get(datastore_name)

    def _camel_key(text: str) -> str:
        parts = text.split("_")
        return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])

    def _read_value(*keys: str) -> str | None:
        for key in keys:
            value = getattr(datastore, key, None)
            if value:
                return str(value)
        if hasattr(datastore, "__dict__"):
            for key in keys:
                value = datastore.__dict__.get(key)
                if value:
                    return str(value)
        props = getattr(datastore, "properties", None) or getattr(datastore, "_properties", None)
        if isinstance(props, dict):
            for key in keys:
                value = props.get(key) or props.get(_camel_key(key))
                if value:
                    return str(value)
        for attr in ("_to_dict", "to_dict", "_as_dict", "as_dict"):
            fn = getattr(datastore, attr, None)
            if not callable(fn):
                continue
            try:
                data = fn()
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for key in keys:
                value = data.get(key) or data.get(_camel_key(key))
                if value:
                    return str(value)
        return None

    account_name = _read_value(
        "account_name",
        "account",
        "storage_account_name",
        "storageAccountName",
    )
    container_name = _read_value(
        "container_name",
        "container",
        "containerName",
    )
    endpoint = _read_value(
        "endpoint",
        "account_url",
        "accountUrl",
        "blob_endpoint",
        "blobEndpoint",
    )

    endpoint_str = (endpoint or "").rstrip("/")
    account_url: str | None = None
    if endpoint_str and "blob.core.windows.net" in endpoint_str:
        account_url = endpoint_str
        if not account_name:
            match = re.search(r"https://([^.]+)\\.blob\\.core\\.windows\\.net", endpoint_str)
            if match:
                account_name = match.group(1)

    if not account_name:
        workspace_name = getattr(ml_client, "workspace_name", None) or getattr(
            ml_client, "_workspace_name", None
        )
        if workspace_name:
            try:
                workspace = ml_client.workspaces.get(name=workspace_name)
            except Exception:
                workspace = None
            if workspace is not None:
                storage_id = getattr(workspace, "storage_account", None) or getattr(
                    workspace, "storage_account_id", None
                )
                if storage_id and isinstance(storage_id, str) and "/storageAccounts/" in storage_id:
                    account_name = storage_id.split("/storageAccounts/")[-1]

    if not account_name or not container_name:
        raise RuntimeError(f"Unsupported datastore configuration for '{datastore_name}'.")

    account_url = account_url or f"https://{account_name}.blob.core.windows.net"
    credential = _build_runtime_credential()
    service = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = service.get_container_client(container_name)

    prefix = blob_path.strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    download_path.mkdir(parents=True, exist_ok=True)

    for blob in container_client.list_blobs(name_starts_with=prefix):
        name = getattr(blob, "name", None) or ""
        if not name or name.endswith("/"):
            continue
        rel = name[len(prefix) :] if name.startswith(prefix) else name
        dest = download_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as handle:
            stream = container_client.download_blob(name)
            handle.write(stream.readall())


def ensure_assets_from_env() -> RuntimeAssets:
    """Ensure model + scalers are available locally and set env vars for scoring."""
    settings = get_settings().require_az_id()
    credential = _build_runtime_credential()
    ml_client = get_ml_client(settings=settings, credential=credential)

    model_name = _require_env("MODEL_NAME")
    model_version = _optional_env("MODEL_VERSION")
    scaler_asset_name = _optional_env("SCALER_ASSET_NAME", settings.serving_scalers_name)
    if not scaler_asset_name:
        raise ValueError("SCALER_ASSET_NAME is required (or set serving_scalers_name in settings).")

    scaler_asset_version = _optional_env("SCALER_ASSET_VERSION")

    model_dir = Path(_optional_env("MODEL_DIR", "/app/assets/model") or "/app/assets/model").resolve()
    scaler_dir = Path(_optional_env("SCALER_DIR", "/app/assets/scalers") or "/app/assets/scalers").resolve()

    resolved_model = resolve_model(
        ml_client,
        model_name=model_name,
        model_version=model_version,
    )

    if not _has_any_files(model_dir, MODEL_ARTIFACT_FILES):
        _ensure_dir(model_dir)
        logger.info(
            "Downloading model",
            extra={"model_name": resolved_model.name, "model_version": resolved_model.version},
        )
        ml_client.models.download(
            name=resolved_model.name,
            version=resolved_model.version,
            download_path=str(model_dir),
        )

    resolved_model_dir = _resolve_model_dir(model_dir)

    if not _has_any_files(scaler_dir, SCALER_HINT_FILES):
        _ensure_dir(scaler_dir)
        resolved_scaler_version = scaler_asset_version or _resolve_latest_data_version(
            ml_client, scaler_asset_name
        )
        logger.info(
            "Downloading scalers",
            extra={"scaler_asset_name": scaler_asset_name, "scaler_asset_version": resolved_scaler_version},
        )
        _download_data_asset(
            ml_client,
            name=scaler_asset_name,
            version=resolved_scaler_version,
            download_path=scaler_dir,
        )
        scaler_asset_version = resolved_scaler_version

    resolved_scaler_dir = _resolve_scaler_dir(scaler_dir)

    os.environ["AZUREML_MODEL_DIR"] = str(resolved_model_dir)
    os.environ["SCALER_DIR"] = str(resolved_scaler_dir)

    return RuntimeAssets(
        model_name=resolved_model.name,
        model_version=resolved_model.version,
        model_dir=resolved_model_dir,
        scaler_asset_name=scaler_asset_name,
        scaler_asset_version=str(scaler_asset_version or ""),
        scaler_dir=resolved_scaler_dir,
    )


__all__ = ["RuntimeAssets", "ensure_assets_from_env"]
