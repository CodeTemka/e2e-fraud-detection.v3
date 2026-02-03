"""Load tabular data for monitoring (MLTable/CSV/Parquet)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import mltable
import pandas as pd
from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

_CSV_EXTENSIONS = {".csv"}
_PARQUET_EXTENSIONS = {".parquet", ".pq", ".parq"}


def _is_mltable_dir(path: Path) -> bool:
    return (path / "MLTable").exists()


def _read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in _CSV_EXTENSIONS:
        return pd.read_csv(path)
    if suffix in _PARQUET_EXTENSIONS:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path}")


def _collect_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    files = [path for path in files if path.is_file()]
    files.sort()
    return files


def load_dataframe_from_path(path: Path, *, sample_rows: int | None = None) -> pd.DataFrame:
    """Load a dataframe from a local path, supporting MLTable/CSV/Parquet."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Data path not found: {resolved}")

    if resolved.is_file():
        df = _read_tabular_file(resolved)
        return df.head(sample_rows) if sample_rows else df

    if _is_mltable_dir(resolved):
        table = mltable.load(str(resolved))
        if sample_rows:
            table = table.take(sample_rows)
        return table.to_pandas_dataframe()

    files = _collect_files(resolved, _CSV_EXTENSIONS | _PARQUET_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet files found under {resolved}")

    frames = [_read_tabular_file(file) for file in files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return df.head(sample_rows) if sample_rows else df


def _parse_azureml_short_uri(uri: str) -> tuple[str, str] | None:
    if not uri.startswith("azureml:"):
        return None
    parts = uri.split(":")
    if len(parts) >= 3 and parts[1] and parts[2]:
        return parts[1], parts[2]
    return None


def _download_asset(
    ml_client: MLClient,
    *,
    name: str,
    version: str,
    download_path: Path,
) -> Path:
    download_path.mkdir(parents=True, exist_ok=True)
    download_fn = getattr(ml_client.data, "download", None)
    if not callable(download_fn):
        raise RuntimeError("Azure ML SDK does not support data downloads; upgrade azure-ai-ml.")
    download_fn(name=name, version=version, download_path=str(download_path))
    return download_path


def load_dataframe(
    source: str | Path,
    *,
    ml_client: MLClient | None = None,
    data_label: str | None = None,
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """Load a dataframe from a local path or Azure ML data asset."""
    if isinstance(source, Path):
        return load_dataframe_from_path(source, sample_rows=sample_rows)

    source = str(source)
    path = Path(source)
    if path.exists():
        return load_dataframe_from_path(path, sample_rows=sample_rows)

    if source.startswith("azureml://"):
        try:
            table = mltable.load(source)
            if sample_rows:
                table = table.take(sample_rows)
            return table.to_pandas_dataframe()
        except Exception as exc:
            logger.warning("Failed to load MLTable from azureml URI; falling back to download", extra={"error": str(exc)})

    ml_client = ml_client or get_ml_client()
    settings = get_settings()
    resolved_label = data_label or settings.registered_dataset_label

    asset = None
    parsed = _parse_azureml_short_uri(source)
    try:
        if parsed is not None:
            name, version = parsed
            asset = ml_client.data.get(name=name, version=version)
        else:
            asset = ml_client.data.get(name=source, label=resolved_label)
    except ResourceNotFoundError as exc:
        raise FileNotFoundError(f"Unable to resolve data asset: {source}") from exc

    asset_path = getattr(asset, "path", None) or ""
    if asset_path:
        try:
            table = mltable.load(asset_path)
            if sample_rows:
                table = table.take(sample_rows)
            return table.to_pandas_dataframe()
        except Exception as exc:
            logger.warning(
                "Failed to load MLTable from asset path; downloading instead.",
                extra={"asset": f"{asset.name}:{asset.version}", "error": str(exc)},
            )

    tmp_dir = Path(tempfile.mkdtemp(prefix="fraud-data-"))
    _download_asset(ml_client, name=str(asset.name), version=str(asset.version), download_path=tmp_dir)
    return load_dataframe_from_path(tmp_dir, sample_rows=sample_rows)


__all__ = ["load_dataframe", "load_dataframe_from_path"]
