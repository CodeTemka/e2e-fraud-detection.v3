"""Utilities for incrementing Azure ML asset versions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError


def _parse_version(value: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _format_version(parts: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in parts)


def next_version(existing_versions: Iterable[str], base_version: str = "1.0") -> str:
    parsed = [_parse_version(value) for value in existing_versions]
    numeric = [value for value in parsed if value is not None]
    if not numeric:
        return base_version
    latest = max(numeric)
    bumped = list(latest)
    bumped[-1] += 1
    return _format_version(tuple(bumped))


def _list_versions(items: Iterable[Any]) -> list[str]:
    versions: list[str] = []
    for item in items:
        version_value = getattr(item, "version", None)
        if version_value:
            versions.append(str(version_value))
    return versions


def resolve_next_environment_version(
    ml_client: MLClient,
    *,
    name: str,
    base_version: str = "1.0",
) -> str:
    try:
        existing = _list_versions(ml_client.environments.list(name=name))
    except ResourceNotFoundError:
        return base_version
    return next_version(existing, base_version=base_version)


def resolve_next_component_version(
    ml_client: MLClient,
    *,
    name: str,
    base_version: str = "1.0",
) -> str:
    try:
        existing = _list_versions(ml_client.components.list(name=name))
    except ResourceNotFoundError:
        return base_version
    return next_version(existing, base_version=base_version)


__all__ = [
    "next_version",
    "resolve_next_environment_version",
    "resolve_next_component_version",
]
