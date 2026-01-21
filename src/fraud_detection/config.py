"""Configuration utilities for the fraud detection project."""
from __future__ import annotations

import hashlib
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / '.env',
        env_file_encoding="utf-8",
        extra='ignore',
        loc_by_alias=False,
    )

    subscription_id: str | None = Field(default=None, validation_alias=AliasChoices("SUBSCRIPTION_ID"))
    resource_group: str = Field(default='fraud-detection-rg')
    workspace_name: str = Field(default='e2e-fraud-detection-ws')
    location: str = Field(default='eastasia')

    # DEFAULT CONFIGS
    # Compute settings is configured considering 20 quota limit.
    training_compute_cluster_name: str = Field(default="training-cluster")
    training_compute_cluster_type: str = Field(default="Standard_D2s_v3")
    training_compute_cluster_node_max_count: int = Field(default=3)
    compute_idle_time_before_scale_down: int = Field(default=200)
    deployment_instance_type: str = Field(default="Standard_D2a_v4")
    deployment_instance_count: int = Field(default=1)

    # Dataset settings
    registered_dataset_name: str = Field(default="original-creditcard")
    local_data_path: Path = Field(default=ROOT_DIR / "data")
    
    # Metric settings
    default_metric_automl_train: str = Field(default="average_precision_score_weighted")
    default_metric_serving: str = Field(default="average_precision_score_macro")

    # Experiment settings
    automl_train_exp: str = Field(default="automl-exp")
    custom_train_exp: str = Field(default="custom-train-exp")

    # Model and deployment settings
    prod_model_name: str = Field(default="fraud-detection-model")
    endpoint_name: str = Field(default="fraud-detection-app")
    deployment_name: str = Field(default='blue')

    # Github settings
    github_owner: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_OWNER"))
    github_repo: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_REPO"))
    github_token: str | None = Field(default=None, validation_alias=AliasChoices("GITHUB_TOKEN"))


    def require_az_id(self) -> Settings:
        missing = [
            name
            for name, value in {
                "SUBSCRIPTION_ID": self.subscription_id,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required Azure subscription id {missing}")
        return self


    def require_github(self) -> Settings:
        missing = [
            name 
            for name, value in {
                "GITHUB_OWNER": self.github_owner,
                "GITHUB_REPO": self.github_repo,
                "GITHUB_TOKEN": self.github_token,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required Github settings: {', '.join(missing)}")
        return self
    

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def build_idempotency_key(*parts: str, git_sha: str | None = None) -> str:
    pieces = [p.strip() for p in parts if p]
    if git_sha:
        pieces.append(git_sha)
    digest = hashlib.sha1("::".join(pieces).encode("utf-8")).hexdigest()
    return digest[:7]


def build_job_name(prefix: str, key: str) -> str:
    return f"{prefix}-{key}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or 'job'


def preflight_validate_training_data(path: str) -> bool:
    if path.startswith("azureml:"):
        return True
    else:
        return False


def get_git_sha(short: bool = True) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    

__all__ = [
    "ROOT_DIR",
    "Settings",
    "build_job_name",
    "build_idempotency_key",
    "get_git_sha",
    "get_settings",
    "preflight_validate_training_data",
    "slugify",
]
    
