from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

logger = logging.getLogger("fraud_detection.scoring")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

ID_COLUMNS = ("id", "transaction_id", "row_id")
LABEL_COLUMN_DEFAULT = "Class"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"
MAX_ALERTS_ENV = "MAX_ALERTS"
ALERT_CAP_ENV = "ALERT_CAP"
SCALER_DIR_ENV = "SCALER_DIR"
DEFAULT_ALERT_CAP = 100

DEFAULT_SCALER_FILES = {
    AMOUNT_COLUMN: (
        "amount_scaler.joblib",
        "amount_scaler.pkl",
        "scaler_amount.joblib",
        "scaler_amount.pkl",
    ),
    TIME_COLUMN: (
        "time_scaler.joblib",
        "time_scaler.pkl",
        "scaler_time.joblib",
        "scaler_time.pkl",
    ),
}


def _ensure_numpy_pickle_compat() -> None:
    """Best-effort compatibility for pickles created with newer NumPy internals."""
    import sys

    try:
        import numpy as np  # noqa: F401
    except Exception:
        return

    if "numpy._core" not in sys.modules:
        try:
            import numpy.core as core
        except Exception:
            core = None
        if core is not None:
            sys.modules["numpy._core"] = core

    aliases = (
        ("numpy._core.multiarray", "numpy.core.multiarray"),
        ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"),
    )
    for old, new in aliases:
        if old in sys.modules:
            continue
        try:
            sys.modules[old] = __import__(new, fromlist=["*"])
        except Exception:
            continue


class ModelAssets:
    def __init__(
        self,
        *,
        model: Any,
        feature_columns: list[str],
        label_column: str,
        id_columns: list[str],
        scalers: dict[str, Any],
        model_kind: str,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.id_columns = id_columns
        self.scalers = scalers
        self.model_kind = model_kind


MODEL_ASSETS: ModelAssets | None = None


MODEL_ARTIFACT_FILES = {
    "MLmodel",
    "model.json",
    "model.ubj",
    "model.txt",
    "model.joblib",
    "model.pkl",
    "model.pickle",
}


def _find_model_dir(root: Path) -> Path | None:
    if root.is_dir():
        for filename in MODEL_ARTIFACT_FILES:
            if (root / filename).exists():
                return root

    matches: list[Path] = []
    for filename in MODEL_ARTIFACT_FILES:
        matches.extend(root.rglob(filename))

    if not matches:
        return None

    matches.sort(key=lambda path: len(path.parts))
    return matches[0].parent


def _resolve_model_dir() -> Path:
    raw = os.environ.get("AZUREML_MODEL_DIR", "").strip() or "."
    resolved = Path(raw).resolve()
    found = _find_model_dir(resolved)
    return found or resolved


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning("metadata.json not found in model dir: %s", model_dir)
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse metadata.json: %s", exc)
        return {}


def _load_mlflow_model(model_dir: Path) -> tuple[Any, str]:
    import mlflow
    import yaml

    mlmodel_path = model_dir / "MLmodel"
    config = yaml.safe_load(mlmodel_path.read_text(encoding="utf-8")) or {}
    flavors = config.get("flavors", {}) if isinstance(config, dict) else {}

    if "sklearn" in flavors:
        try:
            return mlflow.sklearn.load_model(str(model_dir)), "mlflow_sklearn"
        except ModuleNotFoundError as exc:
            logger.warning("Failed to load sklearn flavor; falling back to pyfunc.", extra={"error": str(exc)})
            return mlflow.pyfunc.load_model(str(model_dir)), "mlflow_pyfunc"
    if "xgboost" in flavors:
        return mlflow.xgboost.load_model(str(model_dir)), "mlflow_xgboost"
    if "lightgbm" in flavors:
        return mlflow.lightgbm.load_model(str(model_dir)), "mlflow_lightgbm"

    return mlflow.pyfunc.load_model(str(model_dir)), "mlflow_pyfunc"


def _load_model(model_dir: Path) -> tuple[Any, str]:
    mlmodel_path = model_dir / "MLmodel"
    if mlmodel_path.exists():
        return _load_mlflow_model(model_dir)

    for name in ("model.joblib", "model.pkl", "model.pickle"):
        candidate = model_dir / name
        if candidate.exists():
            return load(candidate), "joblib"

    for name in ("model.json", "model.ubj"):
        candidate = model_dir / name
        if candidate.exists():
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(candidate))
            return booster, "xgboost_booster"

    candidate = model_dir / "model.txt"
    if candidate.exists():
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(candidate))
        return booster, "lightgbm_booster"

    raise FileNotFoundError(f"No supported model artifact found in {model_dir}")


def _resolve_scaler_dir(model_dir: Path) -> Path:
    raw = os.environ.get(SCALER_DIR_ENV, "").strip()
    if not raw:
        return model_dir
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def _scaler_candidates_from_metadata(metadata: dict[str, Any]) -> dict[str, list[str]]:
    mapping = metadata.get("scalers")
    if not isinstance(mapping, dict):
        return {}
    candidates: dict[str, list[str]] = {}
    for key, value in mapping.items():
        if not value:
            continue
        column = str(key)
        if column.lower() == "amount":
            column = AMOUNT_COLUMN
        elif column.lower() == "time":
            column = TIME_COLUMN
        candidates[column] = [str(value)]
    return candidates


def _load_scalers(scaler_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if not scaler_dir.exists():
        raise FileNotFoundError(f"Scaler directory not found: {scaler_dir}")

    scalers: dict[str, Any] = {}
    for column, filenames in _scaler_candidates_from_metadata(metadata).items():
        for filename in filenames:
            path = scaler_dir / filename
            if path.exists():
                scalers[column] = load(path)
                break

    for column, filenames in DEFAULT_SCALER_FILES.items():
        if column in scalers:
            continue
        for filename in filenames:
            path = scaler_dir / filename
            if path.exists():
                scalers[column] = load(path)
                break

    return scalers


def _predict_proba(model: Any, features: pd.DataFrame) -> np.ndarray:
    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    if xgb is not None and isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(features)
        preds = model.predict(dmatrix)
        return np.asarray(preds, dtype=float).reshape(-1)

    try:
        import lightgbm as lgb
    except Exception:
        lgb = None

    if lgb is not None and isinstance(model, lgb.Booster):
        preds = model.predict(features)
        return np.asarray(preds, dtype=float).reshape(-1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        arr = np.asarray(proba, dtype=float)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return arr[:, 1]
        return arr.reshape(-1)

    preds = model.predict(features)
    if isinstance(preds, pd.DataFrame):
        for key in ("probability", "score", "prediction", "predictions"):
            if key in preds.columns:
                return preds[key].to_numpy(dtype=float)
        if preds.shape[1] == 1:
            return preds.iloc[:, 0].to_numpy(dtype=float)
    if isinstance(preds, pd.Series):
        return preds.to_numpy(dtype=float)
    arr = np.asarray(preds, dtype=float)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.reshape(-1)


def _parse_max_alerts(raw_data: dict[str, Any] | None) -> int | None:
    if not raw_data:
        return None
    for key in ("max_alerts", "alert_cap"):
        value = raw_data.get(key)
        if value is None:
            continue
        try:
            return max(int(value), 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key} '{value}'. Must be an integer.") from exc
    return None


def _parse_request_id(raw_data: Any) -> str:
    if raw_data is None:
        return str(uuid.uuid4())

    parsed = raw_data
    if isinstance(raw_data, (str, bytes)):
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError:
            return str(uuid.uuid4())

    if isinstance(parsed, dict):
        for key in ("request_id", "requestId", "correlation_id", "correlationId"):
            value = parsed.get(key)
            if value:
                return str(value)
    return str(uuid.uuid4())


def _records_to_df(data: Any, columns: list[str] | None = None) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        if any(isinstance(value, list) for value in data.values()):
            return pd.DataFrame(data)
        return pd.DataFrame([data])
    if isinstance(data, list):
        if not data:
            return pd.DataFrame(columns=columns)
        if isinstance(data[0], dict):
            return pd.DataFrame(data)
        if columns:
            return pd.DataFrame(data, columns=columns)
    raise ValueError("Input data must be a dict, list of dicts, or list of lists with columns.")


def _coerce_records(raw_data: Any) -> tuple[pd.DataFrame, int | None]:
    if raw_data is None:
        return pd.DataFrame(), None

    parsed = raw_data
    if isinstance(raw_data, (str, bytes)):
        parsed = json.loads(raw_data)

    if isinstance(parsed, dict):
        if "input_data" in parsed:
            payload = parsed.get("input_data") or {}
            columns = payload.get("columns") or parsed.get("columns") or []
            data_rows = payload.get("data") or []
            df = _records_to_df(data_rows, columns if isinstance(columns, list) else None)
            return df, _parse_max_alerts(parsed) or _parse_max_alerts(payload)
        if "data" in parsed and "columns" in parsed:
            columns = parsed.get("columns")
            df = _records_to_df(parsed.get("data"), columns if isinstance(columns, list) else None)
            return df, _parse_max_alerts(parsed)
        if "data" in parsed:
            df = _records_to_df(parsed.get("data"))
            return df, _parse_max_alerts(parsed)
        return _records_to_df(parsed), _parse_max_alerts(parsed)

    if isinstance(parsed, list):
        return _records_to_df(parsed), None

    raise ValueError("Input data must be JSON serializable and contain records.")


def _extract_row_ids(df: pd.DataFrame, id_columns: list[str]) -> tuple[pd.DataFrame, list[Any] | None]:
    for col in id_columns:
        if col in df.columns:
            row_ids = df[col].tolist()
            return df.drop(columns=[col]), row_ids
    return df, None


def _apply_scalers(df: pd.DataFrame, scalers: dict[str, Any]) -> pd.DataFrame:
    transformed = df.copy()
    for required in (AMOUNT_COLUMN, TIME_COLUMN):
        if required in transformed.columns and required not in scalers:
            raise ValueError(f"Scaler for column '{required}' not found.")

    for column, scaler in scalers.items():
        if column in transformed.columns:
            values = scaler.transform(transformed[[column]])
            transformed[column] = np.asarray(values).reshape(-1)
    return transformed


def _prepare_features(df: pd.DataFrame, assets: ModelAssets) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    if assets.label_column in df.columns:
        df = df.drop(columns=[assets.label_column])

    df = _apply_scalers(df, assets.scalers)

    feature_columns = assets.feature_columns or list(df.columns)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[feature_columns]


def apply_top_k_threshold(
    probabilities: np.ndarray, max_alerts: int
) -> tuple[list[int], float, int, list[int]]:
    if probabilities.size == 0:
        return [], 1.0, 0, []

    n = probabilities.size
    k = min(max(int(max_alerts), 0), n)
    order = np.argsort(-probabilities, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    if k == 0:
        return [0] * n, 1.0, 0, ranks.tolist()

    top_indices = order[:k]
    preds = np.zeros(n, dtype=int)
    preds[top_indices] = 1
    threshold = float(np.min(probabilities[top_indices]))
    return preds.tolist(), threshold, k, ranks.tolist()


def _resolve_default_alert_cap() -> int:
    for key in (MAX_ALERTS_ENV, ALERT_CAP_ENV):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return max(int(raw), 0)
        except (TypeError, ValueError):
            logger.warning("Invalid %s env var '%s'; using default.", key, raw)
    return DEFAULT_ALERT_CAP


def init() -> None:
    global MODEL_ASSETS
    global DEFAULT_ALERT_CAP

    _ensure_numpy_pickle_compat()

    model_dir = _resolve_model_dir()
    metadata = _load_metadata(model_dir)
    model, model_kind = _load_model(model_dir)
    scaler_dir = _resolve_scaler_dir(model_dir)
    scalers = _load_scalers(scaler_dir, metadata)

    feature_columns = list(
        metadata.get("features")
        or metadata.get("feature_columns")
        or metadata.get("feature_list")
        or []
    )
    label_column = str(metadata.get("label_column") or LABEL_COLUMN_DEFAULT)
    id_columns = list(metadata.get("id_columns") or ID_COLUMNS)

    MODEL_ASSETS = ModelAssets(
        model=model,
        feature_columns=feature_columns,
        label_column=label_column,
        id_columns=id_columns,
        scalers=scalers,
        model_kind=model_kind,
    )

    DEFAULT_ALERT_CAP = _resolve_default_alert_cap()

    logger.info(
        "Model assets loaded",
        extra={
            "model_kind": MODEL_ASSETS.model_kind,
            "feature_count": len(MODEL_ASSETS.feature_columns),
            "label_column": MODEL_ASSETS.label_column,
            "scaler_dir": str(scaler_dir),
            "default_max_alerts": DEFAULT_ALERT_CAP,
        },
    )


def run(raw_data: Any) -> dict[str, Any]:
    if MODEL_ASSETS is None:
        raise RuntimeError("Model assets not initialized. Call init() first.")

    start_time = time.perf_counter()
    request_id = _parse_request_id(raw_data)
    df, max_alerts_override = _coerce_records(raw_data)
    df, row_ids = _extract_row_ids(df, MODEL_ASSETS.id_columns)
    features = _prepare_features(df, MODEL_ASSETS)

    if features.empty:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "inference_empty",
            extra={
                "request_id": request_id,
                "record_count": len(df),
                "max_alerts": 0,
                "duration_ms": duration_ms,
            },
        )
        response = {
            "predicted_label": [],
            "probability": [],
            "threshold_used": 1.0,
            "max_alerts": 0,
            "num_alerts": 0,
            "rank": [],
        }
        if row_ids is not None:
            response["row_ids"] = row_ids
        return response

    max_alerts = max_alerts_override if max_alerts_override is not None else DEFAULT_ALERT_CAP
    probabilities = _predict_proba(MODEL_ASSETS.model, features)
    predictions, threshold, num_alerts, ranks = apply_top_k_threshold(probabilities, max_alerts)

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "inference",
        extra={
            "request_id": request_id,
            "record_count": len(df),
            "max_alerts": max_alerts,
            "num_alerts": num_alerts,
            "threshold": threshold,
            "duration_ms": duration_ms,
        },
    )

    response = {
        "predicted_label": predictions,
        "probability": probabilities.tolist(),
        "threshold_used": threshold,
        "max_alerts": max_alerts,
        "num_alerts": num_alerts,
        "rank": ranks,
        "predictions": predictions,
        "probabilities": probabilities.tolist(),
    }
    if row_ids is not None:
        response["row_ids"] = row_ids
    return response
