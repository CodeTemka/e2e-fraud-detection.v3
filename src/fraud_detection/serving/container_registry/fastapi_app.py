"""FastAPI wrapper for fraud detection scoring."""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException

from .. import score
from .container_app_runtime import ensure_assets_from_env
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Fraud Detection Scoring API", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        assets = ensure_assets_from_env()
        score.init()
        app.state.runtime_assets = assets
        logger.info("FastAPI startup complete", extra={"assets": assets.to_dict()})
    except Exception as exc:
        logger.exception("FastAPI startup failed", extra={"error": str(exc)})
        raise


@app.get("/health")
def health() -> dict[str, Any]:
    assets = getattr(app.state, "runtime_assets", None)
    return {
        "status": "ok",
        "model_loaded": score.MODEL_ASSETS is not None,
        "runtime_assets": assets.to_dict() if assets else None,
    }


@app.post("/score")
def score_endpoint(payload: Any = Body(...)) -> dict[str, Any]:
    try:
        return score.run(payload)
    except Exception as exc:
        logger.warning("Scoring request failed", extra={"error": str(exc)})
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["app"]
