"""Serving utilities for fraud detection deployments."""

from fraud_detection.serving.serve_prod_model import ServeResult, serve_prod_model

__all__ = ["ServeResult", "serve_prod_model"]
