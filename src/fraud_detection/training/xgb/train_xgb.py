"""Training script for XGBoost (sweep-friendly, no test_data, WITH early stopping).

- Uses internal train/val split from the provided training dataset.
- Trains with early stopping using the validation split.
- Logs validation metrics to MLflow (used by sweep primary_metric).
- Writes a deployable model folder to --output_dir/model:
  - model.json
  - feature_columns.json
  - metadata.json
- DOES NOT register the model (avoid registering hundreds of models in sweeps).
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Annotated

import mlflow
import mltable
import typer
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_git_sha
from fraud_detection.training.automl import resolve_latest_data_version
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer()


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _train_with_early_stopping(
    model: XGBClassifier,
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    feature_columns: list[str],
    early_stopping_rounds: int,
    n_estimators: int,
    random_state: int,
):
    """Train with early stopping across XGBoost sklearn API changes."""
    fit_sig = inspect.signature(model.fit)
    fit_kwargs = {
        "eval_set": [(X_val[feature_columns], y_val)],
        "verbose": False,
    }

    if "early_stopping_rounds" in fit_sig.parameters:
        model.fit(
            X_train[feature_columns],
            y_train,
            **fit_kwargs,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model, "sklearn"

    if "callbacks" in fit_sig.parameters:
        model.fit(
            X_train[feature_columns],
            y_train,
            **fit_kwargs,
            callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
        )
        return model, "sklearn"

    # Fall back to xgboost.train when sklearn fit doesn't support early stopping.
    params = model.get_xgb_params()
    if "random_state" in params and "seed" not in params:
        params["seed"] = params.pop("random_state")
    params.setdefault("seed", random_state)

    dtrain = xgb.DMatrix(X_train[feature_columns], label=y_train.astype(int))
    dval = xgb.DMatrix(X_val[feature_columns], label=y_val.astype(int))

    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
    except TypeError:
        logger.warning("Early stopping not supported in this XGBoost version; training without it.")
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "validation")],
            verbose_eval=False,
        )

    return booster, "booster"


@app.command()
def main(
    train_data: Annotated[str, typer.Option("--train_data", help="Registered name of training data asset.")] = ...,
    label_col: Annotated[str, typer.Option("--label_col", help="Label column name.")] = "Class",
    val_size: Annotated[float, typer.Option("--val_size", help="Fraction of data used for validation")] = 0.2,
    n_estimators: Annotated[int, typer.Option("--n_estimators", help="Max trees (upper bound).")] = 2000,
    max_depth: Annotated[int, typer.Option("--max_depth")] = 5,
    learning_rate: Annotated[float, typer.Option("--learning_rate")] = 0.05,
    subsample: Annotated[float, typer.Option("--subsample")] = 0.8,
    colsample_bytree: Annotated[float, typer.Option("--colsample_bytree")] = 0.8,
    min_child_weight: Annotated[float, typer.Option("--min_child_weight")] = 1.0,
    gamma: Annotated[float, typer.Option("--gamma")] = 0.0,
    reg_lambda: Annotated[float, typer.Option("--reg_lambda")] = 1.0,
    random_state: Annotated[int, typer.Option("--random_state")] = 42,
    early_stopping_rounds: Annotated[
        int, typer.Option("--early_stopping_rounds", help="Stop if no improvement for N rounds.")
    ] = 50,
    dataset_version: Annotated[str | None, typer.Option("--dataset_version")] = None,
    output_dir: Annotated[str, typer.Option("--output_dir", help="Output folder (URI_FOLDER in Azure ML).")] = "outputs",
) -> None:
    if not (0.0 < val_size < 1.0):
        raise ValueError("--val_size must be between 0 and 1 (exclusive).")
    if early_stopping_rounds < 1:
        raise ValueError("--early_stopping_rounds must be >= 1.")

    ml_client = get_ml_client()
    resolved_train_version = dataset_version or resolve_latest_data_version(ml_client, train_data)

    train_asset = ml_client.data.get(name=train_data, version=resolved_train_version)
    df = mltable.load(train_asset.path).to_pandas_dataframe()

    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in training data columns.")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )

    feature_columns = list(X_train.columns)

    # Compute imbalance weight based on training split only
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # NOTE:
    # We keep eval_metric="aucpr" because it aligns with PR focus and works well for early stopping.
    # Your sweep primary_metric is still based on our own computed AP/AUC on the val split.
    model = XGBClassifier(
        n_estimators=n_estimators,  # upper bound, early stopping will cut it down
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_state,
    )

    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"

    started_run = False
    if mlflow.active_run() is None:
        mlflow.start_run()
        started_run = True

    try:
        mlflow.set_tag("git_sha", get_git_sha())
        mlflow.set_tag("train_dataset_version", resolved_train_version)
        mlflow.set_tag("label_col", label_col)

        mlflow.log_params(
            {
                "n_estimators_max": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "reg_lambda": reg_lambda,
                "scale_pos_weight": scale_pos_weight,
                "random_state": random_state,
                "val_size": val_size,
                "early_stopping_rounds": early_stopping_rounds,
            }
        )

        # Train with early stopping on the validation set.
        trained_model, train_backend = _train_with_early_stopping(
            model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_columns=feature_columns,
            early_stopping_rounds=early_stopping_rounds,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        # best_iteration is 0-based (number of boosting rounds - 1)
        raw_best_iter = getattr(trained_model, "best_iteration", -1)
        try:
            best_iter = int(raw_best_iter if raw_best_iter is not None else -1)
        except (TypeError, ValueError):
            best_iter = -1
        # best_score is model.best_score (depends on eval_metric)
        best_score = getattr(trained_model, "best_score", None)

        # Compute the sweep metrics on validation data
        if train_backend == "booster":
            dval = xgb.DMatrix(X_val[feature_columns])
            proba_val = trained_model.predict(dval)
        else:
            proba_val = trained_model.predict_proba(X_val[feature_columns])[:, 1]
        ap = average_precision_score(y_val, proba_val)
        auc = roc_auc_score(y_val, proba_val)

        # These MUST match sweep primary_metric strings
        mlflow.log_metric("average_precision_score_macro", float(ap))
        mlflow.log_metric("AUC_macro", float(auc))

        # Also log early stopping info
        if best_iter >= 0:
            # Convert to "number of trees used"
            mlflow.log_metric("best_iteration", float(best_iter))
            mlflow.set_tag("best_iteration", str(best_iter))
        if best_score is not None:
            # This is the best value of eval_metric="aucpr" during training
            try:
                mlflow.log_metric("best_eval_aucpr", float(best_score))
            except Exception:
                # Some versions store best_score in a non-float format; keep as tag if needed
                mlflow.set_tag("best_eval_aucpr", str(best_score))

        # ---- Write deployable artifact folder ----
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        model_dir = out_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.json"
        trained_model.save_model(str(model_path))

        _write_json(model_dir / "feature_columns.json", {"feature_columns": feature_columns})

        _write_json(
            model_dir / "metadata.json",
            {
                "framework": "xgboost",
                "model_format": "xgb_json",
                "label_col": label_col,
                "train_dataset_version": resolved_train_version,
                "git_sha": get_git_sha(),
                "early_stopping_rounds": early_stopping_rounds,
                "best_iteration": best_iter,
                "best_eval_aucpr": best_score,
                "metrics": {
                    "average_precision_score_macro": float(ap),
                    "AUC_macro": float(auc),
                },
            },
        )

        # Optional: store artifacts inside the MLflow run for traceability
        try:
            mlflow.log_artifacts(str(model_dir), artifact_path="exported_model")
        except Exception as exc:
            logger.warning(
                "Failed to log model artifacts to MLflow; continuing without artifact logging.",
                extra={"error": str(exc)},
            )

        logger.info(
            "Training complete",
            extra={
                "val_average_precision": float(ap),
                "val_roc_auc": float(auc),
                "best_iteration": best_iter,
                "output_model_dir": str(model_dir),
            },
        )

    finally:
        if started_run:
            mlflow.end_run()


if __name__ == "__main__":
    app()
