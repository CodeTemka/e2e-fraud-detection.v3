"""Training script for LightGBM."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import mlflow
import mltable
import typer
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fraud_detection.azure.client import get_ml_client
from fraud_detection.config import get_git_sha
from fraud_detection.training.automl import resolve_latest_data_version
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def main(
    train_data: Annotated[str, typer.Option("--train_data", help="Registered path to training data.")] = ...,
    test_data: Annotated[str, typer.Option("--test_data", help="Registered path to test data.")] = ...,
    label_col: Annotated[str, typer.Option("--label_col", help="Label column name in the CSV.")] = "Class",
    val_size: Annotated[float, typer.Option("--val_size", help="fraction of TRAIN split used for validation")] = 0.2,
    n_estimators: Annotated[int, typer.Option("--n_estimators")] = 600,
    max_depth: Annotated[int, typer.Option("--max_depth")] = 5,
    num_leaves: Annotated[int, typer.Option("--num_leaves")] = 31,
    learning_rate: Annotated[float, typer.Option("--learning_rate")] = 0.05,
    subsample: Annotated[float, typer.Option("--subsample")] = 0.8,
    colsample_bytree: Annotated[float, typer.Option("--colsample_bytree")] = 0.8,
    min_child_weight: Annotated[float, typer.Option("--min_child_weight")] = 1.0,
    min_child_samples: Annotated[int, typer.Option("--min_child_samples")] = 20,
    reg_alpha: Annotated[float, typer.Option("--reg_alpha")] = 0.0,
    reg_lambda: Annotated[float, typer.Option("--reg_lambda")] = 1.0,
    random_state: Annotated[int, typer.Option("--random_state")] = 42,
    dataset_version: Annotated[str | None, typer.Option("--dataset_version")] = None,
    early_stopping_rounds: Annotated[int, typer.Option("--early_stopping_rounds")] = 50,
    output_dir: Annotated[str, typer.Option("--output_dir")] = "outputs",
    model_name: Annotated[str, typer.Option("--model_name", help="Name to register the model under.")] = "lightgbm-fraud-model",
) -> None:
    ml_client = get_ml_client()
    resolved_version = dataset_version or resolve_latest_data_version(ml_client, train_data)
    resolved_test_version = resolve_latest_data_version(ml_client, test_data)

    data = ml_client.data.get(name=train_data, version=resolved_version)
    df = mltable.load(data.path).to_pandas_dataframe()

    test = ml_client.data.get(name=test_data, version=resolved_test_version)
    test_df = mltable.load(test.path).to_pandas_dataframe()

    X_train, X_val, y_train, y_val = train_test_split(
        df.drop(columns=[label_col]),
        df[label_col],
        test_size=val_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary",
        random_state=random_state,
    )

    # Azure ML sweep expects the primary metric name to match EXACTLY what you log.
    if mlflow.active_run() is None:
        mlflow.start_run()

    mlflow.set_tag("git_sha", get_git_sha())
    mlflow.set_tag("dataset_version", resolved_version)

    mlflow.log_params(
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "min_child_samples": min_child_samples,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
        }
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        verbose=False,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
    )

    proba_test = model.predict_proba(test_df.drop(columns=[label_col]))[:, 1]
    ap = average_precision_score(test_df[label_col], proba_test)
    auc = roc_auc_score(test_df[label_col], proba_test)

    # Important: primary_metric must match the string you set in sweep_job.primary_metric.
    # Logging the metric the same as the AutoML default metrics, which will help later for choosing best model.
    mlflow.log_metric("average_precision_score_macro", ap)
    mlflow.log_metric("AUC_macro", auc)

    model_dir = Path(output_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    ml_client.models.create_or_update(
        Model(
            name=model_name,
            path=str(model_dir),
            type=AssetTypes.CUSTOM_MODEL,
        )
    )

    mlflow.end_run()

    logger.info(
        "Training complete",
        extra={
            "average_precision": ap,
            "roc_auc": auc,
            "model_name": model_name,
        },
    )


if __name__ == "__main__":
    app()
