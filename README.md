# End-to-End Credit Card Fraud Detection (Azure ML MLOps)

Production-grade MLOps workflow for credit card fraud detection on Azure ML (v2). This repo provides CLI-driven pipelines for data validation, training, model promotion, deployment, and post-deployment monitoring with CI/CD automation.

## Key Features
- Typer-based CLI (`fraud-cli`) to orchestrate end-to-end workflows
- Data validation and preparation (train/test split, robust scaling)
- Training pipelines: AutoML, XGBoost sweep, LightGBM sweep
- Model promotion based on MLflow metrics
- Managed online endpoint deployment on Azure ML
- Post-deployment monitoring: evaluation + drift detection (PSI/KS)
- CI (lint/test/security) and CD (manual/tagged deploy) with GitHub Actions

## Architecture / Workflow Overview
High-level flow:

```
Data -> Validate -> Prep -> Train -> Promote -> Deploy -> Monitor
```

Key components:
- Azure ML data assets (registered datasets, scalers)
- Azure ML jobs/components (validation, prep, train, deploy)
- MLflow tracking for experiments, metrics, and artifacts
- Azure ML managed online endpoints for production serving

## Tech Stack
- Python 3.12
- Azure ML v2 (azure-ai-ml)
- MLflow (tracking + artifacts)
- scikit-learn, LightGBM, XGBoost
- Typer CLI
- GitHub Actions (CI/CD)

## Repository Structure

```
.
+- .github/workflows/          # CI/CD workflows
+- scripts/                    # helper scripts (Azure setup, secrets, data registration)
+- src/fraud_detection/        # Python package
¦  +- data/                    # data validation + preparation
¦  +- training/                # AutoML, XGB, LGBM jobs
¦  +- registry/                # model promotion + registration
¦  +- serving/                 # deployment + scoring utils
¦  +- monitoring/              # evaluation + drift monitoring
¦  +- pipeline/                # Azure ML pipelines
+- tests/                      # pytest suite
+- pyproject.toml
```

## Prerequisites
- Python 3.12
- Azure subscription + Azure ML workspace
- Azure CLI (`az`) and authentication

## Quickstart (Local)

```bash
# create venv
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell

# install
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# verify CLI
fraud-cli --help
```

## Azure Setup (High Level)

Environment variables (names only; set via `.env` or shell):
- `SUBSCRIPTION_ID`
- `RESOURCE_GROUP`
- `WORKSPACE_NAME`
- `LOCATION`
- `REGISTERED_DATASET_NAME`
- `REGISTERED_DATASET_LABEL`
- `REGISTERED_TRAIN`
- `REGISTERED_TEST`
- `SERVING_SCALERS_NAME`
- `PROD_MODEL_NAME`
- `ENDPOINT_NAME`
- `DEPLOYMENT_NAME`
- `MONITOR_REFERENCE_DATA`
- `MONITOR_ALERT_RATE`

Local auth:
- `az login` (device code)

CI/CD auth:
- Service principal via GitHub Secrets:
  - `AZURE_CLIENT_ID`
  - `AZURE_TENANT_ID`
  - `AZURE_CLIENT_SECRET`
  - `AZURE_SUBSCRIPTION_ID`

## Data Registration

1) Download the Kaggle "Credit Card Fraud" dataset.
2) Place files under `data/` and create an MLTable definition (recommended).
3) Register the data asset in Azure ML.

Example (helper script):

```bash
python scripts/register_data_az.py
```

## Pipelines & Commands (Copy-Ready)

All commands use defaults from settings unless overridden.

Validate dataset:
```bash
fraud-cli validate-data --dataset-name original-creditcard --validation-report outputs/validation.json
```

Prepare data (train/test split + scalers):
```bash
fraud-cli prep-data-for-train --data-name original-creditcard --scalers outputs/scalers --metadata outputs/metadata.json
```

Or run the data pipeline:
```bash
fraud-cli run-data-pipeline --data-name original-creditcard --wait
```

Train AutoML:
```bash
fraud-cli train-automl --metric average_precision_score_weighted --wait
```

Train XGBoost sweep:
```bash
fraud-cli train-xgb --metric average_precision_score_macro --val-size 0.2 --wait
```

Train LightGBM sweep:
```bash
fraud-cli train-lgbm --metric average_precision_score_macro --val-size 0.2 --wait
```

Promote best model:
```bash
fraud-cli promote-prod-model --compare-metric average_precision_score_macro --production-info outputs/promotion.json
```

Deployment pipeline:
```bash
fraud-cli run-deployment-pipeline --wait
```

Direct serving (optional):
```bash
fraud-cli serve-prod-model --scaler-dir azureml:fraud-detection-scalers:1 --wait
```

Evaluate endpoint:
```bash
fraud-cli evaluate-endpoint --test-data fraud-detection-test-data --out outputs/monitoring --wait
```

Check drift:
```bash
fraud-cli check-drift --test-data fraud-detection-test-data --ref-data original-creditcard --out outputs/monitoring --wait
```

Run monitoring (evaluation + drift):
```bash
fraud-cli monitor --mode monitor --submit --wait
```

## Monitoring & Drift
- Evaluation metrics: accuracy, precision, recall, F1, ROC AUC, PR AUC
- Drift checks: PSI and KS statistics
- Artifacts:
  - MLflow metrics + artifacts
  - Local outputs under `outputs/monitoring/` (e.g., `metrics.json`, confusion matrix)

## CI/CD
- CI (`.github/workflows/ci.yml`): Ruff + Black, pytest (unit), pip-audit
- CD (`.github/workflows/cd.yml`): manual deploy + monitor, plus tag trigger `v*.*.*`

## Testing

```bash
pytest -q -m "not integration"
```

## Troubleshooting
- Azure auth: run `az login` and verify `az account show`.
- Provider registration: `fraud-cli check-providers --subscription-id <id>`.
- Endpoint key missing: set `ENDPOINT_KEY` or `AZUREML_ENDPOINT_KEY` for local calls.
- Compute quota errors: reduce max instances or delete unused clusters.
- MLflow tracking issues: ensure workspace is configured and accessible.

## Roadmap
- Add model explainability reports
- Expand monitoring to data quality + bias checks
- Automated model retraining triggers

## License
TBD

## Acknowledgements
- Kaggle "Credit Card Fraud" dataset
- Azure Machine Learning and MLflow
