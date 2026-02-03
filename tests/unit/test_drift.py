import pandas as pd

from fraud_detection.data.data_val.data_validate import compute_drift_metrics
from fraud_detection.monitoring.drift import DriftConfig, run_drift_check


def test_compute_drift_metrics_returns_metric_keys():
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    current = pd.DataFrame({"a": [2.0, 3.0, 4.0], "b": [6.0, 7.0, 8.0]})

    metrics = compute_drift_metrics(reference, current, method="psi", bins=3)
    assert "drift.psi.a" in metrics and "drift.psi.b" in metrics


def test_compute_drift_metrics_rejects_unknown_method():
    reference = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    current = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})

    try:
        compute_drift_metrics(reference, current, method="unknown")
    except ValueError as exc:
        assert "Unsupported drift method" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_run_drift_check_writes_outputs(tmp_path, mock_mlflow):
    reference = pd.DataFrame({"Time": [1, 2, 3], "Amount": [10.0, 11.0, 12.0], "Class": [0, 0, 1]})
    current = pd.DataFrame({"Time": [10, 11, 12], "Amount": [20.0, 21.0, 22.0], "Class": [0, 1, 0]})

    config = DriftConfig(psi_threshold=0.0, ks_threshold=0.0, include_ks=True)
    result = run_drift_check(
        reference_data="ref",
        current_data="cur",
        output_dir=tmp_path,
        config=config,
        reference_df=reference,
        current_df=current,
    )

    summary_path = tmp_path / "drift_summary.json"
    assert summary_path.exists() and result.summary["drift_detected"] is True
