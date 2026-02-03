import json
from types import SimpleNamespace

from fraud_detection.data.data_for_train.data_for_train import (
    fit_scalers,
    read_is_valid_flag,
    save_outputs,
    split_data,
    transform_with_scalers,
)


def test_read_is_valid_flag_handles_true_and_false(tmp_path):
    true_path = tmp_path / "true.flag"
    false_path = tmp_path / "false.flag"
    true_path.write_text("true", encoding="utf-8")
    false_path.write_text("false", encoding="utf-8")

    assert read_is_valid_flag(true_path)[0] is True and read_is_valid_flag(false_path)[0] is False


def test_read_is_valid_flag_rejects_unknown_value(tmp_path):
    bad_path = tmp_path / "bad.flag"
    bad_path.write_text("maybe", encoding="utf-8")

    assert read_is_valid_flag(bad_path)[0] is False


def test_split_data_returns_empty_test_set_when_ratio_zero(sample_train_data):
    train_df, test_df = split_data(sample_train_data, test=0.0, stratify=False)
    assert test_df.empty is True


def test_fit_scalers_and_transform_updates_columns(sample_train_data):
    scalers = fit_scalers(sample_train_data)
    transformed = transform_with_scalers(sample_train_data, scalers)
    assert set(scalers.keys()) == {"Amount", "Time"} and not transformed.equals(sample_train_data)


def test_save_outputs_writes_metadata_and_registers_assets(
    tmp_path, sample_train_data, fake_ml_client, monkeypatch
):
    settings = SimpleNamespace(
        serving_scalers_name="fraud-scalers",
        registered_train="fraud-train",
        registered_test="fraud-test",
    )
    monkeypatch.setattr(
        "fraud_detection.data.data_for_train.data_for_train.get_settings",
        lambda: settings,
    )

    fake_ml_client.data.list.return_value = []
    fake_ml_client.data.create_or_update.side_effect = lambda asset: asset

    train_df, test_df = split_data(sample_train_data, test=0.2, stratify=False)
    scalers = fit_scalers(train_df)

    scalers_output = tmp_path / "scalers"
    metadata_output = tmp_path / "metadata.json"

    metadata = save_outputs(
        fake_ml_client,
        train_df=train_df,
        test_df=test_df,
        scalers=scalers,
        scalers_output=scalers_output,
        metadata_output=metadata_output,
        label_col="Class",
        seed=123,
        split_ratio=0.2,
    )

    saved = json.loads(metadata_output.read_text(encoding="utf-8"))
    assert metadata["split_sizes"]["train"] == saved["split_sizes"]["train"]
