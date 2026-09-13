from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from cell_msca.baselines import (
    BaselineArraySplit,
    BaselinePredictions,
    BaselineProvenance,
    ModelContract,
    TestEvaluationBlockedError,
    _original_prediction_pair,
)
from cell_msca.phase6 import (
    PHASE6_ARTIFACT_KEYS,
    FinalTestGate,
    FrozenArtifactSpec,
    LoadedFrozenModel,
    _PHASE6_GATE_AUTHORITY,
    _aggregate_seed_metrics,
    _extract_verified_member,
    execute_locked_final_test,
    load_phase6_protocol,
    validate_test_structure,
    verify_final_test_gate,
)
from cell_msca.target import target_transform


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase6_locked_test_protocol.json"


class _FrozenPredictor:
    def __init__(self, contract: ModelContract, offset: float) -> None:
        self.contract = contract
        self.offset = offset

    def predict(self, features: np.ndarray) -> BaselinePredictions:
        prediction = np.full(features.shape[0], self.offset, dtype=np.float64)
        prediction[0] = -1.0
        return _original_prediction_pair(
            prediction,
            target_scale=self.contract.target_scale,
            model_name=self.contract.model_name,
        )


class _SyntheticProtocol:
    def __init__(self, split: BaselineArraySplit) -> None:
        digest = "a" * 64
        self.provenance = BaselineProvenance(
            data_version="synthetic",
            data_sha256=digest,
            split_sha256="b" * 64,
            split_config_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            split_seed=42,
            train_seed=42,
            target_scale=1.0,
        )
        self.split = split
        self.test_data_calls = 0

    def test_data(self, authorization: object) -> BaselineArraySplit:
        if authorization is None:
            raise AssertionError("authorization is required")
        self.test_data_calls += 1
        if self.test_data_calls > 1:
            raise AssertionError("test materialized more than once")
        return self.split


def _synthetic_test_split() -> BaselineArraySplit:
    rows_per_cell = 36
    cell_ids = np.repeat(["r0_c0", "r0_c1"], rows_per_cell)
    month_ids = np.tile(
        [f"2022-{month:02d}" for month in range(1, 13)]
        + [f"2023-{month:02d}" for month in range(1, 13)]
        + [f"2024-{month:02d}" for month in range(1, 13)],
        2,
    )
    target = np.linspace(0.0, 20.0, cell_ids.size)
    return BaselineArraySplit(
        name="test",
        features=np.arange(cell_ids.size * 7, dtype=np.float64).reshape(-1, 7),
        target_original=target,
        target_log=target_transform(target),
        cell_ids=cell_ids,
        month_ids=month_ids,
    )


def _synthetic_gate(protocol: _SyntheticProtocol) -> FinalTestGate:
    models: dict[tuple[str, int], LoadedFrozenModel] = {}
    for model_name, seed in sorted(PHASE6_ARTIFACT_KEYS):
        configuration_sha256 = f"{seed:02x}" * 32
        spec = FrozenArtifactSpec(
            model_name=model_name,
            train_seed=seed,
            source_package=(
                "seed42" if model_name.startswith("cell_msca_") and seed == 42
                else "phase5a"
            ),
            artifact_directory=f"runs/{model_name}_{seed}",
            manifest_path=f"runs/{model_name}_{seed}/run_manifest.json",
            manifest_sha256="1" * 64,
            validation_prediction_path=(
                f"runs/{model_name}_{seed}/validation_predictions.csv"
            ),
            validation_prediction_sha256="2" * 64,
            model_artifact_path=f"runs/{model_name}_{seed}/model.bin",
            model_artifact_sha256="3" * 64,
            model_artifact_type=(
                "lightgbm_text" if model_name == "lightgbm_raw" else "torch_weights"
            ),
            configuration_sha256=configuration_sha256,
            git_commit_sha="4" * 40,
        )
        contract = ModelContract(
            model_name=model_name,
            config_sha256=configuration_sha256,
            target_transform="identity",
            target_scale=1.0,
            loss_objective="synthetic_frozen_prediction",
            inverse_mode="none",
            smearing_factor=None,
            best_iteration=1 if model_name == "lightgbm_raw" else None,
            best_epoch=None if model_name == "lightgbm_raw" else 1,
        )
        predictor = _FrozenPredictor(contract, offset=2.0 + seed / 100.0)
        models[(model_name, seed)] = LoadedFrozenModel(spec, predictor, {})
    config = {
        "schema_version": "cell_msca.phase6_locked_test_protocol.v1",
        "validation_code_git_sha": "5" * 40,
        "test_contract": {
            "expected_unique_cells": 2,
            "expected_rows_per_cell": 36,
            "expected_samples": 72,
        },
        "metrics": {"seed_summary_standard_deviation_ddof": 1},
        "paired_cell_cluster_bootstrap": {
            "n_boot": 50,
            "seed": 3407,
            "alpha": 0.05,
            "rows_per_cell": 36,
            "prediction_basis": "rowwise_mean_of_three_frozen_seed_predictions",
        },
    }
    return FinalTestGate(
        config=config,
        config_sha256="6" * 64,
        loaded_models=models,
        audit={
            "status": "passed",
            "test_subset_materialized": False,
            "test_evaluation_performed": False,
        },
        _authority=_PHASE6_GATE_AUTHORITY,
    )


class Phase6ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_contains_exact_nine_artifacts(self) -> None:
        values = load_phase6_protocol(CONFIG_PATH)
        keys = {
            (str(item["model_name"]), int(item["train_seed"]))
            for item in values["artifacts"]
        }
        self.assertEqual(keys, PHASE6_ARTIFACT_KEYS)
        self.assertEqual(values["prediction_support_policy"], "nonnegative_max_zero_v1")

    def test_missing_and_duplicate_artifact_keys_are_rejected(self) -> None:
        values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            missing = copy.deepcopy(values)
            missing["artifacts"].pop()
            path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly nine"):
                load_phase6_protocol(path)

            duplicate = copy.deepcopy(values)
            duplicate["artifacts"][-1] = copy.deepcopy(duplicate["artifacts"][0])
            path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or duplicated"):
                load_phase6_protocol(path)

    def test_explicit_allow_is_checked_before_any_materialization_or_file_access(self) -> None:
        with self.assertRaises(TestEvaluationBlockedError):
            verify_final_test_gate(
                config_path="does-not-exist.json",
                phase5a_zip="does-not-exist.zip",
                seed42_zip="does-not-exist.zip",
                output_dir="does-not-exist-output",
                extraction_root="does-not-exist-models",
                allow_final_test=False,
            )

    def test_existing_output_is_rejected_before_package_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                verify_final_test_gate(
                    config_path=CONFIG_PATH,
                    phase5a_zip=root / "missing.zip",
                    seed42_zip=root / "missing2.zip",
                    output_dir=output,
                    extraction_root=root / "models",
                    allow_final_test=True,
                )

    def test_wrong_package_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase5a = root / "cell-msca-phase5a-full-validation-35696c7e8c25.zip"
            seed42 = root / "cell-msca-v1-legacy-validation-034712a.zip"
            phase5a.write_bytes(b"wrong")
            seed42.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "package SHA-256 mismatch"):
                verify_final_test_gate(
                    config_path=CONFIG_PATH,
                    phase5a_zip=phase5a,
                    seed42_zip=seed42,
                    output_dir=root / "output",
                    extraction_root=root / "models",
                    allow_final_test=True,
                )

    def test_wrong_checkpoint_member_hash_is_rejected(self) -> None:
        spec = FrozenArtifactSpec(
            model_name="lightgbm_raw",
            train_seed=42,
            source_package="phase5a",
            artifact_directory="run",
            manifest_path="run/run_manifest.json",
            manifest_sha256="1" * 64,
            validation_prediction_path="run/validation_predictions.csv",
            validation_prediction_sha256="2" * 64,
            model_artifact_path="run/fitted_model.txt",
            model_artifact_sha256="0" * 64,
            model_artifact_type="lightgbm_text",
            configuration_sha256="3" * 64,
            git_commit_sha="4" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "models.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(spec.model_artifact_path, b"tampered")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    _extract_verified_member(
                        archive,
                        {spec.model_artifact_path},
                        spec,
                        root,
                    )

    def test_test_structure_rejects_incomplete_cells(self) -> None:
        split = _synthetic_test_split()
        summary = validate_test_structure(
            split,
            expected_cells=2,
            rows_per_cell=36,
            expected_samples=72,
        )
        self.assertTrue(summary["complete_cell_month_structure_verified"])
        broken = BaselineArraySplit(
            name="test",
            features=split.features[:-1],
            target_original=split.target_original[:-1],
            target_log=split.target_log[:-1],
            cell_ids=split.cell_ids[:-1],
            month_ids=split.month_ids[:-1],
        )
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            validate_test_structure(
                broken,
                expected_cells=2,
                rows_per_cell=36,
                expected_samples=72,
            )

    def test_seed_metric_summary_uses_sample_standard_deviation(self) -> None:
        rows = {}
        for model_name in (
            "lightgbm_raw",
            "cell_msca_token_no_attention",
            "cell_msca_bidirectional",
        ):
            for seed, value in zip((42, 43, 44), (1.0, 2.0, 3.0), strict=True):
                rows[(model_name, seed)] = {
                    "primary_original_unit": {"mae": value, "rmse": value, "r2": value},
                    "secondary": {
                        "bias": value,
                        "spearman": value,
                        "log_unit_mae": value,
                        "log_unit_rmse": value,
                        "log_unit_r2": value,
                    },
                }
        summary = _aggregate_seed_metrics(rows, ddof=1)
        self.assertEqual(summary["lightgbm_raw"]["aggregate"]["mae"]["std"], 1.0)

    def test_synthetic_end_to_end_materializes_test_once_without_training(self) -> None:
        split = _synthetic_test_split()
        protocol = _SyntheticProtocol(split)
        gate = _synthetic_gate(protocol)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "cell_msca.baselines.fit_lightgbm_baseline",
            side_effect=AssertionError("training must not be called"),
        ), mock.patch(
            "cell_msca.baselines.tune_on_validation",
            side_effect=AssertionError("selection must not be called"),
        ):
            output = Path(directory) / "phase6"
            result = execute_locked_final_test(
                protocol=protocol,
                gate=gate,
                output_dir=output,
            )
            self.assertEqual(protocol.test_data_calls, 1)
            self.assertTrue(Path(result["final_zip"]).is_file())
            manifest = json.loads(
                (output / "phase6_test_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["test_subset_materialized"])
            self.assertTrue(manifest["test_evaluation_performed"])
            self.assertFalse(manifest["training_performed"])
            self.assertFalse(manifest["model_selection_performed"])
            self.assertFalse(manifest["hyperparameter_change_performed"])
            self.assertEqual(manifest["test_materialization_count"], 1)
            self.assertEqual(len(manifest["run_artifacts"]), 9)
            metrics_path = (
                output / "runs" / "lightgbm_raw_seed42" / "test_metrics.json"
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["pre_projection_negative_count"], 1)
            self.assertEqual(metrics["projection_applied_count"], 1)
            self.assertEqual(metrics["train_seed"], 42)
            seed44_metrics = json.loads(
                (
                    output
                    / "runs"
                    / "cell_msca_bidirectional_seed44"
                    / "test_metrics.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(seed44_metrics["train_seed"], 44)
            predictions = np.asarray(
                [
                    float(row["pred_original"])
                    for row in csv.DictReader(
                        (
                            output
                            / "runs"
                            / "lightgbm_raw_seed42"
                            / "test_predictions.csv"
                        ).read_text(encoding="utf-8").splitlines()
                    )
                ]
            )
            self.assertEqual(predictions[0], 0.0)
            self.assertTrue(np.all(predictions >= 0.0))
            bootstrap = json.loads(
                (output / "paired_test_cell_cluster_bootstrap.json").read_text(
                    encoding="utf-8"
                )
            )
            for comparison in bootstrap["comparisons"].values():
                self.assertTrue(comparison["complete_cell_resampling"])
                self.assertEqual(comparison["rows_per_cell"], 36)

    def test_invalid_gate_never_materializes_test(self) -> None:
        protocol = _SyntheticProtocol(_synthetic_test_split())
        gate = _synthetic_gate(protocol)
        invalid = FinalTestGate(
            config=gate.config,
            config_sha256=gate.config_sha256,
            loaded_models=gate.loaded_models,
            audit=gate.audit,
            _authority=object(),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TestEvaluationBlockedError):
                execute_locked_final_test(
                    protocol=protocol,
                    gate=invalid,
                    output_dir=Path(directory) / "output",
                )
        self.assertEqual(protocol.test_data_calls, 0)


if __name__ == "__main__":
    unittest.main()
