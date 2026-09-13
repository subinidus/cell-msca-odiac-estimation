from __future__ import annotations

import copy
import contextlib
import csv
import hashlib
import json
import sys
import tempfile
import types
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np

from cell_msca.baselines import (
    BaselineArraySplit,
    BaselinePredictions,
    BaselineProvenance,
    ModelContract,
    TestEvaluationBlockedError,
    _original_prediction_pair,
    evaluate_fitted_baseline,
)
from cell_msca.phase6 import (
    PHASE6_ARTIFACT_KEYS,
    FinalTestGate,
    FrozenArtifactSpec,
    LoadedFrozenModel,
    _PHASE6_GATE_AUTHORITY,
    _aggregate_seed_metrics,
    _claim_execution,
    _extract_verified_member,
    _gate_identity,
    _phase6_metrics_from_csv,
    _read_registry_claim,
    deterministic_execution_id,
    execute_locked_final_test,
    load_phase6_protocol,
    protocol_fingerprint,
    validate_test_structure,
    verify_final_test_gate,
)
from cell_msca.evaluate import MetricMismatchError, write_prediction_csv
from cell_msca.data import canonical_sha256
from cell_msca.target import target_transform


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase6_locked_test_protocol.json"


class _FrozenPredictor:
    def __init__(
        self,
        contract: ModelContract,
        offset: float,
        *,
        fail: bool = False,
        mutate: bool = False,
    ) -> None:
        self.contract = contract
        self.offset = offset
        self.fail = fail
        self.mutate = mutate

    def phase6_state_fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.contract.model_name}|{self.offset}".encode("utf-8")
        ).hexdigest()

    def predict(self, features: np.ndarray) -> BaselinePredictions:
        if self.fail:
            raise RuntimeError("injected frozen prediction failure")
        prediction = np.full(features.shape[0], self.offset, dtype=np.float64)
        prediction[0] = -1.0
        if self.mutate:
            self.offset += 1.0
        return _original_prediction_pair(
            prediction,
            target_scale=self.contract.target_scale,
            model_name=self.contract.model_name,
        )


class _SyntheticProtocol:
    def __init__(
        self,
        split: BaselineArraySplit,
        *,
        fail_materialization: bool = False,
        data_sha256: str = "a" * 64,
    ) -> None:
        self.provenance = BaselineProvenance(
            data_version="synthetic",
            data_sha256=data_sha256,
            split_sha256="b" * 64,
            split_config_sha256="c" * 64,
            preprocessing_sha256="d" * 64,
            split_seed=42,
            train_seed=42,
            target_scale=1.0,
        )
        self.split = split
        self.test_data_calls = 0
        self.fail_materialization = fail_materialization

    def test_data(self, authorization: object) -> BaselineArraySplit:
        if authorization is None:
            raise AssertionError("authorization is required")
        self.test_data_calls += 1
        if self.fail_materialization:
            raise RuntimeError("injected materialization failure")
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


def _protocol_for_frozen_config() -> _SyntheticProtocol:
    values = load_phase6_protocol(CONFIG_PATH)
    protocol = _SyntheticProtocol(_synthetic_test_split())
    required = values["required_hashes"]
    protocol.provenance = BaselineProvenance(
        data_version="v1_legacy",
        data_sha256=required["data_sha256"],
        split_sha256=required["split_sha256"],
        split_config_sha256=required["split_config_sha256"],
        preprocessing_sha256=required["preprocessing_sha256"],
        split_seed=42,
        train_seed=42,
        target_scale=1.0,
    )
    return protocol


def _synthetic_gate(
    protocol: _SyntheticProtocol,
    *,
    registry_root: Path,
    output_dir: Path,
    failing_key: tuple[str, int] | None = None,
    mutating_key: tuple[str, int] | None = None,
) -> FinalTestGate:
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
        key = (model_name, seed)
        predictor = _FrozenPredictor(
            contract,
            offset=2.0 + seed / 100.0,
            fail=key == failing_key,
            mutate=key == mutating_key,
        )
        models[key] = LoadedFrozenModel(
            spec,
            predictor,
            MappingProxyType({}),
            predictor.phase6_state_fingerprint(),
        )
    package_hash = "7" * 64
    seed42_hash = "8" * 64
    config = {
        "schema_version": "cell_msca.phase6_locked_test_protocol.v1",
        "validation_code_git_sha": "5" * 40,
        "data": {"data_version": "synthetic", "target_scale": 1.0},
        "required_hashes": {
            "data_sha256": protocol.provenance.data_sha256,
            "split_sha256": protocol.provenance.split_sha256,
            "split_config_sha256": protocol.provenance.split_config_sha256,
            "preprocessing_sha256": protocol.provenance.preprocessing_sha256,
        },
        "validation_packages": {
            "phase5a": {"file_sha256": package_hash},
            "seed42": {"file_sha256": seed42_hash},
        },
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
    config_sha256 = "6" * 64
    source_git_sha = "4" * 40
    execution_id = deterministic_execution_id(
        config_sha256=config_sha256,
        phase5a_package_sha256=package_hash,
        seed42_package_sha256=seed42_hash,
        data_sha256=protocol.provenance.data_sha256,
        split_sha256=protocol.provenance.split_sha256,
        split_config_sha256=protocol.provenance.split_config_sha256,
        preprocessing_sha256=protocol.provenance.preprocessing_sha256,
        source_git_sha=source_git_sha,
    )
    gate = FinalTestGate(
        config=MappingProxyType(config),
        config_sha256=config_sha256,
        config_canonical_sha256=canonical_sha256(config),
        phase5a_package_sha256=package_hash,
        seed42_package_sha256=seed42_hash,
        data_sha256=protocol.provenance.data_sha256,
        split_sha256=protocol.provenance.split_sha256,
        split_config_sha256=protocol.provenance.split_config_sha256,
        preprocessing_sha256=protocol.provenance.preprocessing_sha256,
        source_git_sha=source_git_sha,
        evaluation_keys=tuple(sorted(models)),
        protocol_fingerprint=protocol_fingerprint(protocol),
        execution_id=execution_id,
        registry_claim_path=registry_root / "pending.json",
        loaded_models=MappingProxyType(models),
        audit=MappingProxyType({
            "status": "passed",
            "test_subset_materialized": False,
            "test_evaluation_performed": False,
        }),
        _authority=_PHASE6_GATE_AUTHORITY,
    )
    claim_path, _ = _claim_execution(
        registry_root=registry_root,
        execution_id=execution_id,
        output_dir=output_dir,
        gate_identity=_gate_identity(gate),
    )
    return replace(gate, registry_claim_path=claim_path)


class Phase6ProtocolTests(unittest.TestCase):
    def test_frozen_protocol_contains_exact_nine_artifacts(self) -> None:
        values = load_phase6_protocol(CONFIG_PATH)
        keys = {
            (str(item["model_name"]), int(item["train_seed"]))
            for item in values["artifacts"]
        }
        self.assertEqual(keys, PHASE6_ARTIFACT_KEYS)
        self.assertEqual(values["prediction_support_policy"], "nonnegative_max_zero_v1")
        self.assertFalse(values["run_registry"]["recovery_allowed"])
        self.assertEqual(
            values["run_registry"]["states"],
            [
                "claimed",
                "materialization_started",
                "materialized",
                "evaluating",
                "failed",
                "completed",
            ],
        )

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

    def test_protocol_is_mandatory_for_executable_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                TestEvaluationBlockedError, "exact verified frozen protocol"
            ):
                verify_final_test_gate(
                    config_path=root / "missing.json",
                    phase5a_zip=root / "missing-phase5a.zip",
                    seed42_zip=root / "missing-seed42.zip",
                    output_dir=root / "output",
                    extraction_root=root / "models",
                    allow_final_test=True,
                    protocol=None,
                    registry_root=root / "registry",
                    repository_root=root,
                )

    def test_failed_run_resume_request_is_always_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(TestEvaluationBlockedError, "recovery is disabled"):
                verify_final_test_gate(
                    config_path=root / "missing.json",
                    phase5a_zip=root / "missing-phase5a.zip",
                    seed42_zip=root / "missing-seed42.zip",
                    output_dir=root / "output",
                    extraction_root=root / "models",
                    allow_final_test=True,
                    protocol=_SyntheticProtocol(_synthetic_test_split()),
                    registry_root=root / "registry",
                    repository_root=root,
                    allow_resume_failed_run=True,
                    resume_execution_id="wrong-execution-id",
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
                    protocol=_SyntheticProtocol(_synthetic_test_split()),
                    registry_root=root / "registry",
                    repository_root=root,
                )

    def test_wrong_package_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase5a = root / "cell-msca-phase5a-full-validation-35696c7e8c25.zip"
            seed42 = root / "cell-msca-v1-legacy-validation-034712a.zip"
            phase5a.write_bytes(b"wrong")
            seed42.write_bytes(b"wrong")
            def fake_git(command: list[str], **_: object) -> mock.Mock:
                if command[1:3] == ["rev-parse", "HEAD"]:
                    return mock.Mock(returncode=0, stdout="9" * 40 + "\n")
                if command[1] == "merge-base":
                    return mock.Mock(returncode=0, stdout="")
                if command[1] == "status":
                    return mock.Mock(returncode=0, stdout="")
                raise AssertionError(command)

            with self.assertRaisesRegex(ValueError, "package SHA-256 mismatch"), mock.patch(
                "cell_msca.phase6.subprocess.run", side_effect=fake_git
            ):
                verify_final_test_gate(
                    config_path=CONFIG_PATH,
                    phase5a_zip=phase5a,
                    seed42_zip=seed42,
                    output_dir=root / "output",
                    extraction_root=root / "models",
                    allow_final_test=True,
                    protocol=_protocol_for_frozen_config(),
                    registry_root=root / "registry",
                    repository_root=ROOT,
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

    def test_gate_protocol_mismatch_stops_before_output_and_materialization(self) -> None:
        protocol = _SyntheticProtocol(_synthetic_test_split())
        mismatched = _SyntheticProtocol(
            _synthetic_test_split(), data_sha256="e" * 64
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            gate = _synthetic_gate(
                protocol,
                registry_root=root / "registry",
                output_dir=output,
            )
            with self.assertRaisesRegex(ValueError, "data_sha256 mismatch"):
                execute_locked_final_test(
                    protocol=mismatched,
                    gate=gate,
                    output_dir=output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(mismatched.test_data_calls, 0)
            self.assertEqual(_read_registry_claim(gate.registry_claim_path)["state"], "claimed")

    def test_metric_recalculation_allows_roundoff_but_rejects_material_delta(self) -> None:
        split = _synthetic_test_split()
        protocol = _SyntheticProtocol(split)
        contract = ModelContract(
            model_name="lightgbm_raw",
            config_sha256="f" * 64,
            target_transform="identity",
            target_scale=1.0,
            loss_objective="synthetic",
            inverse_mode="none",
            smearing_factor=None,
            best_iteration=1,
        )
        predictor = _FrozenPredictor(contract, 2.5)
        evaluation = evaluate_fitted_baseline(predictor, split, protocol.provenance)
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = write_prediction_csv(
                Path(directory) / "predictions.csv",
                y_true_original=split.target_original,
                y_pred_original=evaluation.predictions.pred_original,
                y_true_log=split.target_log,
                y_pred_log=evaluation.predictions.pred_log,
                cell_ids=split.cell_ids,
            )
            roundoff = copy.deepcopy(evaluation.result)
            roundoff["headline_metrics"]["original_unit"]["mae"] += 1e-14
            audit = _phase6_metrics_from_csv(prediction_path, roundoff)
            self.assertLessEqual(
                audit["metric_recalculation_maximum_absolute_delta"], 1.1e-14
            )
            material = copy.deepcopy(evaluation.result)
            material["headline_metrics"]["original_unit"]["mae"] += 1e-8
            with self.assertRaises(MetricMismatchError):
                _phase6_metrics_from_csv(prediction_path, material)

            nonfinite = copy.deepcopy(evaluation.result)
            nonfinite["headline_metrics"]["original_unit"]["mae"] = float("nan")
            with self.assertRaises(ValueError):
                _phase6_metrics_from_csv(prediction_path, nonfinite)

    def test_materialization_failure_is_recorded_and_cannot_restart(self) -> None:
        protocol = _SyntheticProtocol(
            _synthetic_test_split(), fail_materialization=True
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run-a"
            gate = _synthetic_gate(
                protocol,
                registry_root=root / "registry",
                output_dir=output,
            )
            with self.assertRaisesRegex(RuntimeError, "materialization failure"):
                execute_locked_final_test(
                    protocol=protocol,
                    gate=gate,
                    output_dir=output,
                )
            registry = _read_registry_claim(gate.registry_claim_path)
            self.assertEqual(registry["state"], "failed")
            self.assertFalse(registry["test_subset_materialized"])
            failure = json.loads(
                (output / "phase6_failure_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(failure["test_subset_materialized"])
            self.assertFalse(failure["recovery_allowed"])
            with self.assertRaises(TestEvaluationBlockedError):
                _claim_execution(
                    registry_root=root / "registry",
                    execution_id=gate.execution_id,
                    output_dir=root / "run-b",
                    gate_identity=_gate_identity(gate),
                )

    def test_failure_immediately_after_materialization_records_true_without_evaluation(
        self,
    ) -> None:
        complete = _synthetic_test_split()
        incomplete = BaselineArraySplit(
            name="test",
            features=complete.features[:-1],
            target_original=complete.target_original[:-1],
            target_log=complete.target_log[:-1],
            cell_ids=complete.cell_ids[:-1],
            month_ids=complete.month_ids[:-1],
        )
        protocol = _SyntheticProtocol(incomplete)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            gate = _synthetic_gate(
                protocol,
                registry_root=root / "registry",
                output_dir=output,
            )
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                execute_locked_final_test(
                    protocol=protocol,
                    gate=gate,
                    output_dir=output,
                )
            registry = _read_registry_claim(gate.registry_claim_path)
            self.assertEqual(registry["state"], "failed")
            self.assertTrue(registry["test_subset_materialized"])
            self.assertFalse(registry["test_evaluation_performed"])
            self.assertEqual(registry["materialized_row_count"], 71)
            failure = json.loads(
                (output / "phase6_failure_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(failure["test_subset_materialized"])
            self.assertFalse(failure["test_evaluation_performed"])

    def test_nth_prediction_failure_records_materialization_and_completed_artifacts(self) -> None:
        protocol = _SyntheticProtocol(_synthetic_test_split())
        failing_key = tuple(sorted(PHASE6_ARTIFACT_KEYS))[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run-a"
            gate = _synthetic_gate(
                protocol,
                registry_root=root / "registry",
                output_dir=output,
                failing_key=failing_key,
            )
            with self.assertRaisesRegex(RuntimeError, "prediction failure"):
                execute_locked_final_test(
                    protocol=protocol,
                    gate=gate,
                    output_dir=output,
                )
            registry = _read_registry_claim(gate.registry_claim_path)
            self.assertEqual(registry["state"], "failed")
            self.assertTrue(registry["test_subset_materialized"])
            self.assertTrue(registry["test_evaluation_performed"])
            self.assertEqual(len(registry["completed_model_seeds"]), 1)
            self.assertEqual(len(registry["completed_artifacts"]), 1)
            self.assertEqual(
                registry["failed_model_seed"],
                {"model_name": failing_key[0], "train_seed": failing_key[1]},
            )
            for artifact in registry["completed_artifacts"]:
                for details in artifact["files"].values():
                    self.assertEqual(
                        hashlib.sha256(Path(details["path"]).read_bytes()).hexdigest(),
                        details["sha256"],
                    )

    def test_parameter_mutation_during_inference_is_rejected(self) -> None:
        protocol = _SyntheticProtocol(_synthetic_test_split())
        mutating_key = tuple(sorted(PHASE6_ARTIFACT_KEYS))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            gate = _synthetic_gate(
                protocol,
                registry_root=root / "registry",
                output_dir=output,
                mutating_key=mutating_key,
            )
            with self.assertRaisesRegex(
                TestEvaluationBlockedError, "parameters changed during inference"
            ):
                execute_locked_final_test(
                    protocol=protocol,
                    gate=gate,
                    output_dir=output,
                )

    def test_inference_path_never_calls_training_or_selection_apis(self) -> None:
        protocol = _SyntheticProtocol(_synthetic_test_split())
        forbidden = (
            "cell_msca.baselines.fit_lightgbm_baseline",
            "cell_msca.baselines.tune_on_validation",
            "cell_msca.baselines.select_log_inverse_on_validation",
            "cell_msca.phase5a.run_phase5a_lightgbm",
            "cell_msca.phase5a.freeze_lightgbm_convergence_selection",
            "cell_msca.phase5a.load_lightgbm_selection",
            "cell_msca.phase5a.run_phase5a_cell_repeat",
            "cell_msca.phase5a.aggregate_phase5a_results",
            "cell_msca.experiment.run_validation_tuning_from_config",
            "cell_msca.kaggle_runner.run_kaggle_validation",
            "cell_msca.neural_baselines.fit_concat_mlp",
            "cell_msca.neural_baselines.fit_log_neural_model",
            "cell_msca.train.fit_cell_msca",
            "cell_msca.train.save_selected_checkpoint",
        )
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            output = root / "output"
            gate = _synthetic_gate(
                protocol,
                registry_root=root / "registry",
                output_dir=output,
            )
            for target in forbidden:
                stack.enter_context(
                    mock.patch(target, side_effect=AssertionError(f"forbidden API: {target}"))
                )
            fake_lightgbm = types.SimpleNamespace(
                early_stopping=mock.Mock(
                    side_effect=AssertionError("LightGBM early stopping is forbidden")
                ),
                LGBMRegressor=mock.Mock(
                    side_effect=AssertionError("LightGBM fit is forbidden")
                ),
            )
            stack.enter_context(mock.patch.dict(sys.modules, {"lightgbm": fake_lightgbm}))
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                stack.enter_context(
                    mock.patch.object(
                        torch.optim.Optimizer,
                        "step",
                        side_effect=AssertionError("optimizer.step is forbidden"),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        torch.Tensor,
                        "backward",
                        side_effect=AssertionError("backward is forbidden"),
                    )
                )
            result = execute_locked_final_test(
                protocol=protocol,
                gate=gate,
                output_dir=output,
            )
            self.assertEqual(result["manifest"]["training_performed"], False)
            fake_lightgbm.early_stopping.assert_not_called()
            fake_lightgbm.LGBMRegressor.assert_not_called()

    def test_synthetic_end_to_end_materializes_test_once_without_training(self) -> None:
        split = _synthetic_test_split()
        protocol = _SyntheticProtocol(split)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "cell_msca.baselines.fit_lightgbm_baseline",
            side_effect=AssertionError("training must not be called"),
        ), mock.patch(
            "cell_msca.baselines.tune_on_validation",
            side_effect=AssertionError("selection must not be called"),
        ):
            output = Path(directory) / "phase6"
            gate = _synthetic_gate(
                protocol,
                registry_root=Path(directory) / "registry",
                output_dir=output,
            )
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
            registry = _read_registry_claim(gate.registry_claim_path)
            self.assertEqual(registry["state"], "completed")
            self.assertEqual(len(registry["completed_artifacts"]), 9)
            with self.assertRaises(TestEvaluationBlockedError):
                _claim_execution(
                    registry_root=Path(directory) / "registry",
                    execution_id=gate.execution_id,
                    output_dir=Path(directory) / "run-b",
                    gate_identity=_gate_identity(gate),
                )

    def test_invalid_gate_never_materializes_test(self) -> None:
        protocol = _SyntheticProtocol(_synthetic_test_split())
        with tempfile.TemporaryDirectory() as directory:
            gate = _synthetic_gate(
                protocol,
                registry_root=Path(directory) / "registry",
                output_dir=Path(directory) / "output",
            )
            invalid = replace(gate, _authority=object())
            with self.assertRaises(TestEvaluationBlockedError):
                execute_locked_final_test(
                    protocol=protocol,
                    gate=invalid,
                    output_dir=Path(directory) / "output",
                )
        self.assertEqual(protocol.test_data_calls, 0)


if __name__ == "__main__":
    unittest.main()
