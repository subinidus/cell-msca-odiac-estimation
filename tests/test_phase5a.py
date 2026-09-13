"""Regression tests for the frozen Phase 5A validation-robustness contract."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import numpy as np

from cell_msca.baselines import (
    FittedLightGBM,
    LightGBMConfig,
    ModelContract,
    _original_prediction_pair,
)
from cell_msca.data import canonical_npz_content_sha256, canonical_sha256, file_sha256
from cell_msca.evaluate import (
    LEGACY_BASELINE_RESULT_FIELDS,
    LEGACY_BASELINE_RESULT_SCHEMA_VERSION,
    BaselineResultContext,
    evaluate_baseline_predictions,
    write_metrics_json,
    write_prediction_csv,
)
from cell_msca.kaggle_runner import GitIdentity
from cell_msca.phase5a import (
    _aligned_prediction_arrays,
    _build_parser,
    _lightgbm_actual_iterations,
    _lightgbm_experiment_id,
    _lightgbm_config,
    _lightgbm_iteration_summary,
    _seed_metric_summary,
    _save_and_verify_lightgbm_model,
    _validate_phase5a_output_path,
    SEED42_REQUIRED_ARTIFACT_KEYS,
    aggregate_phase5a_results,
    discover_validation_artifacts,
    freeze_lightgbm_convergence_selection,
    load_lightgbm_selection,
    load_phase5a_assignments,
    load_phase5a_config,
    load_stored_validation_artifact,
    paired_cell_cluster_mae_difference,
    run_phase5a_lightgbm,
)
from cell_msca.v1_validation import (
    V1_ARCHIVE_NAME,
    _verify_v1_npz_files,
    verified_v1_input,
)


def _write_validation_artifact(
    directory: Path,
    *,
    model_name: str,
    train_seed: int,
    config_sha256: str,
    git_sha: str,
    hashes: dict[str, str],
    prediction_offset: float,
    run_type: str = "repeat",
    rows_per_cell: int = 1,
    legacy_schema: bool = False,
) -> Path:
    directory.mkdir(parents=True)
    true_original = np.repeat(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        rows_per_cell,
    )
    pred_original = true_original + prediction_offset
    true_log = np.log1p(true_original)
    pred_log = np.log1p(pred_original)
    context = BaselineResultContext(
        model_name=model_name,
        data_version="v1_legacy",
        data_sha256=hashes["data_sha256"],
        split_sha256=hashes["split_sha256"],
        split_config_sha256=hashes["split_config_sha256"],
        config_sha256=config_sha256,
        preprocessing_sha256=hashes["preprocessing_sha256"],
        split_seed=42,
        train_seed=train_seed,
        evaluation_split="validation",
        target_transform="log1p" if model_name.endswith("log1p") else "identity",
        target_scale=1.0,
        loss_objective="synthetic_test",
        inverse_mode="median" if model_name.endswith("log1p") else "none",
        best_iteration=25 if model_name.startswith("lightgbm") else None,
        best_epoch=3 if model_name.startswith("cell_msca") else None,
    )
    result = evaluate_baseline_predictions(
        context,
        y_true_original=true_original,
        y_pred_original=pred_original,
        y_true_log=true_log,
        y_pred_log=pred_log,
    )
    stored_result = dict(result)
    if legacy_schema:
        stored_result["schema_version"] = LEGACY_BASELINE_RESULT_SCHEMA_VERSION
        for field in set(stored_result) - LEGACY_BASELINE_RESULT_FIELDS:
            del stored_result[field]
    write_prediction_csv(
        directory / "validation_predictions.csv",
        y_true_original=true_original,
        y_pred_original=pred_original,
        y_true_log=true_log,
        y_pred_log=pred_log,
        cell_ids=np.repeat(
            np.asarray(["cell-a", "cell-b", "cell-c"]),
            rows_per_cell,
        ),
    )
    write_metrics_json(
        directory / "validation_metrics.json",
        {
            "artifact_classification": "validation-only",
            **(
                {"actual_iterations": 25}
                if model_name.startswith("lightgbm")
                else {}
            ),
            "result": stored_result,
        },
    )
    write_metrics_json(
        directory / "run_manifest.json",
        {
            "status": "completed",
            "run_type": run_type,
            "model_name": model_name,
            "train_seed": train_seed,
            "split_seed": 42,
            **hashes,
            "configuration_sha256": config_sha256,
            **(
                {}
                if legacy_schema
                else {
                    "prediction_support_policy": result[
                        "prediction_support_policy"
                    ],
                    "pre_projection_negative_count": result[
                        "pre_projection_negative_count"
                    ],
                    "pre_projection_negative_fraction": result[
                        "pre_projection_negative_fraction"
                    ],
                    "pre_projection_minimum": result[
                        "pre_projection_minimum"
                    ],
                    "projection_applied_count": result[
                        "projection_applied_count"
                    ],
                }
            ),
            "git_commit_sha": git_sha,
            "allowed_materialized_splits": ["train", "validation"],
            "test_subset_materialized": False,
            "test_evaluation_performed": False,
            **(
                {"actual_iterations": 25}
                if model_name.startswith("lightgbm")
                else {}
            ),
            "maximum_iteration_reached": False,
        },
    )
    return directory


class Phase5AContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config_path = cls.root / "configs" / "phase5a_validation_robustness.json"
        cls.assignments_path = cls.root / "configs" / "phase5a_experiment_assignments.json"
        cls.values = load_phase5a_config(cls.config_path)

    def _write_required_discovery_fixture(self, root: Path) -> None:
        hashes = dict(self.values["required_hashes"])
        for index, (model_name, train_seed) in enumerate(
            sorted(SEED42_REQUIRED_ARTIFACT_KEYS),
            start=1,
        ):
            _write_validation_artifact(
                root / f"required-{index}",
                model_name=model_name,
                train_seed=train_seed,
                config_sha256=str(index) * 64,
                git_sha="a" * 40,
                hashes=hashes,
                prediction_offset=index / 100.0,
                legacy_schema=True,
            )

    def test_frozen_config_and_assignments_reject_scope_expansion(self) -> None:
        values = self.values
        self.assertEqual(values["seeds"]["new_training"], [43, 44])
        self.assertEqual(
            values["prediction_support_policy"],
            "nonnegative_max_zero_v1",
        )
        self.assertEqual(values["lightgbm_convergence"]["parameters"]["n_estimators"], 5000)
        self.assertEqual(
            values["lightgbm_convergence"]["parameters"]["early_stopping_rounds"],
            200,
        )
        self.assertEqual(values["allowed_materialized_splits"], ["train", "validation"])
        assignments = load_phase5a_assignments(self.assignments_path)["assignments"]
        self.assertEqual(len(assignments), 12)
        self.assertEqual(
            len({item["experiment_id"] for item in assignments}),
            len(assignments),
        )
        self.assertEqual(
            len({item["output_location"] for item in assignments}),
            len(assignments),
        )
        scheduled = {
            item["variant"]
            for item in assignments
            if item["model_family"] in {"cell_msca", "lightgbm"}
        }
        self.assertNotIn("forward", scheduled)
        self.assertNotIn("reverse", scheduled)
        self.assertNotIn("lightgbm_tweedie", scheduled)

    def test_full_lightgbm_validation_is_closed_before_data_materialization(self) -> None:
        verified = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "disabled by default"):
            run_phase5a_lightgbm(
                verified,
                config_path=self.config_path,
                output_root=self.root / "never-created-phase5a",
                repository_root=self.root,
                expected_git_sha="a" * 40,
                model_name="lightgbm_raw",
                train_seed=42,
                run_type="convergence",
            )
        verified.protocol.tuning_data.assert_not_called()
        verified.protocol.test_data.assert_not_called()

    def test_cli_has_no_test_execution_command(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _build_parser().parse_args(
                    ["--config", str(self.config_path), "test"]
                )

    def test_kaggle_output_path_is_validation_working_only(self) -> None:
        accepted = _validate_phase5a_output_path(
            "/kaggle/working/phase5a",
            kaggle=True,
        )
        self.assertTrue(str(accepted).replace("\\", "/").endswith("/kaggle/working/phase5a"))
        with self.assertRaisesRegex(ValueError, "read-only"):
            _validate_phase5a_output_path("/kaggle/input/results", kaggle=True)
        with self.assertRaisesRegex(ValueError, "/kaggle/working"):
            _validate_phase5a_output_path("/tmp/results", kaggle=True)

    def test_expanded_npz_verification_checks_file_and_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "month.npz"
            np.savez_compressed(path, values=np.arange(6, dtype=np.float32))
            expected = {
                path.name: {
                    "size_bytes": path.stat().st_size,
                    "file_sha256": file_sha256(path),
                    "canonical_content_sha256": canonical_npz_content_sha256(path),
                }
            }
            _verify_v1_npz_files([path], expected)
            expected[path.name]["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "file SHA-256"):
                _verify_v1_npz_files([path], expected)

    def test_archive_and_expanded_input_together_fail_closed(self) -> None:
        manifest = json.loads(
            (self.root / "configs" / "v1_legacy_archive_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory) / "input"
            expanded = input_root / "expanded"
            expanded.mkdir(parents=True)
            (input_root / V1_ARCHIVE_NAME).touch()
            for item in manifest["files"]:
                (expanded / item["file_name"]).touch()
            with self.assertRaisesRegex(RuntimeError, "exactly one v1_legacy input mode"):
                with verified_v1_input(
                    input_root=input_root,
                    working_root=Path(directory) / "working",
                    manifest_path=self.root / "configs" / "v1_legacy_archive_manifest.json",
                    repository_root=self.root,
                ):
                    self.fail("ambiguous inputs must never yield a verified protocol")

    def test_paired_cluster_bootstrap_is_deterministic_and_cell_complete(self) -> None:
        cells = np.repeat(np.asarray(["a", "b", "c"]), 36)
        true = np.zeros(cells.size, dtype=np.float64)
        bidirectional = np.repeat(np.asarray([1.0, 2.0, 3.0]), 36)
        comparator = np.repeat(np.asarray([2.0, 2.0, 2.0]), 36)
        kwargs = {
            "y_true_original": true,
            "bidirectional_prediction": bidirectional,
            "comparator_prediction": comparator,
            "cell_ids": cells,
            "n_boot": 250,
            "seed": 17,
            "rows_per_cell": 36,
        }
        first = paired_cell_cluster_mae_difference(**kwargs)
        second = paired_cell_cluster_mae_difference(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["unique_cell_count"], 3)
        self.assertTrue(first["complete_cell_resampling"])
        self.assertAlmostEqual(first["estimate"], 0.0)

        with self.assertRaisesRegex(ValueError, "exactly 36 rows"):
            paired_cell_cluster_mae_difference(
                **{**kwargs, "cell_ids": cells[:-1], "y_true_original": true[:-1],
                   "bidirectional_prediction": bidirectional[:-1],
                   "comparator_prediction": comparator[:-1]}
            )

    def test_stored_prediction_metrics_and_alignment_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = dict(self.values["required_hashes"])
            first = load_stored_validation_artifact(
                _write_validation_artifact(
                    root / "first",
                    model_name="cell_msca_bidirectional",
                    train_seed=42,
                    config_sha256="1" * 64,
                    git_sha="a" * 40,
                    hashes=hashes,
                    prediction_offset=0.1,
                )
            )
            second = load_stored_validation_artifact(
                _write_validation_artifact(
                    root / "second",
                    model_name="cell_msca_token_no_attention",
                    train_seed=43,
                    config_sha256="2" * 64,
                    git_sha="b" * 40,
                    hashes=hashes,
                    prediction_offset=0.2,
                )
            )
            cells, original, log_values = _aligned_prediction_arrays([first, second])
            self.assertEqual(cells.tolist(), ["cell-a", "cell-b", "cell-c"])
            self.assertEqual(original.shape, log_values.shape)

            prediction = second.prediction_path.read_text(encoding="utf-8")
            second.prediction_path.write_text(
                prediction.replace("cell-a", "changed-cell", 1),
                encoding="utf-8",
            )
            changed = load_stored_validation_artifact(second.directory)
            with self.assertRaisesRegex(ValueError, "cell_id alignment"):
                _aligned_prediction_arrays([first, changed])

    def test_legacy_seed42_prediction_is_verified_as_projection_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_validation_artifact(
                Path(directory) / "legacy",
                model_name="cell_msca_bidirectional",
                train_seed=42,
                config_sha256="1" * 64,
                git_sha="a" * 40,
                hashes=dict(self.values["required_hashes"]),
                prediction_offset=0.1,
            )
            metrics_path = path / "validation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            result = metrics["result"]
            result["schema_version"] = LEGACY_BASELINE_RESULT_SCHEMA_VERSION
            for field in set(result) - LEGACY_BASELINE_RESULT_FIELDS:
                del result[field]
            write_metrics_json(metrics_path, metrics, overwrite=True)
            manifest_path = path / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field in (
                "prediction_support_policy",
                "pre_projection_negative_count",
                "pre_projection_negative_fraction",
                "pre_projection_minimum",
                "projection_applied_count",
            ):
                del manifest[field]
            write_metrics_json(manifest_path, manifest, overwrite=True)

            artifact = load_stored_validation_artifact(path)

            self.assertTrue(artifact.manifest["projection_identity_verified"])
            self.assertEqual(
                artifact.manifest["pre_projection_negative_count"],
                0,
            )

    def test_filtered_discovery_skips_nonrequired_name_mismatch_but_strict_rejects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_required_discovery_fixture(root)
            concat = _write_validation_artifact(
                root / "concat-mismatch",
                model_name="concat_mlp_log1p",
                train_seed=42,
                config_sha256="9" * 64,
                git_sha="a" * 40,
                hashes=dict(self.values["required_hashes"]),
                prediction_offset=0.1,
                legacy_schema=True,
            )
            metrics_path = concat / "validation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["result"]["model_name"] = "concat_mlp"
            write_metrics_json(metrics_path, metrics, overwrite=True)

            audit: dict[str, object] = {}
            artifacts = discover_validation_artifacts(
                root,
                required_keys=SEED42_REQUIRED_ARTIFACT_KEYS,
                audit=audit,
            )

            self.assertEqual(set(artifacts), SEED42_REQUIRED_ARTIFACT_KEYS)
            excluded = audit["excluded_artifacts"]
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["model_name"], "concat_mlp")
            self.assertEqual(excluded[0]["directory"], str(concat.resolve()))
            with self.assertRaisesRegex(ValueError, "model_name mismatch"):
                discover_validation_artifacts(root)

    def test_filtered_discovery_still_rejects_required_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_required_discovery_fixture(root)
            required = next(
                path
                for path in root.iterdir()
                if json.loads(
                    (path / "validation_metrics.json").read_text(encoding="utf-8")
                )["result"]["model_name"]
                == "lightgbm_raw"
            )
            manifest_path = required / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model_name"] = "changed_required_model"
            write_metrics_json(manifest_path, manifest, overwrite=True)

            with self.assertRaisesRegex(ValueError, "model_name mismatch"):
                discover_validation_artifacts(
                    root,
                    required_keys=SEED42_REQUIRED_ARTIFACT_KEYS,
                )

    def test_filtered_discovery_rejects_missing_required_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_required_discovery_fixture(root)
            missing = next(
                path
                for path in root.iterdir()
                if json.loads(
                    (path / "validation_metrics.json").read_text(encoding="utf-8")
                )["result"]["model_name"]
                == "lightgbm_log1p"
            )
            for path in missing.iterdir():
                path.unlink()
            missing.rmdir()

            with self.assertRaisesRegex(ValueError, "required validation artifacts are missing"):
                discover_validation_artifacts(
                    root,
                    required_keys=SEED42_REQUIRED_ARTIFACT_KEYS,
                )

    def test_filtered_discovery_rejects_duplicate_required_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_required_discovery_fixture(root)
            _write_validation_artifact(
                root / "duplicate-raw",
                model_name="lightgbm_raw",
                train_seed=42,
                config_sha256="8" * 64,
                git_sha="a" * 40,
                hashes=dict(self.values["required_hashes"]),
                prediction_offset=0.2,
                legacy_schema=True,
            )

            with self.assertRaisesRegex(ValueError, "duplicate validation artifact"):
                discover_validation_artifacts(
                    root,
                    required_keys=SEED42_REQUIRED_ARTIFACT_KEYS,
                )

    def test_filtered_discovery_keeps_required_test_gate_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_required_discovery_fixture(root)
            required = next(root.iterdir())
            manifest_path = required / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["test_subset_materialized"] = True
            write_metrics_json(manifest_path, manifest, overwrite=True)

            with self.assertRaisesRegex(ValueError, "closed test gate"):
                discover_validation_artifacts(
                    root,
                    required_keys=SEED42_REQUIRED_ARTIFACT_KEYS,
                )

    @unittest.skipUnless(
        os.environ.get("CELL_MSCA_SEED42_PACKAGE_ZIP"),
        "immutable seed-42 package path is required for the real discovery smoke",
    )
    def test_actual_seed42_zip_filtered_discovery_smoke(self) -> None:
        source = Path(os.environ["CELL_MSCA_SEED42_PACKAGE_ZIP"]).resolve()
        self.assertEqual(
            file_sha256(source),
            "f40aa8db3cff1a56933a69abe9397f43e8df9daa642319848c928cea4ed98555",
        )
        with tempfile.TemporaryDirectory() as directory:
            extraction = Path(directory)
            with zipfile.ZipFile(source, "r") as archive:
                for member in archive.infolist():
                    relative = PurePosixPath(member.filename.replace("\\", "/"))
                    self.assertFalse(relative.is_absolute())
                    self.assertNotIn("..", relative.parts)
                    self.assertFalse(stat.S_ISLNK(member.external_attr >> 16))
                archive.extractall(extraction)
            audit: dict[str, object] = {}
            artifacts = discover_validation_artifacts(
                extraction,
                required_keys=SEED42_REQUIRED_ARTIFACT_KEYS,
                audit=audit,
            )
            self.assertEqual(set(artifacts), SEED42_REQUIRED_ARTIFACT_KEYS)
            self.assertEqual(len(audit["excluded_artifacts"]), 5)
            self.assertIn(
                "concat_mlp",
                {row["model_name"] for row in audit["excluded_artifacts"]},
            )
            for artifact in artifacts.values():
                self.assertFalse(artifact.manifest["test_subset_materialized"])
                self.assertTrue(artifact.manifest["projection_identity_verified"])

    def test_lightgbm_atomic_model_save_cleans_temporary_file_on_failure(self) -> None:
        class FailingBooster:
            @staticmethod
            def save_model(path: str, *, num_iteration: int) -> None:
                del path, num_iteration
                raise RuntimeError("synthetic serialization failure")

        fitted = SimpleNamespace(
            estimator=SimpleNamespace(booster_=FailingBooster()),
            contract=SimpleNamespace(best_iteration=3),
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.txt"
            with self.assertRaisesRegex(RuntimeError, "serialization failure"):
                _save_and_verify_lightgbm_model(
                    destination,
                    fitted=fitted,
                    validation_features=np.zeros((2, 7)),
                    expected_final_prediction=np.ones(2),
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(
        importlib.util.find_spec("lightgbm") is not None,
        "LightGBM is required for model artifact round-trip",
    )
    def test_actual_lightgbm_model_save_reload_prediction_round_trip(self) -> None:
        import lightgbm as lgb

        rng = np.random.default_rng(42)
        features = rng.normal(size=(48, 7))
        target = np.maximum(0.0, 2.0 + features[:, 0] - features[:, 1])
        estimator = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=12,
            learning_rate=0.1,
            verbosity=-1,
            random_state=42,
        )
        estimator.fit(features, target)
        best_iteration = int(estimator.n_estimators_)
        contract = ModelContract(
            model_name="lightgbm_raw",
            config_sha256="1" * 64,
            target_transform="identity",
            target_scale=1.0,
            loss_objective="regression_l1",
            inverse_mode="none",
            smearing_factor=None,
            best_iteration=best_iteration,
        )
        fitted = FittedLightGBM(
            estimator=estimator,
            target_space="original",
            contract=contract,
        )
        expected = _original_prediction_pair(
            estimator.predict(features),
            target_scale=1.0,
            model_name="lightgbm_raw",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "fitted_model.txt"
            summary = _save_and_verify_lightgbm_model(
                destination,
                fitted=fitted,
                validation_features=features,
                expected_final_prediction=expected.pred_original,
            )
            self.assertTrue(destination.is_file())
            self.assertTrue(summary["model_reload_validation_prediction_match"])
            self.assertRegex(summary["model_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _save_and_verify_lightgbm_model(
                    destination,
                    fitted=fitted,
                    validation_features=features,
                    expected_final_prediction=expected.pred_original,
                )

    def test_three_seed_summary_uses_sample_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = dict(self.values["required_hashes"])
            artifacts = [
                load_stored_validation_artifact(
                    _write_validation_artifact(
                        root / f"seed-{seed}",
                        model_name="cell_msca_bidirectional",
                        train_seed=seed,
                        config_sha256=str(seed)[-1] * 64,
                        git_sha="a" * 40,
                        hashes=hashes,
                        prediction_offset=offset,
                    )
                )
                for seed, offset in zip((42, 43, 44), (0.1, 0.2, 0.3))
            ]
            summary = _seed_metric_summary(artifacts, ddof=1)
            self.assertEqual(set(summary["per_seed"]), {"42", "43", "44"})
            self.assertAlmostEqual(summary["aggregate"]["mae"]["mean"], 0.2)
            self.assertAlmostEqual(summary["aggregate"]["mae"]["std"], 0.1)

    def test_lightgbm_selection_is_frozen_and_provenance_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = dict(self.values["required_hashes"])
            configuration_hashes = {}
            for model_name in ("lightgbm_raw", "lightgbm_log1p"):
                configuration_hashes[model_name] = canonical_sha256(
                    _lightgbm_config(self.values, model_name=model_name).to_dict(
                        train_seed=42,
                        target_scale=float(self.values["data"]["target_scale"]),
                    )
                )
            raw = _write_validation_artifact(
                root / "raw",
                model_name="lightgbm_raw",
                train_seed=42,
                config_sha256=configuration_hashes["lightgbm_raw"],
                git_sha="a" * 40,
                hashes=hashes,
                prediction_offset=0.1,
                run_type="convergence",
            )
            log1p = _write_validation_artifact(
                root / "log1p",
                model_name="lightgbm_log1p",
                train_seed=42,
                config_sha256=configuration_hashes["lightgbm_log1p"],
                git_sha="a" * 40,
                hashes=hashes,
                prediction_offset=0.2,
                run_type="convergence",
            )
            selection_path = root / "selection.json"
            freeze_lightgbm_convergence_selection(
                raw_run_dir=raw,
                log1p_run_dir=log1p,
                config_path=self.config_path,
                output_path=selection_path,
                expected_git_sha="a" * 40,
            )
            selection = load_lightgbm_selection(
                selection_path,
                config_path=self.config_path,
            )
            self.assertEqual(selection["selected_model_name"], "lightgbm_raw")
            self.assertFalse(selection["test_subset_materialized"])

            tampered = json.loads(json.dumps(selection))
            tampered["selected_model_name"] = "lightgbm_log1p"
            selection_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_lightgbm_selection(selection_path, config_path=self.config_path)

            for invalid_mae in (
                float("nan"),
                float("inf"),
                float("-inf"),
                "1.0",
                True,
            ):
                with self.subTest(invalid_mae=invalid_mae):
                    invalid = json.loads(json.dumps(selection))
                    invalid["candidate_results"]["lightgbm_raw"][
                        "validation_original_unit_mae"
                    ] = invalid_mae
                    selection_path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "finite number"):
                        load_lightgbm_selection(
                            selection_path,
                            config_path=self.config_path,
                        )

            tie = json.loads(json.dumps(selection))
            tie["candidate_results"]["lightgbm_log1p"][
                "validation_original_unit_mae"
            ] = tie["candidate_results"]["lightgbm_raw"][
                "validation_original_unit_mae"
            ]
            selection_path.write_text(json.dumps(tie), encoding="utf-8")
            tied_selection = load_lightgbm_selection(
                selection_path,
                config_path=self.config_path,
            )
            self.assertEqual(tied_selection["selected_model_name"], "lightgbm_raw")

            wrong_tie_winner = json.loads(json.dumps(tie))
            wrong_tie_winner["selected_model_name"] = "lightgbm_log1p"
            selection_path.write_text(json.dumps(wrong_tie_winner), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_lightgbm_selection(selection_path, config_path=self.config_path)

            missing_candidate = json.loads(json.dumps(selection))
            del missing_candidate["candidate_results"]["lightgbm_log1p"]
            selection_path.write_text(json.dumps(missing_candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be exactly"):
                load_lightgbm_selection(selection_path, config_path=self.config_path)

            invalid_schema = json.loads(json.dumps(selection))
            del invalid_schema["candidate_results"]["lightgbm_raw"][
                "actual_iterations"
            ]
            selection_path.write_text(json.dumps(invalid_schema), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate schema"):
                load_lightgbm_selection(selection_path, config_path=self.config_path)

            selection["required_hashes"]["data_sha256"] = "f" * 64
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance hashes"):
                load_lightgbm_selection(selection_path, config_path=self.config_path)

    def test_lightgbm_parameters_preserve_phase3_conditions_except_cap_and_patience(self) -> None:
        old = LightGBMConfig(
            model_name="lightgbm_raw",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            early_stopping_rounds=50,
        )
        new = _lightgbm_config(self.values, model_name="lightgbm_raw")
        old_values = old.to_dict(train_seed=42, target_scale=1.0)
        new_values = new.to_dict(train_seed=42, target_scale=1.0)
        changed = {
            key for key in old_values if old_values[key] != new_values[key]
        }
        self.assertEqual(changed, {"n_estimators", "early_stopping_rounds"})

    def test_actual_lightgbm_iterations_use_documented_api_and_fail_closed(self) -> None:
        estimator = SimpleNamespace(n_estimators_=100, n_iter_=99)
        self.assertEqual(_lightgbm_actual_iterations(estimator), 100)
        at_cap = _lightgbm_iteration_summary(
            estimator,
            best_iteration=80,
            configured_n_estimators=100,
        )
        self.assertEqual(at_cap["best_iteration"], 80)
        self.assertEqual(at_cap["actual_iterations"], 100)
        self.assertTrue(at_cap["maximum_iteration_reached"])

        early_stopped = _lightgbm_iteration_summary(
            SimpleNamespace(n_iter_=70),
            best_iteration=60,
            configured_n_estimators=100,
        )
        self.assertEqual(early_stopped["actual_iterations"], 70)
        self.assertFalse(early_stopped["maximum_iteration_reached"])

        booster_fallback = SimpleNamespace(
            booster_=SimpleNamespace(current_iteration=lambda: 37)
        )
        self.assertEqual(_lightgbm_actual_iterations(booster_fallback), 37)
        with self.assertRaisesRegex(RuntimeError, "unable to determine"):
            _lightgbm_actual_iterations(SimpleNamespace())

    def test_lightgbm_run_persists_actual_iterations_and_closed_test_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hashes = dict(self.values["required_hashes"])
            target = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
            target_log = np.log1p(target)
            config_sha256 = canonical_sha256(
                _lightgbm_config(self.values, model_name="lightgbm_raw").to_dict(
                    train_seed=42,
                    target_scale=float(self.values["data"]["target_scale"]),
                )
            )
            context = BaselineResultContext(
                model_name="lightgbm_raw",
                data_version="v1_legacy",
                data_sha256=hashes["data_sha256"],
                split_sha256=hashes["split_sha256"],
                split_config_sha256=hashes["split_config_sha256"],
                config_sha256=config_sha256,
                preprocessing_sha256=hashes["preprocessing_sha256"],
                split_seed=42,
                train_seed=42,
                evaluation_split="validation",
                target_transform="identity",
                target_scale=float(self.values["data"]["target_scale"]),
                loss_objective="regression_l1",
                inverse_mode="none",
                best_iteration=4800,
            )
            result = evaluate_baseline_predictions(
                context,
                y_true_original=target,
                y_pred_original=target,
                y_true_log=target_log,
                y_pred_log=target_log,
            )
            validation = SimpleNamespace(
                features=np.zeros((3, 7), dtype=np.float64),
                target_original=target,
                target_log=target_log,
                cell_ids=np.asarray(["cell-a", "cell-b", "cell-c"]),
            )
            provenance = SimpleNamespace(
                **hashes,
                split_seed=42,
                train_seed=42,
            )
            data = SimpleNamespace(validation=validation, provenance=provenance)
            protocol = SimpleNamespace(
                provenance=provenance,
                tuning_data=mock.Mock(return_value=data),
                test_data=mock.Mock(),
            )
            verified = SimpleNamespace(
                protocol=protocol,
                kaggle=False,
                input_mode="synthetic-test",
                summary={"artifact_classification": "engineering-only"},
            )
            fitted = SimpleNamespace(estimator=SimpleNamespace(n_estimators_=5000))
            evaluation = SimpleNamespace(
                result=result,
                predictions=SimpleNamespace(
                    pred_original=target,
                    pred_log=target_log,
                    unprojected_original=target,
                ),
            )
            git_sha = "a" * 40
            identity = GitIdentity(
                commit_sha=git_sha,
                worktree_dirty=False,
                dirty_state_policy="tracked_and_untracked_clean",
                source="git",
            )
            with (
                mock.patch(
                    "cell_msca.phase5a.fit_lightgbm_baseline",
                    return_value=fitted,
                ),
                mock.patch(
                    "cell_msca.phase5a.evaluate_fitted_baseline",
                    return_value=evaluation,
                ),
                mock.patch(
                    "cell_msca.phase5a._save_and_verify_lightgbm_model",
                    return_value={
                        "model_artifact": "fitted_model.txt",
                        "model_sha256": "f" * 64,
                        "model_reload_validation_prediction_match": True,
                        "model_reload_maximum_absolute_prediction_difference": 0.0,
                        "model_saved_best_iteration": 4800,
                    },
                ),
            ):
                output = run_phase5a_lightgbm(
                    verified,
                    config_path=self.config_path,
                    output_root=Path(directory) / "runs",
                    repository_root=self.root,
                    expected_git_sha=git_sha,
                    model_name="lightgbm_raw",
                    train_seed=42,
                    run_type="convergence",
                    allow_full_validation=True,
                    git_identity=identity,
                )
            metrics = json.loads(
                (output / "validation_metrics.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["actual_iterations"], 5000)
            self.assertEqual(manifest["best_iteration"], 4800)
            self.assertEqual(manifest["actual_iterations"], 5000)
            self.assertTrue(manifest["maximum_iteration_reached"])
            self.assertEqual(
                manifest["prediction_support_policy"],
                "nonnegative_max_zero_v1",
            )
            self.assertEqual(manifest["pre_projection_negative_count"], 0)
            self.assertTrue(manifest["model_reload_validation_prediction_match"])
            self.assertFalse(manifest["test_subset_materialized"])
            self.assertFalse(manifest["test_evaluation_performed"])
            protocol.test_data.assert_not_called()

    def test_synthetic_three_seed_aggregation_recomputes_and_bootstraps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed42_root = root / "seed42"
            new_root = root / "new"
            hashes = dict(self.values["required_hashes"])
            seed42_git = self.values["required_base_git_sha"]
            new_git = "b" * 40
            seed42_offsets = self.values["seed42_reference_mae_rounded_6dp"]
            for model_name, offset in seed42_offsets.items():
                _write_validation_artifact(
                    seed42_root / model_name,
                    model_name=model_name,
                    train_seed=42,
                    config_sha256=self.values[
                        "seed42_expected_configuration_sha256"
                    ][model_name],
                    git_sha=seed42_git,
                    hashes=hashes,
                    prediction_offset=float(offset),
                    rows_per_cell=36,
                    legacy_schema=True,
                )
            excluded_concat = _write_validation_artifact(
                seed42_root / "concat-mismatch",
                model_name="concat_mlp_log1p",
                train_seed=42,
                config_sha256="9" * 64,
                git_sha=seed42_git,
                hashes=hashes,
                prediction_offset=0.5,
                rows_per_cell=36,
                legacy_schema=True,
            )
            excluded_metrics_path = excluded_concat / "validation_metrics.json"
            excluded_metrics = json.loads(
                excluded_metrics_path.read_text(encoding="utf-8")
            )
            excluded_metrics["result"]["model_name"] = "concat_mlp"
            write_metrics_json(
                excluded_metrics_path,
                excluded_metrics,
                overwrite=True,
            )

            convergence_paths = {}
            for model_name, offset in (
                ("lightgbm_raw", 0.1),
                ("lightgbm_log1p", 0.2),
            ):
                config_hash = canonical_sha256(
                    _lightgbm_config(self.values, model_name=model_name).to_dict(
                        train_seed=42,
                        target_scale=float(self.values["data"]["target_scale"]),
                    )
                )
                convergence_paths[model_name] = _write_validation_artifact(
                    new_root / f"{model_name}-42",
                    model_name=model_name,
                    train_seed=42,
                    config_sha256=config_hash,
                    git_sha=new_git,
                    hashes=hashes,
                    prediction_offset=offset,
                    run_type="convergence",
                    rows_per_cell=36,
                )
            selection_path = root / "selection.json"
            freeze_lightgbm_convergence_selection(
                raw_run_dir=convergence_paths["lightgbm_raw"],
                log1p_run_dir=convergence_paths["lightgbm_log1p"],
                config_path=self.config_path,
                output_path=selection_path,
                expected_git_sha=new_git,
            )

            from cell_msca.kaggle_runner import load_kaggle_validation_config

            for seed in (43, 44):
                runtime = load_kaggle_validation_config(
                    self.root / "configs" / f"phase5a_cell_msca_seed{seed}.json"
                )
                configuration_hashes = runtime["required_hashes"][
                    "configuration_sha256_by_variant_and_device"
                ]
                for variant, offset in (
                    ("bidirectional", 0.2 + seed / 1000.0),
                    ("token_no_attention", 0.3 + seed / 1000.0),
                ):
                    _write_validation_artifact(
                        new_root / f"{variant}-{seed}",
                        model_name=f"cell_msca_{variant}",
                        train_seed=seed,
                        config_sha256=configuration_hashes[f"{variant}:cuda"],
                        git_sha=new_git,
                        hashes=hashes,
                        prediction_offset=offset,
                        rows_per_cell=36,
                    )
                lightgbm_hash = canonical_sha256(
                    _lightgbm_config(
                        self.values,
                        model_name="lightgbm_raw",
                    ).to_dict(train_seed=seed, target_scale=1.0)
                )
                _write_validation_artifact(
                    new_root / f"lightgbm-raw-{seed}",
                    model_name="lightgbm_raw",
                    train_seed=seed,
                    config_sha256=lightgbm_hash,
                    git_sha=new_git,
                    hashes=hashes,
                    prediction_offset=0.4 + seed / 1000.0,
                    rows_per_cell=36,
                )

            output = aggregate_phase5a_results(
                seed42_package_root=seed42_root,
                phase5a_results_root=new_root,
                lightgbm_selection_path=selection_path,
                config_path=self.config_path,
                repository_root=self.root,
                output_dir=root / "aggregate",
                expected_new_git_sha=new_git,
            )
            manifest = json.loads(
                (output / "aggregation_manifest.json").read_text(encoding="utf-8")
            )
            bootstrap = json.loads(
                (output / "paired_cell_cluster_bootstrap.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["prediction_metrics_recalculated"])
            self.assertTrue(
                manifest["seed42_artifact_discovery"]["required_filter_applied"]
            )
            self.assertEqual(
                len(manifest["seed42_artifact_discovery"]["loaded_keys"]),
                4,
            )
            self.assertEqual(
                manifest["seed42_artifact_discovery"]["excluded_artifacts"][0][
                    "model_name"
                ],
                "concat_mlp",
            )
            self.assertFalse(manifest["test_subset_materialized"])
            for comparison in bootstrap["comparisons"].values():
                self.assertTrue(comparison["complete_cell_resampling"])
                self.assertEqual(comparison["rows_per_cell"], 36)

    def test_lightgbm_experiment_id_is_deterministic_and_hash_sensitive(self) -> None:
        arguments = {
            "run_type": "convergence",
            "model_name": "lightgbm_raw",
            "seed": 42,
            "git_commit_sha": "a" * 40,
        }
        first = _lightgbm_experiment_id(self.values, **arguments)
        self.assertEqual(first, _lightgbm_experiment_id(self.values, **arguments))
        self.assertNotEqual(
            first,
            _lightgbm_experiment_id(
                self.values,
                **{**arguments, "git_commit_sha": "b" * 40},
            ),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is required to validate frozen Cell-MSCA configuration hashes",
    )
    def test_seed43_and_seed44_cell_configuration_hashes_match(self) -> None:
        from cell_msca.kaggle_runner import (
            _build_training_config,
            load_kaggle_validation_config,
        )

        for seed in (43, 44):
            path = self.root / "configs" / f"phase5a_cell_msca_seed{seed}.json"
            values = load_kaggle_validation_config(path)
            expected = values["required_hashes"][
                "configuration_sha256_by_variant_and_device"
            ]
            for key, expected_hash in expected.items():
                variant, device = key.split(":", maxsplit=1)
                config = _build_training_config(
                    values,
                    variant=variant,
                    device=device,
                )
                actual = canonical_sha256(
                    config.to_dict(
                        train_seed=seed,
                        target_scale=float(values["data"]["target_scale"]),
                    )
                )
                self.assertEqual(actual, expected_hash, msg=f"seed={seed}, key={key}")


if __name__ == "__main__":
    unittest.main()
