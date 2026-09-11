"""Regression tests for the frozen Phase 5A validation-robustness contract."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cell_msca.baselines import LightGBMConfig
from cell_msca.data import canonical_npz_content_sha256, canonical_sha256, file_sha256
from cell_msca.evaluate import (
    BaselineResultContext,
    evaluate_baseline_predictions,
    write_metrics_json,
    write_prediction_csv,
)
from cell_msca.phase5a import (
    _aligned_prediction_arrays,
    _build_parser,
    _lightgbm_experiment_id,
    _lightgbm_config,
    _seed_metric_summary,
    _validate_phase5a_output_path,
    aggregate_phase5a_results,
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
        {"artifact_classification": "validation-only", "result": result},
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
            "git_commit_sha": git_sha,
            "allowed_materialized_splits": ["train", "validation"],
            "test_subset_materialized": False,
            "test_evaluation_performed": False,
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

    def test_frozen_config_and_assignments_reject_scope_expansion(self) -> None:
        values = self.values
        self.assertEqual(values["seeds"]["new_training"], [43, 44])
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
