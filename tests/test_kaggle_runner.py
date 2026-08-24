"""Regression tests for the validation-only Kaggle execution layer."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - dependency-error path
    torch = None  # type: ignore[assignment]

from cell_msca.baselines import TestEvaluationBlockedError, TuningData
from cell_msca.kaggle_runner import (
    ARTIFACT_CLASSIFICATIONS,
    EXPERIMENT_ASSIGNMENT_SCHEMA_VERSION,
    GitIdentity,
    deterministic_experiment_id,
    load_experiment_assignments,
    load_kaggle_validation_config,
    resolve_git_identity,
    run_kaggle_validation,
    validate_execution_paths,
    validate_experiment_assignments,
)


class KaggleRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.smoke_config = cls.root / "configs" / "kaggle_synthetic_smoke.json"
        cls.assignment_config = (
            cls.root / "configs" / "experiment_assignment.example.json"
        )

    def test_kaggle_paths_enforce_read_only_input_and_working_output(self) -> None:
        validate_execution_paths(
            data_path="/kaggle/input/private-data",
            output_path="/kaggle/working/cell-msca",
            data_mode="npz",
            kaggle=True,
        )
        validate_execution_paths(
            data_path="generated",
            output_path="/kaggle/working/cell-msca",
            data_mode="synthetic",
            kaggle=True,
        )
        with self.assertRaisesRegex(ValueError, "read-only /kaggle/input"):
            validate_execution_paths(
                data_path="/kaggle/input/private-data",
                output_path="/kaggle/input/private-data/output",
                data_mode="npz",
                kaggle=True,
            )
        with self.assertRaisesRegex(ValueError, "under /kaggle/working"):
            validate_execution_paths(
                data_path="/kaggle/input/private-data",
                output_path="/tmp/output",
                data_mode="npz",
                kaggle=True,
            )
        with self.assertRaisesRegex(ValueError, "must be under /kaggle/input"):
            validate_execution_paths(
                data_path="/tmp/private-data",
                output_path="/kaggle/working/output",
                data_mode="npz",
                kaggle=True,
            )

    def test_experiment_id_is_deterministic_and_contract_sensitive(self) -> None:
        values = {
            "owner": "owner-a",
            "variant": "forward",
            "train_seed": 3407,
            "split_seed": 42,
            "device": "cpu",
            "data_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "preprocessing_sha256": "3" * 64,
            "configuration_sha256": "4" * 64,
            "git_commit_sha": "a" * 40,
        }
        first = deterministic_experiment_id(**values)
        second = deterministic_experiment_id(**values)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            deterministic_experiment_id(**{**values, "train_seed": 2026}),
        )
        self.assertNotEqual(
            first,
            deterministic_experiment_id(**{**values, "variant": "reverse"}),
        )

    def test_config_and_assignment_templates_are_frozen_and_unique(self) -> None:
        config = load_kaggle_validation_config(self.smoke_config)
        self.assertEqual(config["stage"], "validation_only")
        self.assertEqual(
            config["allowed_materialized_splits"],
            ["train", "validation"],
        )
        configuration_hashes = config["required_hashes"][
            "configuration_sha256_by_variant_and_device"
        ]
        self.assertEqual(len(configuration_hashes), 8)
        assignments = load_experiment_assignments(self.assignment_config)
        self.assertEqual(len(assignments["assignments"]), 4)

        duplicate = json.loads(json.dumps(assignments))
        duplicate["assignments"][1]["experiment_id"] = duplicate["assignments"][0][
            "experiment_id"
        ]
        with self.assertRaisesRegex(ValueError, "duplicate experiment IDs"):
            validate_experiment_assignments(duplicate)

    def test_config_structurally_rejects_test_materialization(self) -> None:
        original = load_kaggle_validation_config(self.smoke_config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for mutation in ("stage", "splits"):
                values = json.loads(json.dumps(original))
                if mutation == "stage":
                    values["stage"] = "test"
                else:
                    values["allowed_materialized_splits"].append("test")
                path.write_text(json.dumps(values), encoding="utf-8")
                with self.subTest(mutation=mutation):
                    with self.assertRaises(TestEvaluationBlockedError):
                        load_kaggle_validation_config(path)
            values = json.loads(json.dumps(original))
            values["data"]["test_path"] = "forbidden"
            path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaises(TestEvaluationBlockedError):
                load_kaggle_validation_config(path)

    def test_attached_code_sha_file_supports_source_without_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = root / "GIT_COMMIT_SHA.txt"
            declaration.write_text("a" * 40 + "\n", encoding="utf-8")
            error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
            with mock.patch(
                "cell_msca.train.discover_git_state",
                side_effect=error,
            ):
                identity = resolve_git_identity(
                    root,
                    expected_git_sha="a" * 40,
                    source_git_sha_file=declaration,
                )
            self.assertEqual(identity.commit_sha, "a" * 40)
            self.assertEqual(identity.source, "attached_private_code_dataset")
            self.assertFalse(identity.worktree_dirty)
            self.assertEqual(
                identity.dirty_state_policy,
                "attached_code_dataset_exact_sha_file_no_worktree_status",
            )

    def test_notebook_is_thin_has_no_outputs_and_contains_no_install_step(self) -> None:
        notebook_path = (
            self.root / "notebooks" / "cell_msca_kaggle_validation.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertIn("cell_msca.kaggle_runner", code)
        self.assertIn("git', 'checkout', '--detach'", code)
        self.assertNotIn("pip install", code.lower())
        self.assertNotIn("api_token", code.lower())
        self.assertNotIn("kaggle.json", code.lower())

    def test_exact_configuration_and_git_hashes_are_required(self) -> None:
        original = load_kaggle_validation_config(self.smoke_config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "changed-config.json"
            values = json.loads(json.dumps(original))
            values["required_hashes"][
                "configuration_sha256_by_variant_and_device"
            ]["forward:cpu"] = "f" * 64
            config_path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration_sha256 mismatch"):
                run_kaggle_validation(
                    variant="forward",
                    train_seed=3407,
                    config_path=config_path,
                    data_path="generated",
                    output_path=root / "output",
                    device="cpu",
                    expected_git_sha="a" * 40,
                    repository_root=self.root,
                    git_identity=GitIdentity(
                        "a" * 40,
                        False,
                        "tracked_and_untracked_files",
                        "unit_test",
                    ),
                )
            with self.assertRaisesRegex(ValueError, "Git identity"):
                run_kaggle_validation(
                    variant="forward",
                    train_seed=3407,
                    config_path=self.smoke_config,
                    data_path="generated",
                    output_path=root / "output-2",
                    device="cpu",
                    expected_git_sha="b" * 40,
                    repository_root=self.root,
                    git_identity=GitIdentity(
                        "a" * 40,
                        False,
                        "tracked_and_untracked_files",
                        "unit_test",
                    ),
                )
            values = json.loads(json.dumps(original))
            values["required_hashes"]["data_sha256"] = "f" * 64
            config_path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data_sha256 mismatch"):
                run_kaggle_validation(
                    variant="forward",
                    train_seed=3407,
                    config_path=config_path,
                    data_path="generated",
                    output_path=root / "output-3",
                    device="cpu",
                    expected_git_sha="a" * 40,
                    repository_root=self.root,
                    git_identity=GitIdentity(
                        "a" * 40,
                        False,
                        "tracked_and_untracked_files",
                        "unit_test",
                    ),
                )
            failed_manifests = list((root / "output-3").glob("*/run_manifest.json"))
            self.assertEqual(len(failed_manifests), 1)
            failed = json.loads(failed_manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertFalse(failed["test_subset_materialized"])

    @unittest.skipIf(torch is None, "Kaggle runner smoke requires PyTorch")
    def test_synthetic_run_writes_complete_manifest_and_keeps_test_closed(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        identity = GitIdentity(
            "a" * 40,
            False,
            "attached_code_dataset_exact_sha_file_no_worktree_status",
            "attached_private_code_dataset",
        )
        subset_names: list[str] = []
        original_subset = TuningData.subset

        def record_subset(data: TuningData, name: str):
            subset_names.append(name)
            return original_subset(data, name)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(TuningData, "subset", new=record_subset):
                artifacts = run_kaggle_validation(
                    variant="forward",
                    train_seed=3407,
                    config_path=self.smoke_config,
                    data_path="generated",
                    output_path=directory,
                    device="cpu",
                    expected_git_sha="a" * 40,
                    repository_root=self.root,
                    git_identity=identity,
                )
            self.assertEqual(subset_names, ["train", "validation"])
            self.assertEqual(
                {path.name for path in artifacts.output_dir.iterdir()},
                set(ARTIFACT_CLASSIFICATIONS),
            )
            manifest = json.loads(artifacts.run_manifest_json.read_text("utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(manifest["test_subset_materialized"])
            self.assertEqual(
                manifest["git_dirty_state_policy"],
                "attached_code_dataset_exact_sha_file_no_worktree_status",
            )
            self.assertEqual(
                manifest["allowed_materialized_splits"],
                ["train", "validation"],
            )
            for name, classification in ARTIFACT_CLASSIFICATIONS.items():
                self.assertEqual(
                    manifest["artifacts"][name]["artifact_classification"],
                    classification,
                )
            required_manifest_fields = {
                "owner",
                "model_name",
                "variant",
                "train_seed",
                "split_seed",
                "data_sha256",
                "split_sha256",
                "preprocessing_sha256",
                "configuration_sha256",
                "git_commit_sha",
                "runtime_versions",
                "start_time_utc",
                "completion_time_utc",
                "status",
            }
            self.assertTrue(required_manifest_fields <= set(manifest))
            metrics = json.loads(artifacts.validation_metrics_json.read_text("utf-8"))
            self.assertEqual(metrics["artifact_classification"], "validation-only")
            self.assertEqual(
                metrics["prediction_metric_verification"],
                "passed_including_spearman",
            )
            self.assertGreater(artifacts.selected_checkpoint.stat().st_size, 0)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_kaggle_validation(
                    variant="forward",
                    train_seed=3407,
                    config_path=self.smoke_config,
                    data_path="generated",
                    output_path=directory,
                    device="cpu",
                    expected_git_sha="a" * 40,
                    repository_root=self.root,
                    git_identity=identity,
                )

    def test_assignment_schema_name_is_stable(self) -> None:
        self.assertEqual(
            EXPERIMENT_ASSIGNMENT_SCHEMA_VERSION,
            "cell_msca.experiment_assignment.v1",
        )


if __name__ == "__main__":
    unittest.main()
