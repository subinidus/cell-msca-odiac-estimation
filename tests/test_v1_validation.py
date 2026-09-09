"""Frozen v1_legacy validation-package regression tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cell_msca.kaggle_runner import load_kaggle_validation_config
from cell_msca.v1_validation import (
    V1_ARCHIVE_NAME,
    _load_frozen_baseline_values,
    _validate_month_coverage,
    discover_v1_archive,
    load_frozen_v1_assignments,
    run_frozen_baseline_validation,
    run_frozen_cell_validation,
)


class V1LegacyValidationPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.configs = cls.root / "configs"
        cls.manifest_path = cls.configs / "v1_legacy_archive_manifest.json"
        cls.assignments_path = (
            cls.configs / "v1_legacy_validation_assignments.json"
        )
        cls.baseline_full_path = (
            cls.configs / "v1_legacy_baseline_validation.json"
        )
        cls.baseline_smoke_path = cls.configs / "v1_legacy_baseline_smoke.json"
        cls.cell_full_path = (
            cls.configs / "v1_legacy_cell_msca_validation.json"
        )

    def test_archive_discovery_requires_exactly_one_named_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                discover_v1_archive(root)
            first = root / "first" / V1_ARCHIVE_NAME
            first.parent.mkdir()
            first.touch()
            self.assertEqual(discover_v1_archive(root), first)
            second = root / "second" / V1_ARCHIVE_NAME
            second.parent.mkdir()
            second.touch()
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                discover_v1_archive(root)

    def test_archive_manifest_freezes_36_months_and_npz_hashes(self) -> None:
        values = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        files = values["files"]
        self.assertEqual(len(files), 36)
        names = [item["file_name"] for item in files]
        _validate_month_coverage(names)
        self.assertEqual(len(set(names)), 36)
        for item in files:
            self.assertGreater(item["size_bytes"], 0)
            self.assertRegex(item["file_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(
                item["canonical_content_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_baseline_configs_are_frozen_and_hash_bound(self) -> None:
        full_config, full_values = _load_frozen_baseline_values(
            self.baseline_full_path
        )
        smoke_config, smoke_values = _load_frozen_baseline_values(
            self.baseline_smoke_path
        )
        self.assertEqual(full_config.split_config.split_seed, 42)
        self.assertEqual(smoke_config.train_seed, 42)
        self.assertEqual(
            full_values["required_hashes"],
            smoke_values["required_hashes"],
        )
        self.assertEqual(
            full_values["allowed_materialized_splits"],
            ["train", "validation"],
        )
        self.assertEqual(
            full_values["execution_policy"]["default_action"],
            "integrity_only",
        )

    def test_assignments_have_unique_ids_and_output_locations(self) -> None:
        values = load_frozen_v1_assignments(self.assignments_path)
        assignments = values["assignments"]
        self.assertEqual(len(assignments), 10)
        self.assertEqual(
            len({item["experiment_id"] for item in assignments}),
            10,
        )
        self.assertEqual(
            len({item["output_location"] for item in assignments}),
            10,
        )
        unsupported = [
            item for item in assignments if item["status"] == "unsupported"
        ]
        self.assertEqual(
            [item["experiment_id"] for item in unsupported],
            ["concat_mlp_raw_huber_seed42"],
        )

    def test_duplicate_assignment_output_is_rejected(self) -> None:
        values = json.loads(self.assignments_path.read_text(encoding="utf-8"))
        values["assignments"][1]["output_location"] = values["assignments"][0][
            "output_location"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignments.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate assignment output"):
                load_frozen_v1_assignments(path)

    def test_full_baseline_validation_is_closed_by_default(self) -> None:
        verified = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "disabled by default"):
            run_frozen_baseline_validation(
                verified,
                model_name="train_mean",
                config_path=self.baseline_full_path,
                output_root=self.root / "never-created",
                repository_root=self.root,
                expected_git_sha="a" * 40,
            )
        self.assertFalse(verified.protocol.tuning_data.called)
        self.assertFalse(verified.protocol.test_data.called)

    def test_full_cell_validation_is_closed_before_runner_call(self) -> None:
        verified = mock.Mock()
        with mock.patch(
            "cell_msca.v1_validation.run_kaggle_validation"
        ) as runner:
            with self.assertRaisesRegex(RuntimeError, "disabled by default"):
                run_frozen_cell_validation(
                    verified,
                    variant="forward",
                    config_path=self.cell_full_path,
                    output_root=self.root / "never-created",
                    repository_root=self.root,
                    expected_git_sha="a" * 40,
                    device="cpu",
                )
            runner.assert_not_called()

    def test_cell_configs_are_validation_only_and_hash_bound(self) -> None:
        smoke_path = self.configs / "v1_legacy_cell_msca_1epoch_smoke.json"
        for path in (self.cell_full_path, smoke_path):
            values = load_kaggle_validation_config(path)
            self.assertEqual(values["stage"], "validation_only")
            self.assertEqual(
                values["allowed_materialized_splits"],
                ["train", "validation"],
            )
            self.assertEqual(values["frozen_train_seed"], 42)
            self.assertEqual(
                len(
                    values["required_hashes"][
                        "configuration_sha256_by_variant_and_device"
                    ]
                ),
                8,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is required to construct Cell-MSCA training configs",
    )
    def test_cell_configuration_hashes_match_runtime_contract(self) -> None:
        from cell_msca.data import canonical_sha256
        from cell_msca.kaggle_runner import _build_training_config

        smoke_path = self.configs / "v1_legacy_cell_msca_1epoch_smoke.json"
        for path in (self.cell_full_path, smoke_path):
            values = load_kaggle_validation_config(path)
            expected = values["required_hashes"][
                "configuration_sha256_by_variant_and_device"
            ]
            for key, expected_digest in expected.items():
                variant, device = key.split(":", maxsplit=1)
                config = _build_training_config(
                    values,
                    variant=variant,
                    device=device,
                )
                actual = canonical_sha256(
                    config.to_dict(
                        train_seed=values["frozen_train_seed"],
                        target_scale=values["data"]["target_scale"],
                    )
                )
                self.assertEqual(actual, expected_digest, msg=f"{path}: {key}")


if __name__ == "__main__":
    unittest.main()
