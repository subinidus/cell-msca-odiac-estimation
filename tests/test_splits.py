from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cell_msca.data import (
    PREPROCESSING_SCHEMA_VERSION,
    CellDataset,
    fit_train_preprocessing,
)
from cell_msca.splits import (
    SPLIT_SCHEMA_VERSION,
    BufferedBlockSplitConfig,
    CellFixedSplitConfig,
    SeedConfig,
    assess_buffered_block_split,
    build_split_subsets,
    create_persistent_buffered_block_split,
    create_persistent_cell_fixed_split,
    load_persistent_split,
    verify_split_integrity,
)
from tests.synthetic_npz import (
    synthetic_arrays,
    write_synthetic_months,
    write_synthetic_npz,
)


class PersistentSplitTests(unittest.TestCase):
    def test_cell_fixed_split_persists_reloads_and_keeps_all_months_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=3, height=4, width=4)
            )
            split_path = root / "splits" / "cell_fixed_seed42.csv"
            metadata_path = root / "splits" / "cell_fixed_seed42.metadata.json"
            config = CellFixedSplitConfig(
                split_seed=42,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            created = create_persistent_cell_fixed_split(
                dataset,
                split_path,
                metadata_path,
                config=config,
            )

            reloaded = load_persistent_split(
                dataset,
                split_path,
                metadata_json=metadata_path,
                config=config,
            )
            integrity = verify_split_integrity(dataset, reloaded)
            subsets = build_split_subsets(dataset, reloaded)

            self.assertEqual(created.split_sha256, reloaded.split_sha256)
            self.assertEqual(set(integrity["overlaps"].values()), {0})
            self.assertEqual(sum(len(subset) for subset in subsets.values()), len(dataset))
            for split, subset in subsets.items():
                expected_cells = reloaded.cell_ids(split)
                self.assertEqual(
                    {subset[index]["cell_id"] for index in range(len(subset))},
                    set(expected_cells),
                )
                for cell_id in expected_cells:
                    months = [
                        sample["month_id"]
                        for sample in (subset[index] for index in range(len(subset)))
                        if sample["cell_id"] == cell_id
                    ]
                    self.assertEqual(len(months), dataset.n_months)

    def test_train_seed_does_not_change_split_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=2, height=4, width=4)
            )
            first_config = CellFixedSplitConfig.from_seeds(
                SeedConfig(split_seed=42, train_seed=42),
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            second_config = CellFixedSplitConfig.from_seeds(
                SeedConfig(split_seed=42, train_seed=3407),
                validation_ratio=0.25,
                test_ratio=0.25,
            )

            first = create_persistent_cell_fixed_split(
                dataset,
                root / "first.csv",
                root / "first.json",
                config=first_config,
            )
            second = create_persistent_cell_fixed_split(
                dataset,
                root / "second.csv",
                root / "second.json",
                config=second_config,
            )

            self.assertEqual(first_config, second_config)
            self.assertEqual(first.split_sha256, second.split_sha256)

    def test_metadata_records_data_config_split_and_per_split_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=2, height=4, width=4)
            )
            split_path = root / "split.csv"
            metadata_path = root / "metadata.json"
            config = CellFixedSplitConfig(
                split_seed=42,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            manifest = create_persistent_cell_fixed_split(
                dataset,
                split_path,
                metadata_path,
                config=config,
            )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["data_sha256"], dataset.data_sha256)
            self.assertEqual(metadata["split_sha256"], manifest.split_sha256)
            self.assertEqual(metadata["schema_version"], SPLIT_SCHEMA_VERSION)
            self.assertEqual(len(metadata["config_sha256"]), 64)
            for split in ("train", "validation", "test"):
                split_metadata = metadata["counts"]["by_split"][split]
                self.assertGreater(split_metadata["cell_count"], 0)
                self.assertEqual(
                    split_metadata["sample_count"],
                    split_metadata["cell_count"] * dataset.n_months,
                )
                self.assertEqual(len(split_metadata["assignment_sha256"]), 64)
            ratios = metadata["counts"]["actual_cell_ratios"]
            self.assertAlmostEqual(sum(ratios.values()), 1.0)
            self.assertEqual(metadata["counts"]["dropped_cell_ratio"], 0.0)
            self.assertEqual(metadata["counts"]["dropped_sample_ratio"], 0.0)

    def test_safe_loader_requires_metadata_and_unsafe_path_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=2, height=4, width=4)
            )
            config = CellFixedSplitConfig(
                split_seed=42,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            split_path = root / "split.csv"
            metadata_path = root / "metadata.json"
            created = create_persistent_cell_fixed_split(
                dataset,
                split_path,
                metadata_path,
                config=config,
            )

            with self.assertRaisesRegex(ValueError, "metadata_json is required"):
                load_persistent_split(dataset, split_path, config=config)
            with self.assertRaisesRegex(ValueError, "config is required"):
                load_persistent_split(
                    dataset,
                    split_path,
                    metadata_json=metadata_path,
                )

            unsafe = load_persistent_split(
                dataset,
                split_path,
                unsafe_allow_missing_metadata=True,
            )
            self.assertEqual(unsafe.split_sha256, created.split_sha256)

    def test_safe_loader_rejects_changed_dataset_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                self._write_variant_dataset(root / "base", data_offset=0.0)
            )
            changed_dataset = CellDataset(
                self._write_variant_dataset(root / "changed", data_offset=1.0)
            )
            config = CellFixedSplitConfig(
                split_seed=42,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            split_path = root / "split.csv"
            metadata_path = root / "metadata.json"
            create_persistent_cell_fixed_split(
                dataset,
                split_path,
                metadata_path,
                config=config,
            )

            with self.assertRaisesRegex(ValueError, "data_sha256"):
                load_persistent_split(
                    changed_dataset,
                    split_path,
                    metadata_json=metadata_path,
                    config=config,
                )

    def test_safe_loader_rejects_changed_feature_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                self._write_variant_dataset(root / "base", data_version="v2-a")
            )
            changed_dataset = CellDataset(
                self._write_variant_dataset(root / "changed", data_version="v2-b")
            )
            config = CellFixedSplitConfig(
                split_seed=42,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            split_path = root / "split.csv"
            metadata_path = root / "metadata.json"
            create_persistent_cell_fixed_split(
                dataset,
                split_path,
                metadata_path,
                config=config,
            )

            with self.assertRaisesRegex(ValueError, "feature_metadata"):
                load_persistent_split(
                    changed_dataset,
                    split_path,
                    metadata_json=metadata_path,
                    config=config,
                )

    def test_safe_loader_validates_every_metadata_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=2, height=4, width=4)
            )
            config = CellFixedSplitConfig(
                split_seed=42,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
            split_path = root / "split.csv"
            metadata_path = root / "metadata.json"
            create_persistent_cell_fixed_split(
                dataset,
                split_path,
                metadata_path,
                config=config,
            )
            original = json.loads(metadata_path.read_text(encoding="utf-8"))
            mutations = {
                "schema_version": "invalid.schema",
                "split_mode": "buffered_block",
                "split_seed": 3407,
                "config": {**original["config"], "split_seed": 3407},
                "config_sha256": "0" * 64,
                "feature_metadata_sha256": "0" * 64,
                "data_manifest": [],
                "counts": {},
                "split_sha256": "0" * 64,
            }

            for field, value in mutations.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(original)
                    changed[field] = value
                    changed_path = root / f"changed-{field}.json"
                    changed_path.write_text(
                        json.dumps(changed),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, f"metadata {field}"):
                        load_persistent_split(
                            dataset,
                            split_path,
                            metadata_json=changed_path,
                            config=config,
                        )

    def test_train_only_stats_ignore_validation_and_test_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dataset = CellDataset(
                write_synthetic_months(root / "base", n_months=2, height=4, width=4)
            )
            manifest = create_persistent_cell_fixed_split(
                base_dataset,
                root / "split.csv",
                root / "split.json",
                config=CellFixedSplitConfig(
                    split_seed=42,
                    validation_ratio=0.25,
                    test_ratio=0.25,
                ),
            )
            train_ids = manifest.cell_ids("train")
            heldout_ids = manifest.cell_ids("validation") | manifest.cell_ids("test")

            low_paths = self._write_sentinel_dataset(
                root / "low",
                base_dataset,
                heldout_ids,
                sentinel=1000.0,
            )
            high_paths = self._write_sentinel_dataset(
                root / "high",
                base_dataset,
                heldout_ids,
                sentinel=1.0e12,
            )
            low_dataset = CellDataset(low_paths)
            high_dataset = CellDataset(high_paths)

            low_stats = fit_train_preprocessing(
                low_dataset,
                train_ids,
                split_sha256=manifest.split_sha256,
            )
            high_stats = fit_train_preprocessing(
                high_dataset,
                train_ids,
                split_sha256=manifest.split_sha256,
            )

            low_values = low_stats.to_dict()
            high_values = high_stats.to_dict()
            for name in set(low_values) - {"data_sha256"}:
                self.assertEqual(low_values[name], high_values[name])
            self.assertNotEqual(low_values["data_sha256"], high_values["data_sha256"])
            self.assertEqual(high_stats.schema_version, PREPROCESSING_SCHEMA_VERSION)
            self.assertEqual(high_stats.data_sha256, high_dataset.data_sha256)
            self.assertEqual(high_stats.split_sha256, manifest.split_sha256)
            self.assertEqual(high_stats.split_name, "train")
            high_dataset.set_preprocessing(
                high_stats,
                split_sha256=manifest.split_sha256,
            )
            missing_cell = next(iter(heldout_ids))
            sample_index = high_dataset.sample_indices_for_cells({missing_cell})[0]
            transformed = high_dataset[sample_index]
            self.assertTrue(np.all(np.isfinite(transformed["stream_a"])))
            self.assertAlmostEqual(float(transformed["stream_a"][0]), 0.0, places=6)

    def test_preprocessing_stats_reject_a_different_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=2, height=4, width=4)
            )
            first = create_persistent_cell_fixed_split(
                dataset,
                root / "first.csv",
                root / "first.json",
                config=CellFixedSplitConfig(
                    split_seed=42,
                    validation_ratio=0.25,
                    test_ratio=0.25,
                ),
            )
            second = create_persistent_cell_fixed_split(
                dataset,
                root / "second.csv",
                root / "second.json",
                config=CellFixedSplitConfig(
                    split_seed=7,
                    validation_ratio=0.25,
                    test_ratio=0.25,
                ),
            )
            stats = fit_train_preprocessing(
                dataset,
                first.cell_ids("train"),
                split_sha256=first.split_sha256,
            )
            self.assertNotEqual(first.split_sha256, second.split_sha256)

            with self.assertRaisesRegex(ValueError, "split SHA-256"):
                dataset.set_preprocessing(
                    stats,
                    split_sha256=second.split_sha256,
                )
            with self.assertRaisesRegex(ValueError, "split name"):
                dataset.set_preprocessing(
                    stats,
                    split_sha256=first.split_sha256,
                    split_name="validation",
                )

    def test_preprocessing_stats_reject_a_different_data_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                self._write_variant_dataset(root / "base", data_version="v2-a")
            )
            changed_dataset = CellDataset(
                self._write_variant_dataset(root / "changed", data_version="v2-b")
            )
            manifest = create_persistent_cell_fixed_split(
                dataset,
                root / "split.csv",
                root / "split.json",
                config=CellFixedSplitConfig(
                    split_seed=42,
                    validation_ratio=0.25,
                    test_ratio=0.25,
                ),
            )
            stats = fit_train_preprocessing(
                dataset,
                manifest.cell_ids("train"),
                split_sha256=manifest.split_sha256,
            )

            with self.assertRaisesRegex(ValueError, "data SHA-256"):
                changed_dataset.set_preprocessing(
                    stats,
                    split_sha256=manifest.split_sha256,
                )

    def test_buffered_block_split_has_separate_feasibility_and_persistence_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root / "data", n_months=1, height=12, width=12)
            )
            config = BufferedBlockSplitConfig(
                split_seed=42,
                validation_ratio=0.1,
                test_ratio=0.1,
                block_size=4,
                buffer_cells=1,
            )
            feasibility = assess_buffered_block_split(dataset, config=config)

            self.assertTrue(feasibility["feasible"])
            self.assertGreater(
                feasibility["minimum_train_to_heldout_chebyshev_distance"],
                config.buffer_cells,
            )
            pairwise = feasibility["minimum_chebyshev_distance_by_pair"]
            self.assertGreater(pairwise["train_validation"], config.buffer_cells)
            self.assertGreater(pairwise["train_test"], config.buffer_cells)
            self.assertGreaterEqual(pairwise["validation_test"], 1)
            block_path = root / "block" / "buffered_block_seed42.csv"
            metadata_path = root / "block" / "buffered_block_seed42.metadata.json"
            created = create_persistent_buffered_block_split(
                dataset,
                block_path,
                metadata_path,
                config=config,
            )
            reloaded = load_persistent_split(
                dataset,
                block_path,
                metadata_json=metadata_path,
                config=config,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(created.split_mode, "buffered_block")
            self.assertEqual(reloaded.split_mode, "buffered_block")
            self.assertEqual(created.split_sha256, reloaded.split_sha256)
            self.assertEqual(
                metadata["buffer_feasibility"][
                    "minimum_chebyshev_distance_by_pair"
                ],
                pairwise,
            )
            self.assertAlmostEqual(
                metadata["counts"]["dropped_cell_ratio"],
                metadata["counts"]["by_split"]["dropped"]["cell_count"]
                / metadata["counts"]["total_cells"],
            )
            self.assertEqual(
                metadata["research_status"]["protocol_status"],
                "development_only_not_research_frozen",
            )

    def test_buffered_block_feasibility_reports_too_few_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = CellDataset(
                write_synthetic_months(root, n_months=1, height=2, width=2)
            )
            feasibility = assess_buffered_block_split(
                dataset,
                config=BufferedBlockSplitConfig(
                    split_seed=42,
                    validation_ratio=0.2,
                    test_ratio=0.2,
                    block_size=4,
                    buffer_cells=1,
                ),
            )

            self.assertFalse(feasibility["feasible"])
            self.assertIn("at least three", feasibility["reason"])

    @staticmethod
    def _write_variant_dataset(
        directory: Path,
        *,
        data_offset: float = 0.0,
        data_version: str = "v2_corrected_synthetic",
    ) -> list[Path]:
        paths = []
        for month_index in range(2):
            stream_a, stream_b, target, mask = synthetic_arrays(
                month_index,
                height=4,
                width=4,
            )
            stream_a[0, 0, 0] += data_offset
            paths.append(
                write_synthetic_npz(
                    directory / f"month_{month_index:02d}.npz",
                    month_id=f"2026-{month_index + 1:02d}",
                    stream_a=stream_a,
                    stream_b=stream_b,
                    target=target,
                    mask=mask,
                    data_version=data_version,
                )
            )
        return paths

    @staticmethod
    def _write_sentinel_dataset(
        directory: Path,
        base_dataset: CellDataset,
        heldout_ids: frozenset[str],
        *,
        sentinel: float,
    ) -> list[Path]:
        paths = []
        for month_index in range(base_dataset.n_months):
            stream_a, stream_b, target, mask = synthetic_arrays(
                month_index,
                height=4,
                width=4,
            )
            for cell_id in heldout_ids:
                location = base_dataset.cell_location(cell_id)
                stream_a[:, location.row, location.col] = sentinel
                stream_b[:, location.row, location.col] = sentinel
                stream_a[0, location.row, location.col] = np.nan
            paths.append(
                write_synthetic_npz(
                    directory / f"month_{month_index:02d}.npz",
                    month_id=f"2026-{month_index + 1:02d}",
                    stream_a=stream_a,
                    stream_b=stream_b,
                    target=target,
                    mask=mask,
                )
            )
        return paths


if __name__ == "__main__":
    unittest.main()
