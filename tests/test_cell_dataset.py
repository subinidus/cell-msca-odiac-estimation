from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from cell_msca.data import (
    MODEL_SAMPLE_KEYS,
    STREAM_A_FEATURES,
    CellDataset,
)
from tests.synthetic_npz import (
    synthetic_arrays,
    write_synthetic_months,
    write_synthetic_npz,
)


class CellDatasetTests(unittest.TestCase):
    def test_single_cell_sample_contract_has_no_spatial_or_hotspot_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_synthetic_months(Path(directory), n_months=2, height=2, width=3)
            dataset = CellDataset(paths)

            sample = dataset[0]
            audit_metadata = dataset.sample_metadata(0)

            self.assertEqual(set(sample), MODEL_SAMPLE_KEYS)
            self.assertEqual(sample["stream_a"].shape, (3,))
            self.assertEqual(sample["stream_b"].shape, (4,))
            self.assertEqual(len(dataset), 2 * 2 * 3)
            forbidden = {
                "row",
                "col",
                "coordinates",
                "neighbours",
                "mask",
                "label_cls",
                "hotspot_label",
            }
            self.assertTrue(forbidden.isdisjoint(sample))
            self.assertEqual(audit_metadata["row"], 0)
            self.assertEqual(audit_metadata["col"], 0)
            self.assertEqual(audit_metadata["data_version"], "v2_corrected_synthetic")

    def test_target_original_and_log_are_both_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_synthetic_months(Path(directory), n_months=1, height=2, width=2)
            dataset = CellDataset(paths, target_scale=2.0)

            sample = dataset[0]

            self.assertEqual(sample["target_original"], 1.0)
            self.assertAlmostEqual(sample["target_log"], np.log1p(0.5))
            self.assertEqual(sample["cell_id"], "r0_c0")
            self.assertEqual(sample["month_id"], "2026-01")

    def test_feature_order_mismatch_fails_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_order.npz"
            arrays = synthetic_arrays(0, height=2, width=2)
            wrong_order = (STREAM_A_FEATURES[1], STREAM_A_FEATURES[0], STREAM_A_FEATURES[2])
            write_synthetic_npz(
                path,
                month_id="2026-01",
                stream_a=arrays[0],
                stream_b=arrays[1],
                target=arrays[2],
                mask=arrays[3],
                stream_a_features=wrong_order,
            )

            with self.assertRaisesRegex(ValueError, "feature order mismatch"):
                CellDataset([path])

    def test_month_masks_must_match_for_complete_cell_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = synthetic_arrays(0, height=2, width=2)
            second_mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)
            second = synthetic_arrays(1, height=2, width=2, mask=second_mask)
            first_path = write_synthetic_npz(
                root / "first.npz",
                month_id="2026-01",
                stream_a=first[0],
                stream_b=first[1],
                target=first[2],
                mask=first[3],
            )
            second_path = write_synthetic_npz(
                root / "second.npz",
                month_id="2026-02",
                stream_a=second[0],
                stream_b=second[1],
                target=second[2],
                mask=second[3],
            )

            with self.assertRaisesRegex(ValueError, "mask differs"):
                CellDataset([first_path, second_path])

    def test_legacy_feature_keys_require_explicit_version_and_target_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            arrays = synthetic_arrays(0, height=2, width=2)
            write_synthetic_npz(
                path,
                month_id="2026-01",
                stream_a=arrays[0],
                stream_b=arrays[1],
                target=arrays[2],
                mask=arrays[3],
                legacy_metadata_keys=True,
            )

            with self.assertRaisesRegex(ValueError, "explicit legacy_data_version"):
                CellDataset([path])

            dataset = CellDataset(
                [path],
                legacy_data_version="v1_legacy",
                legacy_target_unit=(
                    "ODIAC native value; exact target-cell interpretation unverified"
                ),
            )
            self.assertEqual(dataset.feature_metadata.data_version, "v1_legacy")
            self.assertEqual(dataset.feature_metadata.stream_a, STREAM_A_FEATURES)


if __name__ == "__main__":
    unittest.main()
