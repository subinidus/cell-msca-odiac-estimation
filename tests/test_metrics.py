from __future__ import annotations

from typing import Any
import unittest
from unittest import mock

import numpy as np

import cell_msca.metrics as metrics_module
from cell_msca.metrics import (
    _draw_complete_cell_indices,
    bootstrap_metrics_by_space,
    metrics_by_space,
)
from cell_msca.target import target_transform


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions_have_required_metrics_in_both_spaces(self) -> None:
        true_original = np.array([0.0, 1.0, 4.0, 10.0])
        true_log = target_transform(true_original)

        result = metrics_by_space(
            y_true_original=true_original,
            y_pred_original=true_original.copy(),
            y_true_log=true_log,
            y_pred_log=true_log.copy(),
        )

        for space in ("original_unit", "log_space"):
            self.assertAlmostEqual(result[space]["mae"], 0.0)
            self.assertAlmostEqual(result[space]["rmse"], 0.0)
            self.assertAlmostEqual(result[space]["r2"], 1.0)
            self.assertAlmostEqual(result[space]["bias"], 0.0)

    def test_paper_facing_metrics_contain_no_hotspot_fields(self) -> None:
        result = metrics_by_space(
            y_true_original=[0.0, 1.0, 2.0],
            y_pred_original=[0.1, 0.8, 2.2],
            y_true_log=[0.0, 0.5, 1.0],
            y_pred_log=[0.1, 0.4, 1.1],
        )

        keys = {key for namespace in result.values() for key in namespace}
        self.assertFalse(
            any("hotspot" in key.lower() or "top10" in key.lower() for key in keys)
        )

    def test_original_bootstrap_never_receives_log_arrays(self) -> None:
        true_original = np.array([10.0, 11.0, 20.0, 21.0])
        pred_original = np.array([9.0, 12.0, 19.0, 22.0])
        true_log = np.array([1.0, 1.1, 2.0, 2.1])
        pred_log = np.array([0.9, 1.2, 1.9, 2.2])
        cell_ids = np.array([100, 100, 200, 200])
        received: dict[str, Any] = {}

        def capture_original(**kwargs: Any) -> dict[str, dict[str, float]]:
            received.update(kwargs)
            return {"mae": {"estimate": 1.0, "lower": 1.0, "upper": 1.0}}

        def capture_log(**kwargs: Any) -> dict[str, dict[str, float]]:
            return {"mae": {"estimate": 0.1, "lower": 0.1, "upper": 0.1}}

        with (
            mock.patch.object(
                metrics_module, "bootstrap_original_unit_metrics", capture_original
            ),
            mock.patch.object(metrics_module, "bootstrap_log_space_metrics", capture_log),
        ):
            bootstrap_metrics_by_space(
                y_true_original=true_original,
                y_pred_original=pred_original,
                y_true_log=true_log,
                y_pred_log=pred_log,
                cell_ids=cell_ids,
                n_boot=10,
            )

        np.testing.assert_array_equal(received["y_true_original"], true_original)
        np.testing.assert_array_equal(received["y_pred_original"], pred_original)
        self.assertFalse(np.array_equal(received["y_true_original"], true_log))
        self.assertFalse(np.array_equal(received["y_pred_original"], pred_log))

    def test_cell_cluster_bootstrap_returns_separate_spaces(self) -> None:
        result = bootstrap_metrics_by_space(
            y_true_original=[1.0, 2.0, 10.0, 12.0, 20.0, 21.0],
            y_pred_original=[1.5, 1.5, 11.0, 11.0, 19.0, 22.0],
            y_true_log=[0.1, 0.2, 1.0, 1.2, 2.0, 2.1],
            y_pred_log=[0.15, 0.15, 1.1, 1.1, 1.9, 2.2],
            cell_ids=[1, 1, 2, 2, 3, 3],
            n_boot=50,
            seed=42,
        )

        self.assertEqual(set(result), {"original_unit", "log_space"})
        for space in result.values():
            self.assertEqual(set(space), {"mae", "rmse", "r2", "bias"})
            for interval in space.values():
                self.assertEqual(set(interval), {"estimate", "lower", "upper"})

    def test_metrics_by_space_rejects_different_space_sample_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            metrics_by_space(
                y_true_original=[1.0, 2.0, 3.0],
                y_pred_original=[1.0, 2.0, 3.0],
                y_true_log=[0.1, 0.2],
                y_pred_log=[0.1, 0.2],
            )

    def test_complete_cells_are_resampled_as_deterministic_whole_clusters(self) -> None:
        unique_cells = np.array(["cell-a", "cell-b", "cell-c"])
        sample_indices = {
            "cell-a": np.array([0, 1]),
            "cell-b": np.array([2, 3, 4]),
            "cell-c": np.array([5]),
        }

        first_cells, first_indices = _draw_complete_cell_indices(
            unique_cells,
            sample_indices,
            np.random.default_rng(0),
        )
        second_cells, second_indices = _draw_complete_cell_indices(
            unique_cells,
            sample_indices,
            np.random.default_rng(0),
        )

        np.testing.assert_array_equal(first_cells, second_cells)
        np.testing.assert_array_equal(first_indices, second_indices)
        self.assertLess(len(set(first_cells.tolist())), len(first_cells))

        offset = 0
        for cell_id in first_cells:
            complete_cell = sample_indices[str(cell_id)]
            actual_cell_rows = first_indices[offset : offset + complete_cell.size]
            np.testing.assert_array_equal(actual_cell_rows, complete_cell)
            offset += complete_cell.size
        self.assertEqual(offset, first_indices.size)


if __name__ == "__main__":
    unittest.main()
