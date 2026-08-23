from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cell_msca.evaluate import (
    MetricMismatchError,
    calculate_prediction_bootstrap_from_csv,
    read_prediction_csv,
    verify_prediction_file_metrics,
    write_metrics_json,
    write_prediction_csv,
)
from cell_msca.metrics import bootstrap_metrics_by_space, metrics_by_space
from cell_msca.target import inverse_target, target_transform


class PredictionRoundTripTests(unittest.TestCase):
    def test_saved_predictions_reproduce_json_metrics_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            true_original = np.array([0.0, 1.25, 10.0, 100.0], dtype=np.float64)
            true_log = target_transform(true_original)
            pred_log = true_log + np.array([0.0, 0.02, -0.03, 0.01])
            pred_original = inverse_target(pred_log, mode="median")
            expected = metrics_by_space(
                y_true_original=true_original,
                y_pred_original=pred_original,
                y_true_log=true_log,
                y_pred_log=pred_log,
            )
            prediction_path = temp_path / "predictions.csv"
            metric_path = temp_path / "metrics.json"

            write_prediction_csv(
                prediction_path,
                y_true_original=true_original,
                y_pred_original=pred_original,
                y_true_log=true_log,
                y_pred_log=pred_log,
                cell_ids=["cell-a", "cell-a", "cell-b", "cell-b"],
            )
            write_metrics_json(metric_path, expected)

            recalculated = verify_prediction_file_metrics(prediction_path, metric_path)

            self.assertEqual(recalculated, expected)

    def test_prediction_metric_verifier_reports_changed_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp_path = Path(directory)
            prediction_path = temp_path / "predictions.csv"
            metric_path = temp_path / "metrics.json"
            values = np.array([0.0, 1.0, 2.0])
            logs = target_transform(values)
            expected = metrics_by_space(
                y_true_original=values,
                y_pred_original=values,
                y_true_log=logs,
                y_pred_log=logs,
            )
            expected["original_unit"]["mae"] = 1.0

            write_prediction_csv(
                prediction_path,
                y_true_original=values,
                y_pred_original=values,
                y_true_log=logs,
                y_pred_log=logs,
            )
            metric_path.write_text(json.dumps(expected), encoding="utf-8")

            with self.assertRaisesRegex(MetricMismatchError, "original_unit.mae"):
                verify_prediction_file_metrics(prediction_path, metric_path)

    def test_artifact_writers_refuse_to_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "predictions.csv"
            values = np.array([0.0, 1.0])
            logs = target_transform(values)
            write_prediction_csv(
                prediction_path,
                y_true_original=values,
                y_pred_original=values,
                y_true_log=logs,
                y_pred_log=logs,
            )

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_prediction_csv(
                    prediction_path,
                    y_true_original=values,
                    y_pred_original=values,
                    y_true_log=logs,
                    y_pred_log=logs,
                )

    def test_prediction_reader_preserves_cell_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "predictions.csv"
            values = np.array([1.0, 2.0, 3.0])
            logs = target_transform(values)
            expected_cell_ids = ["001", "cell-beta", "cell,with,comma"]
            write_prediction_csv(
                prediction_path,
                y_true_original=values,
                y_pred_original=values,
                y_true_log=logs,
                y_pred_log=logs,
                cell_ids=expected_cell_ids,
            )

            loaded = read_prediction_csv(prediction_path)

            self.assertEqual(loaded["cell_id"].tolist(), expected_cell_ids)

    def test_saved_csv_reproduces_cell_cluster_bootstrap_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "predictions.csv"
            true_original = np.array([1.0, 2.0, 10.0, 12.0, 20.0, 21.0])
            pred_original = np.array([1.5, 1.5, 11.0, 11.0, 19.0, 22.0])
            true_log = np.array([0.1, 0.2, 1.0, 1.2, 2.0, 2.1])
            pred_log = np.array([0.15, 0.15, 1.1, 1.1, 1.9, 2.2])
            cell_ids = np.array(["cell-a", "cell-a", "cell-b", "cell-b", "cell-c", "cell-c"])
            write_prediction_csv(
                prediction_path,
                y_true_original=true_original,
                y_pred_original=pred_original,
                y_true_log=true_log,
                y_pred_log=pred_log,
                cell_ids=cell_ids,
            )
            expected = bootstrap_metrics_by_space(
                y_true_original=true_original,
                y_pred_original=pred_original,
                y_true_log=true_log,
                y_pred_log=pred_log,
                cell_ids=cell_ids,
                n_boot=50,
                seed=42,
            )

            recalculated = calculate_prediction_bootstrap_from_csv(
                prediction_path,
                n_boot=50,
                seed=42,
            )

            self.assertEqual(recalculated, expected)


if __name__ == "__main__":
    unittest.main()
