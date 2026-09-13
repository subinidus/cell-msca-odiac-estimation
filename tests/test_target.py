from __future__ import annotations

import unittest

import numpy as np

from cell_msca.target import (
    NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
    duan_smearing_factor,
    inverse_target,
    project_nonnegative_predictions,
    target_transform,
)


class TargetTransformTests(unittest.TestCase):
    def test_target_transform_and_median_inverse_round_trip(self) -> None:
        original = np.array([0.0, 2.0, 6.0, 20.0])
        transformed = target_transform(original, scale=2.0)

        recovered = inverse_target(transformed, scale=2.0, mode="median")

        np.testing.assert_allclose(recovered, original, rtol=1e-14, atol=1e-14)

    def test_target_transform_rejects_negative_raw_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            target_transform([0.0, -0.01, 2.0])

    def test_duan_smearing_factor_uses_true_minus_prediction_residual(self) -> None:
        pred_log = np.array([0.0, np.log(3.0)])
        true_log = pred_log + np.log(2.0)

        self.assertAlmostEqual(duan_smearing_factor(pred_log, true_log), 2.0)

    def test_smearing_inverse_regression_pred_zero_factor_two_returns_one(self) -> None:
        corrected = inverse_target(
            np.array([0.0]),
            mode="duan_smearing",
            smearing_factor=2.0,
        )

        self.assertAlmostEqual(corrected.item(), 1.0)

    def test_smearing_factor_is_required_only_for_smearing_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            inverse_target([0.0], mode="duan_smearing")
        with self.assertRaisesRegex(ValueError, "only valid"):
            inverse_target([0.0], mode="median", smearing_factor=1.2)

    def test_inverse_target_rejects_non_finite_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "produced non-finite values"):
            inverse_target([1000.0], mode="median")

    def test_nonnegative_projection_is_identity_without_negative_values(self) -> None:
        values = np.asarray([0.0, 1.5, 9.0], dtype=np.float64)
        projected, diagnostics = project_nonnegative_predictions(values)

        np.testing.assert_array_equal(projected, values)
        self.assertEqual(
            diagnostics.prediction_support_policy,
            NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
        )
        self.assertEqual(diagnostics.pre_projection_negative_count, 0)
        self.assertEqual(diagnostics.projection_applied_count, 0)
        self.assertEqual(diagnostics.pre_projection_minimum, 0.0)

    def test_nonnegative_projection_changes_only_negative_values(self) -> None:
        values = np.asarray([-2.0, -0.25, 0.0, 3.0], dtype=np.float64)
        original = values.copy()
        projected, diagnostics = project_nonnegative_predictions(values)

        np.testing.assert_array_equal(projected, [0.0, 0.0, 0.0, 3.0])
        np.testing.assert_array_equal(values, original)
        self.assertEqual(diagnostics.pre_projection_negative_count, 2)
        self.assertEqual(diagnostics.projection_applied_count, 2)
        self.assertEqual(diagnostics.pre_projection_negative_fraction, 0.5)
        self.assertEqual(diagnostics.pre_projection_minimum, -2.0)

    def test_nonnegative_projection_rejects_nan_and_infinity(self) -> None:
        for values in ([0.0, np.nan], [0.0, np.inf], [0.0, -np.inf]):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "finite"):
                    project_nonnegative_predictions(values)


if __name__ == "__main__":
    unittest.main()
