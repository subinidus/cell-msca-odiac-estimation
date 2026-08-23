from __future__ import annotations

import unittest

import numpy as np

from cell_msca.target import duan_smearing_factor, inverse_target, target_transform


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


if __name__ == "__main__":
    unittest.main()
