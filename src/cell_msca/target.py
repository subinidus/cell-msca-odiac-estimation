"""Shared target transform and inverse-transform implementations.

All evaluation and inference paths should use :func:`inverse_target` instead
of spelling out an inverse formula at a call site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

InverseMode = Literal["median", "duan_smearing"]
NONNEGATIVE_PREDICTION_SUPPORT_POLICY = "nonnegative_max_zero_v1"


@dataclass(frozen=True)
class PredictionSupportDiagnostics:
    """Auditable diagnostics for the frozen nonnegative output projection."""

    prediction_support_policy: str
    sample_count: int
    pre_projection_negative_count: int
    pre_projection_negative_fraction: float
    pre_projection_minimum: float
    projection_applied_count: int

    def __post_init__(self) -> None:
        if self.prediction_support_policy != NONNEGATIVE_PREDICTION_SUPPORT_POLICY:
            raise ValueError("unsupported prediction support policy")
        if self.sample_count <= 0:
            raise ValueError("prediction support diagnostics require samples")
        if not 0 <= self.pre_projection_negative_count <= self.sample_count:
            raise ValueError("invalid pre-projection negative count")
        expected_fraction = self.pre_projection_negative_count / self.sample_count
        if not np.isclose(
            self.pre_projection_negative_fraction,
            expected_fraction,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("pre-projection negative fraction/count mismatch")
        if not np.isfinite(self.pre_projection_minimum):
            raise ValueError("pre-projection minimum must be finite")
        if (
            self.pre_projection_negative_count == 0
            and self.pre_projection_minimum < 0.0
        ) or (
            self.pre_projection_negative_count > 0
            and self.pre_projection_minimum >= 0.0
        ):
            raise ValueError("pre-projection minimum/negative count mismatch")
        if self.projection_applied_count != self.pre_projection_negative_count:
            raise ValueError("projection count must equal the negative count")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def project_nonnegative_predictions(
    unprojected_prediction: ArrayLike,
) -> tuple[NDArray[np.float64], PredictionSupportDiagnostics]:
    """Project finite predictions onto ``[0, +inf)`` and report every change.

    This is an explicit paper-facing output-support policy, not silent clipping.
    The caller must retain the returned diagnostics with the result artifact.
    """

    values = _finite_array(
        unprojected_prediction,
        name="unprojected_prediction",
    )
    if values.size == 0:
        raise ValueError("unprojected_prediction must not be empty")
    negative_count = int(np.count_nonzero(values < 0.0))
    projected = np.maximum(values, 0.0)
    diagnostics = PredictionSupportDiagnostics(
        prediction_support_policy=NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
        sample_count=int(values.size),
        pre_projection_negative_count=negative_count,
        pre_projection_negative_fraction=float(negative_count / values.size),
        pre_projection_minimum=float(np.min(values)),
        projection_applied_count=negative_count,
    )
    return projected, diagnostics


def _positive_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive; got {scale!r}")
    return scale


def _finite_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def target_transform(y_original: ArrayLike, *, scale: float = 1.0) -> NDArray[np.float64]:
    """Return ``log(1 + y / scale)`` after validating the physical target.

    Negative raw targets are rejected. They are never silently clipped before
    the logarithm.
    """

    scale = _positive_scale(scale)
    y_original = _finite_array(y_original, name="y_original")
    if np.any(y_original < 0.0):
        minimum = float(np.min(y_original))
        raise ValueError(f"y_original must be non-negative; minimum is {minimum}")
    return np.log1p(y_original / scale)


def duan_smearing_factor(pred_log: ArrayLike, true_log: ArrayLike) -> float:
    """Estimate Duan's factor from train residuals only.

    The caller is responsible for supplying training predictions and targets.
    Residuals follow the contract ``true_log - pred_log``.
    """

    pred_log = _finite_array(pred_log, name="pred_log").ravel()
    true_log = _finite_array(true_log, name="true_log").ravel()
    if pred_log.shape != true_log.shape:
        raise ValueError(f"shape mismatch: pred_log={pred_log.shape}, true_log={true_log.shape}")
    if pred_log.size == 0:
        raise ValueError("cannot estimate smearing from empty arrays")

    factor = float(np.mean(np.exp(true_log - pred_log)))
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"computed smearing factor must be finite and positive; got {factor!r}")
    return factor


def inverse_target(
    pred_log: ArrayLike,
    *,
    scale: float = 1.0,
    mode: InverseMode = "median",
    smearing_factor: float | None = None,
) -> NDArray[np.float64]:
    """Invert log targets through the shared median or Duan path.

    ``median`` computes ``scale * (exp(pred_log) - 1)``.
    ``duan_smearing`` computes ``scale * (exp(pred_log) * S - 1)``.

    This function performs only the mathematical inverse. Paper-facing callers
    apply and record the separate nonnegative prediction-support policy.
    """

    scale = _positive_scale(scale)
    pred_log = _finite_array(pred_log, name="pred_log")

    if mode == "median":
        if smearing_factor is not None:
            raise ValueError("smearing_factor is only valid when mode='duan_smearing'")
        factor = 1.0
    elif mode == "duan_smearing":
        if smearing_factor is None:
            raise ValueError("smearing_factor is required when mode='duan_smearing'")
        factor = float(smearing_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(
                f"smearing_factor must be finite and positive; got {smearing_factor!r}"
            )
    else:
        raise ValueError(f"unknown inverse mode: {mode!r}")

    with np.errstate(over="ignore", invalid="ignore"):
        inverted = scale * (np.exp(pred_log) * factor - 1.0)
    if not np.all(np.isfinite(inverted)):
        non_finite_count = int(np.size(inverted) - np.count_nonzero(np.isfinite(inverted)))
        raise ValueError(
            "inverse_target produced non-finite values "
            f"(mode={mode!r}, count={non_finite_count}); "
            "check pred_log magnitude, scale, and smearing_factor"
        )
    return inverted
