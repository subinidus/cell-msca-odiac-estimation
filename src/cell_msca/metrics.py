"""Paper-facing regression metrics and cell-cluster uncertainty.

Original-unit and log-space arrays use separate public arguments, functions,
and output namespaces. This makes unit mixing visible at the API boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

METRIC_NAMES = ("mae", "rmse", "r2", "bias")


def _paired_finite_arrays(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    true_name: str,
    pred_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true_array = np.asarray(y_true, dtype=np.float64).ravel()
    pred_array = np.asarray(y_pred, dtype=np.float64).ravel()
    if true_array.shape != pred_array.shape:
        raise ValueError(
            f"shape mismatch: {true_name}={true_array.shape}, {pred_name}={pred_array.shape}"
        )
    if true_array.size == 0:
        raise ValueError("metric arrays must not be empty")
    if not np.all(np.isfinite(true_array)):
        raise ValueError(f"{true_name} must contain only finite values")
    if not np.all(np.isfinite(pred_array)):
        raise ValueError(f"{pred_name} must contain only finite values")
    return true_array, pred_array


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int | float]:
    """Compute the Phase 1 metric contract for one explicitly chosen space."""

    y_true, y_pred = _paired_finite_arrays(
        y_true,
        y_pred,
        true_name="y_true",
        pred_name="y_pred",
    )
    errors = y_pred - y_true
    squared_errors = errors**2
    ss_res = float(np.sum(squared_errors))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    # R2 is undefined for a non-perfect constant target. Returning 0.0 follows
    # sklearn's finite-output convention; an exactly perfect prediction is 1.0.
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "r2": float(r2),
        "bias": float(np.mean(errors)),
    }


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return one-based average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the secondary Spearman metric without adding it to headlines.

    If either input is constant, rank correlation is undefined. This evaluator
    returns the explicit finite convention ``0.0`` so result JSON never stores
    NaN while headline regression metrics remain unaffected.
    """

    true_array, pred_array = _paired_finite_arrays(
        y_true,
        y_pred,
        true_name="y_true_spearman",
        pred_name="y_pred_spearman",
    )
    true_ranks = _average_ranks(true_array)
    pred_ranks = _average_ranks(pred_array)
    true_centered = true_ranks - np.mean(true_ranks)
    pred_centered = pred_ranks - np.mean(pred_ranks)
    denominator = float(
        np.sqrt(np.sum(true_centered**2) * np.sum(pred_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(true_centered * pred_centered) / denominator)


def metrics_by_space(
    *,
    y_true_original: ArrayLike,
    y_pred_original: ArrayLike,
    y_true_log: ArrayLike,
    y_pred_log: ArrayLike,
) -> dict[str, dict[str, int | float]]:
    """Return independently named original-unit and log-space metrics."""

    original_true, original_pred = _paired_finite_arrays(
        y_true_original,
        y_pred_original,
        true_name="y_true_original",
        pred_name="y_pred_original",
    )
    log_true, log_pred = _paired_finite_arrays(
        y_true_log,
        y_pred_log,
        true_name="y_true_log",
        pred_name="y_pred_log",
    )
    if original_true.size != log_true.size:
        raise ValueError(
            "sample count mismatch between original_unit and log_space: "
            f"original_unit={original_true.size}, log_space={log_true.size}"
        )

    return {
        "original_unit": regression_metrics(original_true, original_pred),
        "log_space": regression_metrics(log_true, log_pred),
    }


def _mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(regression_metrics(y_true, y_pred)["r2"])


def _bias(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.mean(y_pred - y_true))


_STATISTICS: dict[str, Callable[[NDArray[np.float64], NDArray[np.float64]], float]] = {
    "mae": _mae,
    "rmse": _rmse,
    "r2": _r2,
    "bias": _bias,
}


def _draw_complete_cell_indices(
    unique_cells: NDArray[Any],
    sample_indices: Mapping[Any, NDArray[np.intp]],
    rng: np.random.Generator,
) -> tuple[NDArray[Any], NDArray[np.intp]]:
    """Draw cells with replacement and concatenate every row of each draw."""

    drawn_cells = rng.choice(unique_cells, size=unique_cells.size, replace=True)
    drawn_indices = np.concatenate([sample_indices[cell] for cell in drawn_cells])
    return drawn_cells, drawn_indices


def _bootstrap_metrics_by_cell(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cell_ids: ArrayLike,
    *,
    true_name: str,
    pred_name: str,
    n_boot: int,
    seed: int,
    alpha: float,
) -> dict[str, dict[str, float]]:
    y_true, y_pred = _paired_finite_arrays(
        y_true,
        y_pred,
        true_name=true_name,
        pred_name=pred_name,
    )
    cell_ids = np.asarray(cell_ids).ravel()
    if cell_ids.shape != y_true.shape:
        raise ValueError(f"cell_ids shape mismatch: {cell_ids.shape} vs {y_true.shape}")
    if not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1; got {alpha!r}")

    unique_cells = np.unique(cell_ids)
    if unique_cells.size == 0:
        raise ValueError("cell_ids must not be empty")
    sample_indices = {cell: np.flatnonzero(cell_ids == cell) for cell in unique_cells}
    rng = np.random.default_rng(seed)
    bootstrap_values = {
        name: np.empty(n_boot, dtype=np.float64) for name in METRIC_NAMES
    }

    for iteration in range(n_boot):
        _, drawn_indices = _draw_complete_cell_indices(unique_cells, sample_indices, rng)
        drawn_true = y_true[drawn_indices]
        drawn_pred = y_pred[drawn_indices]
        for name, statistic in _STATISTICS.items():
            bootstrap_values[name][iteration] = statistic(drawn_true, drawn_pred)

    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    result: dict[str, dict[str, float]] = {}
    for name, statistic in _STATISTICS.items():
        lower, upper = np.percentile(
            bootstrap_values[name],
            [lower_percentile, upper_percentile],
        )
        result[name] = {
            "estimate": statistic(y_true, y_pred),
            "lower": float(lower),
            "upper": float(upper),
        }
    return result


def bootstrap_original_unit_metrics(
    *,
    y_true_original: ArrayLike,
    y_pred_original: ArrayLike,
    cell_ids: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Cell-cluster bootstrap that accepts original-unit arrays only."""

    return _bootstrap_metrics_by_cell(
        y_true_original,
        y_pred_original,
        cell_ids,
        true_name="y_true_original",
        pred_name="y_pred_original",
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )


def bootstrap_log_space_metrics(
    *,
    y_true_log: ArrayLike,
    y_pred_log: ArrayLike,
    cell_ids: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Cell-cluster bootstrap that accepts log-space arrays only."""

    return _bootstrap_metrics_by_cell(
        y_true_log,
        y_pred_log,
        cell_ids,
        true_name="y_true_log",
        pred_name="y_pred_log",
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )


def bootstrap_metrics_by_space(
    *,
    y_true_original: ArrayLike,
    y_pred_original: ArrayLike,
    y_true_log: ArrayLike,
    y_pred_log: ArrayLike,
    cell_ids: ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap both spaces without sharing or transforming their arrays."""

    original_result = bootstrap_original_unit_metrics(
        y_true_original=y_true_original,
        y_pred_original=y_pred_original,
        cell_ids=cell_ids,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    log_result = bootstrap_log_space_metrics(
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
        cell_ids=cell_ids,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    return {"original_unit": original_result, "log_space": log_result}
