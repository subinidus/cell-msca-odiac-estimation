"""Prediction artifact I/O and metric round-trip verification."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .metrics import (
    METRIC_NAMES,
    bootstrap_metrics_by_space,
    metrics_by_space,
    spearman_correlation,
)

PREDICTION_COLUMNS = (
    "cell_id",
    "true_original",
    "pred_original",
    "true_log",
    "pred_log",
)

BASELINE_RESULT_SCHEMA_VERSION = "cell_msca.baseline_result.v1"
BASELINE_RESULT_FIELDS = {
    "schema_version",
    "model_name",
    "data_version",
    "data_sha256",
    "split_sha256",
    "split_config_sha256",
    "config_sha256",
    "preprocessing_sha256",
    "split_seed",
    "train_seed",
    "evaluation_split",
    "target_transform",
    "target_scale",
    "loss_objective",
    "inverse_mode",
    "smearing_factor",
    "headline_metrics",
    "secondary_metrics",
    "best_iteration",
    "best_epoch",
}
_FORBIDDEN_RESULT_KEY_FRAGMENTS = (
    "hotspot",
    "top10",
    "top_10",
    "roc_auc",
    "pr_auc",
)


class MetricMismatchError(AssertionError):
    """Raised when saved metrics do not match the prediction artifact."""


@dataclass(frozen=True)
class BaselineResultContext:
    """Provenance and model choices attached to one baseline result row."""

    model_name: str
    data_version: str
    data_sha256: str
    split_sha256: str
    split_config_sha256: str
    config_sha256: str
    preprocessing_sha256: str
    split_seed: int
    train_seed: int
    evaluation_split: str
    target_transform: str
    target_scale: float
    loss_objective: str
    inverse_mode: str
    smearing_factor: float | None = None
    best_iteration: int | None = None
    best_epoch: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "data_sha256",
            "split_sha256",
            "split_config_sha256",
            "config_sha256",
            "preprocessing_sha256",
        ):
            _validate_sha256(getattr(self, name), name=name)
        for name in ("model_name", "data_version", "loss_objective"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must not be empty")
        if self.evaluation_split not in {"validation", "test"}:
            raise ValueError("evaluation_split must be 'validation' or 'test'")
        if self.target_transform not in {"identity", "log1p"}:
            raise ValueError("target_transform must be 'identity' or 'log1p'")
        if self.inverse_mode not in {"none", "median", "duan_smearing"}:
            raise ValueError(f"invalid inverse_mode: {self.inverse_mode!r}")
        if not np.isfinite(self.target_scale) or self.target_scale <= 0.0:
            raise ValueError("target_scale must be finite and positive")
        if self.inverse_mode == "duan_smearing":
            if self.smearing_factor is None:
                raise ValueError("Duan inverse requires a smearing_factor")
            if not np.isfinite(self.smearing_factor) or self.smearing_factor <= 0.0:
                raise ValueError("smearing_factor must be finite and positive")
        elif self.smearing_factor is not None:
            raise ValueError(
                "smearing_factor must be null unless inverse_mode='duan_smearing'"
            )
        if self.best_iteration is not None and self.best_iteration <= 0:
            raise ValueError("best_iteration must be positive when present")
        if self.best_epoch is not None and self.best_epoch <= 0:
            raise ValueError("best_epoch must be positive when present")


def _validate_sha256(value: str, *, name: str) -> None:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")


def _finite_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _new_output_path(path: str | Path, *, overwrite: bool) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def write_prediction_csv(
    path: str | Path,
    *,
    y_true_original: ArrayLike,
    y_pred_original: ArrayLike,
    y_true_log: ArrayLike,
    y_pred_log: ArrayLike,
    cell_ids: ArrayLike | None = None,
    overwrite: bool = False,
) -> Path:
    """Save round-trip-safe prediction values in the Phase 1 CSV schema."""

    arrays = {
        "true_original": _finite_1d(y_true_original, name="y_true_original"),
        "pred_original": _finite_1d(y_pred_original, name="y_pred_original"),
        "true_log": _finite_1d(y_true_log, name="y_true_log"),
        "pred_log": _finite_1d(y_pred_log, name="y_pred_log"),
    }
    lengths = {name: values.size for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction arrays have different lengths: {lengths}")
    n_rows = next(iter(lengths.values()))

    if cell_ids is None:
        cell_array = np.arange(n_rows)
    else:
        cell_array = np.asarray(cell_ids).ravel()
        if cell_array.size != n_rows:
            raise ValueError(f"cell_ids length is {cell_array.size}; expected {n_rows}")

    output_path = _new_output_path(path, overwrite=overwrite)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for index in range(n_rows):
            writer.writerow(
                {
                    "cell_id": str(cell_array[index]),
                    "true_original": repr(float(arrays["true_original"][index])),
                    "pred_original": repr(float(arrays["pred_original"][index])),
                    "true_log": repr(float(arrays["true_log"][index])),
                    "pred_log": repr(float(arrays["pred_log"][index])),
                }
            )
    return output_path


def write_metrics_json(
    path: str | Path,
    metrics: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write metric JSON without silently replacing an existing result."""

    output_path = _new_output_path(path, overwrite=overwrite)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output_path


def read_prediction_csv(path: str | Path) -> dict[str, NDArray[Any]]:
    """Read predictions while preserving stable cell identifiers as strings."""

    prediction_path = Path(path)
    cell_ids: list[str] = []
    values: dict[str, list[float]] = {
        "true_original": [],
        "pred_original": [],
        "true_log": [],
        "pred_log": [],
    }
    with prediction_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(PREDICTION_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"prediction CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            cell_id = row["cell_id"]
            if cell_id is None or cell_id == "":
                raise ValueError(f"empty cell_id at CSV row {row_number}")
            cell_ids.append(cell_id)
            for name in values:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid float in column {name!r} at CSV row {row_number}"
                    ) from error
                if not np.isfinite(value):
                    raise ValueError(
                        f"non-finite value in column {name!r} at CSV row {row_number}"
                    )
                values[name].append(value)
    if not values["true_original"]:
        raise ValueError("prediction CSV contains no prediction rows")
    return {
        "cell_id": np.asarray(cell_ids, dtype=str),
        **{name: np.asarray(column, dtype=np.float64) for name, column in values.items()},
    }


def calculate_prediction_metrics_from_csv(
    prediction_csv: str | Path,
) -> dict[str, dict[str, int | float]]:
    """Recalculate raw/log metrics solely from a saved prediction CSV."""

    values = read_prediction_csv(prediction_csv)
    return metrics_by_space(
        y_true_original=values["true_original"],
        y_pred_original=values["pred_original"],
        y_true_log=values["true_log"],
        y_pred_log=values["pred_log"],
    )


def calculate_prediction_bootstrap_from_csv(
    prediction_csv: str | Path,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Recalculate both cell-cluster bootstraps from one saved CSV artifact."""

    values = read_prediction_csv(prediction_csv)
    return bootstrap_metrics_by_space(
        y_true_original=values["true_original"],
        y_pred_original=values["pred_original"],
        y_true_log=values["true_log"],
        y_pred_log=values["pred_log"],
        cell_ids=values["cell_id"],
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )


def _load_expected_metrics(
    metric_json_or_mapping: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(metric_json_or_mapping, Mapping):
        return metric_json_or_mapping
    with Path(metric_json_or_mapping).open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError("metric JSON root must be an object")
    return loaded


def verify_prediction_file_metrics(
    prediction_csv: str | Path,
    metric_json_or_mapping: str | Path | Mapping[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, dict[str, int | float]]:
    """Recompute metrics and raise if the stored JSON does not match.

    Exact comparison is the default because :func:`write_prediction_csv`
    stores enough digits to recover each binary float exactly.
    """

    recalculated = calculate_prediction_metrics_from_csv(prediction_csv)
    expected = _load_expected_metrics(metric_json_or_mapping)
    differences: list[str] = []

    for space in ("original_unit", "log_space"):
        expected_space = expected.get(space)
        if not isinstance(expected_space, Mapping):
            differences.append(f"missing metric namespace {space!r}")
            continue
        for name in ("n", *METRIC_NAMES):
            if name not in expected_space:
                differences.append(f"missing metric {space}.{name}")
                continue
            actual_value = recalculated[space][name]
            expected_value = expected_space[name]
            if name == "n":
                matches = int(actual_value) == int(expected_value)
            else:
                matches = bool(
                    np.isclose(
                        float(actual_value),
                        float(expected_value),
                        rtol=rtol,
                        atol=atol,
                        equal_nan=False,
                    )
                )
            if not matches:
                differences.append(
                    f"{space}.{name}: stored={expected_value!r}, "
                    f"recalculated={actual_value!r}"
                )

    if differences:
        raise MetricMismatchError("prediction/metric mismatch:\n" + "\n".join(differences))
    return recalculated


def evaluate_baseline_predictions(
    context: BaselineResultContext,
    *,
    y_true_original: ArrayLike,
    y_pred_original: ArrayLike,
    y_true_log: ArrayLike,
    y_pred_log: ArrayLike,
) -> dict[str, Any]:
    """Build one paper-facing baseline row through the shared evaluator."""

    metrics = metrics_by_space(
        y_true_original=y_true_original,
        y_pred_original=y_pred_original,
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
    )
    result: dict[str, Any] = {
        "schema_version": BASELINE_RESULT_SCHEMA_VERSION,
        "model_name": context.model_name,
        "data_version": context.data_version,
        "data_sha256": context.data_sha256,
        "split_sha256": context.split_sha256,
        "split_config_sha256": context.split_config_sha256,
        "config_sha256": context.config_sha256,
        "preprocessing_sha256": context.preprocessing_sha256,
        "split_seed": context.split_seed,
        "train_seed": context.train_seed,
        "evaluation_split": context.evaluation_split,
        "target_transform": context.target_transform,
        "target_scale": float(context.target_scale),
        "loss_objective": context.loss_objective,
        "inverse_mode": context.inverse_mode,
        "smearing_factor": context.smearing_factor,
        "headline_metrics": {
            "original_unit": metrics["original_unit"],
        },
        "secondary_metrics": {
            "log_space": metrics["log_space"],
            "spearman": spearman_correlation(
                y_true_original,
                y_pred_original,
            ),
        },
        "best_iteration": context.best_iteration,
        "best_epoch": context.best_epoch,
    }
    validate_baseline_result_row(result)
    return result


def _walk_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_mapping_keys(nested))
    return keys


def validate_baseline_result_row(result: Mapping[str, Any]) -> None:
    """Validate the exact Phase 3 baseline result schema and exclusions."""

    actual_fields = set(result)
    if actual_fields != BASELINE_RESULT_FIELDS:
        missing = sorted(BASELINE_RESULT_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - BASELINE_RESULT_FIELDS)
        raise ValueError(
            "baseline result fields do not match the schema: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if result["schema_version"] != BASELINE_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported baseline result schema_version")
    BaselineResultContext(
        model_name=str(result["model_name"]),
        data_version=str(result["data_version"]),
        data_sha256=str(result["data_sha256"]),
        split_sha256=str(result["split_sha256"]),
        split_config_sha256=str(result["split_config_sha256"]),
        config_sha256=str(result["config_sha256"]),
        preprocessing_sha256=str(result["preprocessing_sha256"]),
        split_seed=int(result["split_seed"]),
        train_seed=int(result["train_seed"]),
        evaluation_split=str(result["evaluation_split"]),
        target_transform=str(result["target_transform"]),
        target_scale=float(result["target_scale"]),
        loss_objective=str(result["loss_objective"]),
        inverse_mode=str(result["inverse_mode"]),
        smearing_factor=(
            None
            if result["smearing_factor"] is None
            else float(result["smearing_factor"])
        ),
        best_iteration=(
            None
            if result["best_iteration"] is None
            else int(result["best_iteration"])
        ),
        best_epoch=(
            None if result["best_epoch"] is None else int(result["best_epoch"])
        ),
    )
    model_name = str(result["model_name"])
    if model_name in {"train_mean", "lightgbm_raw", "lightgbm_tweedie"}:
        if result["inverse_mode"] != "none" or result["target_transform"] != "identity":
            raise ValueError(
                f"{model_name} must record identity target and inverse_mode='none'"
            )
    if model_name in {"lightgbm_log1p", "concat_mlp"}:
        if result["target_transform"] != "log1p" or result["inverse_mode"] == "none":
            raise ValueError(
                f"{model_name} must record log1p target and a selected inverse mode"
            )
    forbidden = [
        key
        for key in _walk_mapping_keys(result)
        if any(
            fragment in key.lower()
            for fragment in _FORBIDDEN_RESULT_KEY_FRAGMENTS
        )
    ]
    if forbidden:
        raise ValueError(f"paper-facing baseline result has forbidden keys: {forbidden}")
    headline = result["headline_metrics"]
    secondary = result["secondary_metrics"]
    if not isinstance(headline, Mapping) or set(headline) != {"original_unit"}:
        raise ValueError("headline_metrics must contain original_unit only")
    if not isinstance(secondary, Mapping) or set(secondary) != {
        "log_space",
        "spearman",
    }:
        raise ValueError("secondary_metrics must contain log_space and spearman only")
    expected_metric_fields = {"n", *METRIC_NAMES}
    for namespace, values in (
        ("headline_metrics.original_unit", headline["original_unit"]),
        ("secondary_metrics.log_space", secondary["log_space"]),
    ):
        if not isinstance(values, Mapping) or set(values) != expected_metric_fields:
            raise ValueError(f"{namespace} has invalid metric fields")
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"{namespace} must contain finite values")
    if not np.isfinite(float(secondary["spearman"])):
        raise ValueError("secondary Spearman must be finite")


def write_baseline_result_json(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and write one result row without overwriting by default."""

    validate_baseline_result_row(result)
    return write_metrics_json(path, result, overwrite=overwrite)


def verify_baseline_result_from_prediction_csv(
    prediction_csv: str | Path,
    result: Mapping[str, Any],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> dict[str, Any]:
    """Recompute all stored baseline metrics solely from its prediction CSV."""

    validate_baseline_result_row(result)
    values = read_prediction_csv(prediction_csv)
    recalculated = metrics_by_space(
        y_true_original=values["true_original"],
        y_pred_original=values["pred_original"],
        y_true_log=values["true_log"],
        y_pred_log=values["pred_log"],
    )
    recalculated_spearman = spearman_correlation(
        values["true_original"],
        values["pred_original"],
    )
    expected = {
        "original_unit": result["headline_metrics"]["original_unit"],
        "log_space": result["secondary_metrics"]["log_space"],
    }
    differences: list[str] = []
    for space in ("original_unit", "log_space"):
        for name in ("n", *METRIC_NAMES):
            actual_value = recalculated[space][name]
            expected_value = expected[space][name]
            if name == "n":
                matches = int(actual_value) == int(expected_value)
            else:
                matches = bool(
                    np.isclose(
                        float(actual_value),
                        float(expected_value),
                        rtol=rtol,
                        atol=atol,
                        equal_nan=False,
                    )
                )
            if not matches:
                differences.append(
                    f"{space}.{name}: stored={expected_value!r}, "
                    f"recalculated={actual_value!r}"
                )
    stored_spearman = float(result["secondary_metrics"]["spearman"])
    if not np.isclose(
        recalculated_spearman,
        stored_spearman,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        differences.append(
            "spearman: "
            f"stored={stored_spearman!r}, recalculated={recalculated_spearman!r}"
        )
    if differences:
        raise MetricMismatchError(
            "prediction/baseline-result mismatch:\n" + "\n".join(differences)
        )
    return {
        "headline_metrics": {
            "original_unit": recalculated["original_unit"],
        },
        "secondary_metrics": {
            "log_space": recalculated["log_space"],
            "spearman": recalculated_spearman,
        },
    }
