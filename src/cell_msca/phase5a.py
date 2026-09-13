"""Phase 5A validation-only convergence, repetition, and robust aggregation.

The module reuses the frozen Phase 3/4 trainers and evaluators. It has no test
materialization or test-evaluation entry point.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import (
    LightGBMConfig,
    _log_prediction_pair,
    _original_prediction_pair,
    evaluate_fitted_baseline,
    fit_lightgbm_baseline,
)
from .data import canonical_sha256, file_sha256
from .evaluate import (
    BASELINE_RESULT_SCHEMA_VERSION,
    LEGACY_BASELINE_RESULT_SCHEMA_VERSION,
    read_prediction_csv,
    verify_baseline_result_from_prediction_csv,
    verify_legacy_projection_identity_from_prediction_csv,
    write_metrics_json,
    write_prediction_csv,
    write_prediction_support_diagnostic_csv,
)
from .kaggle_runner import (
    GitIdentity,
    load_kaggle_validation_config,
    resolve_git_identity,
    run_kaggle_validation,
)
from .metrics import METRIC_NAMES
from .target import NONNEGATIVE_PREDICTION_SUPPORT_POLICY
from .v1_validation import VerifiedV1Archive, verified_v1_input

PHASE5A_CONFIG_SCHEMA_VERSION = "cell_msca.phase5a_validation_robustness.v1"
PHASE5A_ASSIGNMENT_SCHEMA_VERSION = "cell_msca.phase5a_assignment.v1"
LEGACY_PHASE5A_RUN_SCHEMA_VERSION = "cell_msca.phase5a_validation_run.v1"
PHASE5A_RUN_SCHEMA_VERSION = "cell_msca.phase5a_validation_run.v2"
PHASE5A_SELECTION_SCHEMA_VERSION = "cell_msca.phase5a_lightgbm_selection.v2"
PHASE5A_AGGREGATION_SCHEMA_VERSION = "cell_msca.phase5a_aggregation.v1"
LIGHTGBM_CANDIDATES = ("lightgbm_raw", "lightgbm_log1p")
NEURAL_CANDIDATES = ("cell_msca_bidirectional", "cell_msca_token_no_attention")
REPEATED_SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
_LIGHTGBM_CANDIDATE_RESULT_FIELDS = frozenset(
    {
        "validation_original_unit_mae",
        "best_iteration",
        "actual_iterations",
        "maximum_iteration_reached",
        "configuration_sha256",
        "manifest_sha256",
        "prediction_sha256",
        "prediction_support_policy",
        "pre_projection_negative_count",
        "pre_projection_negative_fraction",
        "pre_projection_minimum",
        "projection_applied_count",
    }
)


@dataclass(frozen=True)
class StoredValidationArtifact:
    directory: Path
    manifest_path: Path
    metrics_path: Path
    prediction_path: Path
    manifest: Mapping[str, Any]
    result: Mapping[str, Any]
    predictions: Mapping[str, NDArray[Any]]

    @property
    def model_name(self) -> str:
        return str(self.result["model_name"])

    @property
    def train_seed(self) -> int:
        return int(self.result["train_seed"])

    @property
    def git_commit_sha(self) -> str:
        return str(self.manifest["git_commit_sha"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return values


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _version("numpy"),
        "pytorch": _version("torch"),
        "lightgbm": _version("lightgbm"),
    }


def load_phase5a_config(path: str | Path) -> dict[str, Any]:
    """Load the frozen robustness contract and reject scope expansion."""

    values = _read_json(path)
    if values.get("schema_version") != PHASE5A_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 5A configuration schema_version")
    if values.get("stage") != "validation_only":
        raise ValueError("Phase 5A stage must be validation_only")
    if values.get("allowed_materialized_splits") != ["train", "validation"]:
        raise ValueError("Phase 5A may materialize train and validation only")
    if (
        values.get("prediction_support_policy")
        != NONNEGATIVE_PREDICTION_SUPPORT_POLICY
    ):
        raise ValueError("Phase 5A prediction support policy changed")
    seeds = values.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("Phase 5A config requires seeds")
    if tuple(seeds.get("all", ())) != REPEATED_SEEDS:
        raise ValueError("Phase 5A seeds must be fixed to [42, 43, 44]")
    if tuple(seeds.get("new_training", ())) != NEW_SEEDS:
        raise ValueError("Phase 5A new training seeds must be [43, 44]")
    if int(seeds.get("split_seed", -1)) != 42:
        raise ValueError("Phase 5A split_seed must remain 42")

    convergence = values.get("lightgbm_convergence")
    if not isinstance(convergence, dict):
        raise ValueError("Phase 5A config requires lightgbm_convergence")
    if tuple(convergence.get("candidates", ())) != LIGHTGBM_CANDIDATES:
        raise ValueError("convergence candidates must be raw and log1p LightGBM")
    parameters = convergence.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("LightGBM convergence parameters are missing")
    fixed_parameters = {
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "early_stopping_rounds": 200,
    }
    if parameters != fixed_parameters:
        raise ValueError(
            "Phase 5A LightGBM parameters differ from the frozen convergence check"
        )
    if convergence.get("selection_metric") != "validation_original_unit_mae":
        raise ValueError("LightGBM selection must use validation original-unit MAE")

    repeated = values.get("repeated_models")
    if not isinstance(repeated, dict):
        raise ValueError("Phase 5A config requires repeated_models")
    if tuple(repeated.get("neural", ())) != (
        "bidirectional",
        "token_no_attention",
    ):
        raise ValueError("only bidirectional and token_no_attention may be repeated")
    if repeated.get("lightgbm") != "selected_by_convergence_gate":
        raise ValueError("repeated LightGBM must pass the convergence gate")
    if set(repeated.get("excluded", ())) != {
        "forward",
        "reverse",
        "concat_mlp",
        "lightgbm_tweedie",
    }:
        raise ValueError("Phase 5A excluded-model contract changed")

    required_hashes = values.get("required_hashes")
    if not isinstance(required_hashes, dict):
        raise ValueError("Phase 5A required_hashes are missing")
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    ):
        digest = str(required_hashes.get(name, ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid Phase 5A hash: {name}")

    base_git_sha = str(values.get("required_base_git_sha", ""))
    if len(base_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in base_git_sha
    ):
        raise ValueError("required_base_git_sha must be a 40-character Git SHA")
    seed42_hashes = values.get("seed42_expected_configuration_sha256")
    expected_seed42_models = {
        "cell_msca_bidirectional",
        "cell_msca_token_no_attention",
        "lightgbm_raw",
        "lightgbm_log1p",
    }
    if not isinstance(seed42_hashes, dict) or set(seed42_hashes) != (
        expected_seed42_models
    ):
        raise ValueError("seed 42 configuration hashes must cover the four references")
    for model_name, digest_value in seed42_hashes.items():
        digest = str(digest_value)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid seed 42 configuration hash: {model_name}")
    reference_mae = values.get("seed42_reference_mae_rounded_6dp")
    if not isinstance(reference_mae, dict) or set(reference_mae) != (
        expected_seed42_models
    ):
        raise ValueError("seed 42 reference MAE values must cover the four references")
    if not all(np.isfinite(float(value)) for value in reference_mae.values()):
        raise ValueError("seed 42 reference MAE values must be finite")

    runtime_configs = values.get("cell_runtime_configs")
    if not isinstance(runtime_configs, dict) or set(runtime_configs) != {"43", "44"}:
        raise ValueError("Cell-MSCA runtime configs must cover seeds 43 and 44")
    cell_hashes = values.get("cell_required_hashes_by_seed")
    if not isinstance(cell_hashes, dict) or set(cell_hashes) != {"43", "44"}:
        raise ValueError("Cell-MSCA required hashes must cover seeds 43 and 44")
    for seed, hashes in cell_hashes.items():
        if not isinstance(hashes, dict):
            raise ValueError(f"Cell-MSCA seed {seed} hashes must be an object")
        for name, digest in required_hashes.items():
            if hashes.get(name) != digest:
                raise ValueError(f"Cell-MSCA seed {seed} {name} changed")
        configuration_hashes = hashes.get(
            "configuration_sha256_by_variant_and_device"
        )
        expected_keys = {
            "token_no_attention:cpu",
            "token_no_attention:cuda",
            "bidirectional:cpu",
            "bidirectional:cuda",
        }
        if not isinstance(configuration_hashes, dict) or set(
            configuration_hashes
        ) != expected_keys:
            raise ValueError(f"Cell-MSCA seed {seed} configuration hashes changed")

    bootstrap = values.get("paired_cluster_bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("paired_cluster_bootstrap is required")
    if int(bootstrap.get("n_boot", 0)) <= 0:
        raise ValueError("paired cluster bootstrap n_boot must be positive")
    if float(bootstrap.get("alpha", 0.0)) != 0.05:
        raise ValueError("Phase 5A bootstrap alpha must remain 0.05")
    if int(bootstrap.get("rows_per_cell", 0)) != 36:
        raise ValueError("Phase 5A bootstrap requires 36 rows per validation cell")
    if bootstrap.get("difference_orientation") != (
        "mae_bidirectional_minus_mae_comparator"
    ):
        raise ValueError("paired MAE difference orientation changed")
    return values


def load_phase5a_assignments(path: str | Path) -> dict[str, Any]:
    values = _read_json(path)
    if values.get("schema_version") != PHASE5A_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 5A assignment schema_version")
    assignments = values.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Phase 5A assignments must not be empty")
    if values.get("stage") != "validation_only" or values.get("test_gate") != (
        "closed"
    ):
        raise ValueError("Phase 5A assignments must keep the validation-only test gate")
    if values.get("split_seed") != 42 or values.get("fixed_train_seeds") != (
        [42, 43, 44]
    ):
        raise ValueError("Phase 5A assignment seed contract changed")
    required_fields = {
        "owner",
        "experiment_id",
        "model_family",
        "variant",
        "train_seed",
        "device",
        "status",
        "output_location",
        "notes",
    }
    for index, item in enumerate(assignments):
        if not isinstance(item, dict) or set(item) != required_fields:
            raise ValueError(f"invalid Phase 5A assignment at index {index}")
        if int(item["train_seed"]) not in REPEATED_SEEDS:
            raise ValueError("Phase 5A assignment contains an unplanned train seed")
    identifiers = [str(item["experiment_id"]) for item in assignments]
    outputs = [str(item["output_location"]) for item in assignments]
    if any(not value for value in identifiers + outputs):
        raise ValueError("Phase 5A assignment identifiers and outputs must not be empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate Phase 5A experiment IDs are not allowed")
    if len(outputs) != len(set(outputs)):
        raise ValueError("duplicate Phase 5A output locations are not allowed")
    forbidden = {"forward", "reverse", "concat_mlp", "lightgbm_tweedie"}
    for item in assignments:
        if str(item.get("variant")) in forbidden and item.get("status") != "excluded":
            raise ValueError("an excluded Phase 5A model was scheduled")
    return values


def _validate_provenance(
    verified: VerifiedV1Archive,
    config: Mapping[str, Any],
    *,
    train_seed: int,
) -> None:
    provenance = verified.protocol.provenance
    expected = config["required_hashes"]
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    ):
        if getattr(provenance, name) != expected[name]:
            raise ValueError(f"Phase 5A verified input {name} mismatch")
    if provenance.split_seed != 42:
        raise ValueError("Phase 5A split seed must remain 42")
    if provenance.train_seed != train_seed:
        raise ValueError("Phase 5A verified protocol train_seed mismatch")


def _lightgbm_config(
    values: Mapping[str, Any],
    *,
    model_name: str,
) -> LightGBMConfig:
    if model_name not in LIGHTGBM_CANDIDATES:
        raise ValueError("Phase 5A only permits raw or log1p LightGBM")
    return LightGBMConfig(
        model_name=model_name,
        **dict(values["lightgbm_convergence"]["parameters"]),
    )


def _lightgbm_experiment_id(
    values: Mapping[str, Any],
    *,
    run_type: str,
    model_name: str,
    seed: int,
    git_commit_sha: str,
) -> str:
    suffix = model_name.removeprefix("lightgbm_")
    fingerprint = canonical_sha256(
        {
            "run_type": run_type,
            "model_name": model_name,
            "train_seed": seed,
            "split_seed": 42,
            "required_hashes": values["required_hashes"],
            "parameters": values["lightgbm_convergence"]["parameters"],
            "git_commit_sha": git_commit_sha,
        }
    )
    return f"phase5a_{run_type}_lightgbm_{suffix}_seed{seed}-{fingerprint[:12]}"


def _positive_iteration_count(value: Any, *, source: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise RuntimeError(f"{source} must provide a positive integer iteration count")
    count = int(value)
    if count <= 0:
        raise RuntimeError(f"{source} must provide a positive integer iteration count")
    return count


def _lightgbm_actual_iterations(estimator: Any) -> int:
    """Return the true fitted iteration count from documented LightGBM APIs."""

    for attribute in ("n_estimators_", "n_iter_"):
        try:
            value = getattr(estimator, attribute)
        except AttributeError:
            continue
        if value is not None:
            return _positive_iteration_count(value, source=f"LightGBM {attribute}")

    try:
        booster = getattr(estimator, "booster_")
    except AttributeError:
        booster = None
    if booster is not None:
        current_iteration = getattr(booster, "current_iteration", None)
        if callable(current_iteration):
            return _positive_iteration_count(
                current_iteration(),
                source="LightGBM booster_.current_iteration()",
            )
    raise RuntimeError(
        "unable to determine actual LightGBM iterations: fitted estimator exposes "
        "neither n_estimators_, n_iter_, nor booster_.current_iteration()"
    )


def _lightgbm_iteration_summary(
    estimator: Any,
    *,
    best_iteration: int,
    configured_n_estimators: int,
) -> dict[str, Any]:
    best = _positive_iteration_count(
        best_iteration,
        source="LightGBM best_iteration",
    )
    upper_bound = _positive_iteration_count(
        configured_n_estimators,
        source="configured LightGBM n_estimators",
    )
    actual = _lightgbm_actual_iterations(estimator)
    return {
        "best_iteration": best,
        "actual_iterations": actual,
        "maximum_iteration_reached": actual >= upper_bound,
    }


def _support_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prediction_support_policy": result["prediction_support_policy"],
        "pre_projection_negative_count": result[
            "pre_projection_negative_count"
        ],
        "pre_projection_negative_fraction": result[
            "pre_projection_negative_fraction"
        ],
        "pre_projection_minimum": result["pre_projection_minimum"],
        "projection_applied_count": result["projection_applied_count"],
    }


def _save_and_verify_lightgbm_model(
    path: str | Path,
    *,
    fitted: Any,
    validation_features: NDArray[np.float64],
    expected_final_prediction: NDArray[np.float64],
) -> dict[str, Any]:
    """Atomically save a fitted LightGBM booster and verify its predictions."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite LightGBM model: {destination}")
    booster = getattr(fitted.estimator, "booster_", None)
    if booster is None or not callable(getattr(booster, "save_model", None)):
        raise RuntimeError("fitted LightGBM estimator does not expose booster_.save_model")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        booster.save_model(
            str(temporary),
            num_iteration=int(fitted.contract.best_iteration),
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("LightGBM model serialization produced no artifact")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

    lightgbm = importlib.import_module("lightgbm")
    reloaded = lightgbm.Booster(model_file=str(destination))
    reloaded_native = np.asarray(
        reloaded.predict(np.asarray(validation_features, dtype=np.float64)),
        dtype=np.float64,
    ).ravel()
    if fitted.target_space == "log":
        reloaded_predictions = _log_prediction_pair(
            reloaded_native,
            scale=fitted.contract.target_scale,
            mode=fitted.contract.inverse_mode,
            smearing_factor=fitted.contract.smearing_factor,
        )
    elif fitted.target_space == "original":
        reloaded_predictions = _original_prediction_pair(
            reloaded_native,
            target_scale=fitted.contract.target_scale,
            model_name=fitted.contract.model_name,
        )
    else:
        raise RuntimeError(f"unsupported fitted LightGBM target_space: {fitted.target_space}")
    expected = np.asarray(expected_final_prediction, dtype=np.float64).ravel()
    if expected.shape != reloaded_predictions.pred_original.shape or not np.allclose(
        reloaded_predictions.pred_original,
        expected,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("reloaded LightGBM validation predictions do not match")
    maximum_absolute_difference = float(
        np.max(np.abs(reloaded_predictions.pred_original - expected))
    )
    return {
        "model_artifact": destination.name,
        "model_sha256": file_sha256(destination),
        "model_reload_validation_prediction_match": True,
        "model_reload_maximum_absolute_prediction_difference": (
            maximum_absolute_difference
        ),
        "model_saved_best_iteration": int(fitted.contract.best_iteration),
    }


def _ensure_full_validation_allowed(
    values: Mapping[str, Any],
    allow_full_validation: bool,
) -> None:
    if values.get("execution_policy", {}).get("default_action") != "integrity_only":
        raise ValueError("Phase 5A default action must remain integrity_only")
    if not allow_full_validation:
        raise RuntimeError(
            "Phase 5A project-data training is disabled by default; "
            "pass allow_full_validation=True after integrity and smoke"
        )


def _is_posix_relative_to(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_phase5a_output_path(
    output_root: str | Path,
    *,
    kaggle: bool,
) -> Path:
    output = Path(output_root).resolve()
    posix = PurePosixPath(str(output_root).replace("\\", "/"))
    if _is_posix_relative_to(posix, PurePosixPath("/kaggle/input")):
        raise ValueError("Phase 5A output must not be below read-only /kaggle/input")
    if kaggle and not _is_posix_relative_to(posix, PurePosixPath("/kaggle/working")):
        raise ValueError("Kaggle Phase 5A output must be below /kaggle/working")
    return output


def run_phase5a_lightgbm(
    verified: VerifiedV1Archive,
    *,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_git_sha: str,
    model_name: str,
    train_seed: int,
    run_type: str,
    selection_path: str | Path | None = None,
    allow_full_validation: bool = False,
    git_identity: GitIdentity | None = None,
) -> Path:
    """Run one frozen LightGBM convergence or gated repeated-seed validation."""

    values = load_phase5a_config(config_path)
    _ensure_full_validation_allowed(values, allow_full_validation)
    if run_type not in {"convergence", "repeat"}:
        raise ValueError("run_type must be convergence or repeat")
    if run_type == "convergence" and train_seed != 42:
        raise ValueError("the convergence check is fixed to train_seed=42")
    if run_type == "repeat" and train_seed not in NEW_SEEDS:
        raise ValueError("new LightGBM repeats are fixed to seeds 43 and 44")
    if run_type == "repeat":
        if selection_path is None:
            raise RuntimeError("repeated LightGBM requires a frozen convergence selection")
        selection = load_lightgbm_selection(selection_path, config_path=config_path)
        if selection.get("git_commit_sha") != str(expected_git_sha).lower():
            raise ValueError("LightGBM selection Git SHA differs from this runner")
        selected_model = str(selection["selected_model_name"])
        if model_name != selected_model:
            raise RuntimeError(
                f"LightGBM gate selected {selected_model}; refusing {model_name}"
            )
    elif selection_path is not None:
        raise ValueError("convergence runs must not receive a selection file")

    _validate_provenance(verified, values, train_seed=train_seed)
    identity = git_identity or resolve_git_identity(
        repository_root,
        expected_git_sha=expected_git_sha,
    )
    if identity.commit_sha != str(expected_git_sha).lower():
        raise ValueError("injected Git identity does not match expected_git_sha")
    if identity.worktree_dirty:
        raise RuntimeError("Phase 5A LightGBM requires a clean source tree")
    experiment_id = _lightgbm_experiment_id(
        values,
        run_type=run_type,
        model_name=model_name,
        seed=train_seed,
        git_commit_sha=identity.commit_sha,
    )
    output_dir = _validate_phase5a_output_path(
        output_root,
        kaggle=verified.kaggle,
    ) / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Phase 5A run: {output_dir}")
    output_dir.mkdir(parents=True)

    started_at = _utc_now()
    data = verified.protocol.tuning_data()
    candidate = _lightgbm_config(values, model_name=model_name)
    fitted = fit_lightgbm_baseline(data, candidate)
    evaluation = evaluate_fitted_baseline(
        fitted,
        data.validation,
        data.provenance,
    )
    result = evaluation.result
    upper_bound = int(values["lightgbm_convergence"]["parameters"]["n_estimators"])
    iteration_summary = _lightgbm_iteration_summary(
        fitted.estimator,
        best_iteration=int(result["best_iteration"]),
        configured_n_estimators=upper_bound,
    )
    prediction_path = output_dir / "validation_predictions.csv"
    diagnostic_path = output_dir / "validation_negative_predictions.csv"
    metrics_path = output_dir / "validation_metrics.json"
    manifest_path = output_dir / "run_manifest.json"
    environment_path = output_dir / "environment.json"
    write_prediction_csv(
        prediction_path,
        y_true_original=data.validation.target_original,
        y_pred_original=evaluation.predictions.pred_original,
        y_true_log=data.validation.target_log,
        y_pred_log=evaluation.predictions.pred_log,
        cell_ids=data.validation.cell_ids,
    )
    write_prediction_support_diagnostic_csv(
        diagnostic_path,
        y_true_original=data.validation.target_original,
        unprojected_prediction=evaluation.predictions.unprojected_original,
        final_prediction=evaluation.predictions.pred_original,
        cell_ids=data.validation.cell_ids,
        month_ids=getattr(data.validation, "month_ids", None),
    )
    verify_baseline_result_from_prediction_csv(prediction_path, result)
    model_artifact = _save_and_verify_lightgbm_model(
        output_dir / "fitted_model.txt",
        fitted=fitted,
        validation_features=data.validation.features,
        expected_final_prediction=evaluation.predictions.pred_original,
    )
    write_metrics_json(
        metrics_path,
        {
            "schema_version": "cell_msca.validation_metrics_artifact.v1",
            "artifact_classification": "validation-only",
            "prediction_metric_verification": "passed_including_spearman",
            "actual_iterations": iteration_summary["actual_iterations"],
            **_support_fields(result),
            "result": result,
        },
    )
    environment = _runtime_environment()
    write_metrics_json(environment_path, environment)
    write_metrics_json(
        manifest_path,
        {
            "schema_version": PHASE5A_RUN_SCHEMA_VERSION,
            "artifact_classification": "validation-only",
            "stage": "validation_only",
            "status": "completed",
            "owner": "cpu-owner",
            "run_type": run_type,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "train_seed": train_seed,
            "split_seed": 42,
            "data_sha256": result["data_sha256"],
            "split_sha256": result["split_sha256"],
            "split_config_sha256": result["split_config_sha256"],
            "preprocessing_sha256": result["preprocessing_sha256"],
            "configuration_sha256": result["config_sha256"],
            "phase5a_config_sha256": file_sha256(config_path),
            "required_base_git_sha": values["required_base_git_sha"],
            "git_commit_sha": identity.commit_sha,
            "git_dirty_state_policy": identity.dirty_state_policy,
            "input_mode": verified.input_mode,
            "input_verification": dict(verified.summary),
            "n_estimators_upper_bound": upper_bound,
            "early_stopping_rounds": int(
                values["lightgbm_convergence"]["parameters"][
                    "early_stopping_rounds"
                ]
            ),
            **iteration_summary,
            "selection_metric": "validation_original_unit_mae",
            **_support_fields(result),
            "negative_prediction_diagnostic_csv": diagnostic_path.name,
            **model_artifact,
            "allowed_materialized_splits": ["train", "validation"],
            "test_subset_materialized": False,
            "test_evaluation_performed": False,
            "runtime_versions": environment,
            "start_time_utc": started_at,
            "completion_time_utc": _utc_now(),
            "validation_metrics": result,
        },
    )
    return output_dir


def load_stored_validation_artifact(path: str | Path) -> StoredValidationArtifact:
    directory = Path(path).resolve()
    manifest_path = directory / "run_manifest.json"
    metrics_path = directory / "validation_metrics.json"
    prediction_path = directory / "validation_predictions.csv"
    for required in (manifest_path, metrics_path, prediction_path):
        if not required.is_file():
            raise FileNotFoundError(f"validation artifact is missing: {required}")
    manifest = _read_json(manifest_path)
    metrics = _read_json(metrics_path)
    result = metrics.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"validation metrics result is missing: {metrics_path}")
    verify_baseline_result_from_prediction_csv(prediction_path, result)
    predictions = read_prediction_csv(prediction_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"validation artifact is not completed: {directory}")
    if manifest.get("test_subset_materialized") is not False:
        raise ValueError("validation artifact does not prove a closed test gate")
    if manifest.get("test_evaluation_performed", False) is not False:
        raise ValueError("validation artifact reports a test evaluation")
    if manifest.get("allowed_materialized_splits") != ["train", "validation"]:
        raise ValueError("validation artifact materialized an unauthorized split")
    if result.get("evaluation_split") != "validation":
        raise ValueError("Phase 5A accepts validation prediction artifacts only")
    if str(manifest.get("model_name")) != str(result["model_name"]):
        raise ValueError("manifest/result model_name mismatch")
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
        "train_seed",
        "split_seed",
    ):
        if manifest.get(name) != result[name]:
            raise ValueError(f"manifest/result mismatch: {name}")
    manifest_config = manifest.get(
        "configuration_sha256",
        manifest.get("config_sha256"),
    )
    if manifest_config != result["config_sha256"]:
        raise ValueError("manifest/result configuration hash mismatch")
    schema_version = result.get("schema_version")
    if schema_version == LEGACY_BASELINE_RESULT_SCHEMA_VERSION:
        manifest = {
            **manifest,
            **verify_legacy_projection_identity_from_prediction_csv(prediction_path),
        }
    elif schema_version == BASELINE_RESULT_SCHEMA_VERSION:
        expected_support = _support_fields(result)
        for field_name, expected_value in expected_support.items():
            if manifest.get(field_name) != expected_value:
                raise ValueError(
                    f"manifest/result prediction support mismatch: {field_name}"
                )
    else:
        raise ValueError("unsupported stored baseline result schema_version")
    if manifest.get("schema_version") == PHASE5A_RUN_SCHEMA_VERSION:
        model_artifact_name = manifest.get("model_artifact")
        if (
            not isinstance(model_artifact_name, str)
            or Path(model_artifact_name).name != model_artifact_name
        ):
            raise ValueError("Phase 5A LightGBM manifest model_artifact is invalid")
        model_artifact_path = directory / model_artifact_name
        if not model_artifact_path.is_file():
            raise FileNotFoundError(
                f"Phase 5A LightGBM model artifact is missing: {model_artifact_path}"
            )
        if file_sha256(model_artifact_path) != manifest.get("model_sha256"):
            raise ValueError("Phase 5A LightGBM model SHA-256 mismatch")
        if manifest.get("model_reload_validation_prediction_match") is not True:
            raise ValueError("Phase 5A LightGBM model reload was not verified")
    return StoredValidationArtifact(
        directory,
        manifest_path,
        metrics_path,
        prediction_path,
        manifest,
        result,
        predictions,
    )


def discover_validation_artifacts(
    root: str | Path,
) -> dict[tuple[str, int], StoredValidationArtifact]:
    artifact_root = Path(root).resolve()
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"validation package directory is missing: {artifact_root}")
    artifacts: dict[tuple[str, int], StoredValidationArtifact] = {}
    for manifest_path in sorted(artifact_root.rglob("run_manifest.json")):
        directory = manifest_path.parent
        if not (directory / "validation_predictions.csv").is_file():
            continue
        artifact = load_stored_validation_artifact(directory)
        key = (artifact.model_name, artifact.train_seed)
        if key in artifacts:
            raise ValueError(f"duplicate validation artifact for {key}")
        artifacts[key] = artifact
    if not artifacts:
        raise ValueError(f"no complete validation artifacts found below {artifact_root}")
    return artifacts


def _validate_artifact_contract(
    artifact: StoredValidationArtifact,
    values: Mapping[str, Any],
    *,
    expected_git_sha: str,
    expected_config_sha256: str | None = None,
) -> None:
    result = artifact.result
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    ):
        expected = values["required_hashes"][name]
        if result.get(name) != expected:
            raise ValueError(f"stored validation artifact {name} mismatch")
    if artifact.git_commit_sha != expected_git_sha:
        raise ValueError(
            "stored validation artifact Git SHA mismatch: "
            f"expected={expected_git_sha}, actual={artifact.git_commit_sha}"
        )
    if expected_config_sha256 is not None and result["config_sha256"] != (
        expected_config_sha256
    ):
        raise ValueError("stored validation artifact configuration SHA-256 mismatch")


def _selection_positive_iteration(value: Any, *, field: str) -> int:
    try:
        return _positive_iteration_count(value, source=field)
    except RuntimeError as error:
        raise ValueError(
            f"invalid LightGBM selection candidate field: {field}"
        ) from error


def _selection_sha256(value: Any, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"invalid LightGBM selection candidate SHA-256: {field}")
    return digest


def _validated_lightgbm_candidate_results(
    selection: Mapping[str, Any],
    *,
    master: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    candidates = selection.get("candidate_results")
    if not isinstance(candidates, dict) or set(candidates) != set(
        LIGHTGBM_CANDIDATES
    ):
        raise ValueError(
            "LightGBM selection candidates must be exactly lightgbm_raw and "
            "lightgbm_log1p"
        )
    upper_bound = int(master["lightgbm_convergence"]["parameters"]["n_estimators"])
    for model_name in LIGHTGBM_CANDIDATES:
        row = candidates[model_name]
        if not isinstance(row, dict) or set(row) != _LIGHTGBM_CANDIDATE_RESULT_FIELDS:
            raise ValueError(f"invalid LightGBM selection candidate schema: {model_name}")
        mae = row["validation_original_unit_mae"]
        if (
            isinstance(mae, (bool, np.bool_))
            or not isinstance(mae, Real)
            or not np.isfinite(float(mae))
        ):
            raise ValueError(
                "LightGBM candidate validation_original_unit_mae must be a finite number"
            )
        _selection_positive_iteration(
            row["best_iteration"],
            field=f"{model_name}.best_iteration",
        )
        actual_iterations = _selection_positive_iteration(
            row["actual_iterations"],
            field=f"{model_name}.actual_iterations",
        )
        maximum_reached = row["maximum_iteration_reached"]
        if not isinstance(maximum_reached, bool):
            raise ValueError(
                f"{model_name}.maximum_iteration_reached must be a boolean"
            )
        if maximum_reached != (actual_iterations >= upper_bound):
            raise ValueError(
                f"{model_name}.maximum_iteration_reached disagrees with actual_iterations"
            )
        for field in (
            "configuration_sha256",
            "manifest_sha256",
            "prediction_sha256",
        ):
            _selection_sha256(row[field], field=f"{model_name}.{field}")
        support_fields = {
            field: row[field]
            for field in (
                "prediction_support_policy",
                "pre_projection_negative_count",
                "pre_projection_negative_fraction",
                "pre_projection_minimum",
                "projection_applied_count",
            )
        }
        if support_fields["prediction_support_policy"] != (
            NONNEGATIVE_PREDICTION_SUPPORT_POLICY
        ):
            raise ValueError("LightGBM candidate prediction support policy changed")
        if isinstance(
            support_fields["pre_projection_negative_count"], bool
        ) or not isinstance(support_fields["pre_projection_negative_count"], int):
            raise ValueError("LightGBM candidate negative count must be an integer")
        if isinstance(support_fields["projection_applied_count"], bool) or not isinstance(
            support_fields["projection_applied_count"], int
        ):
            raise ValueError("LightGBM candidate projection count must be an integer")
        if support_fields["pre_projection_negative_count"] < 0:
            raise ValueError("LightGBM candidate negative count must be nonnegative")
        if support_fields["projection_applied_count"] != support_fields[
            "pre_projection_negative_count"
        ]:
            raise ValueError("LightGBM candidate projection diagnostics disagree")
        for field in (
            "pre_projection_negative_fraction",
            "pre_projection_minimum",
        ):
            if not isinstance(support_fields[field], Real) or not np.isfinite(
                float(support_fields[field])
            ):
                raise ValueError(f"LightGBM candidate {field} must be finite")
        fraction = float(support_fields["pre_projection_negative_fraction"])
        minimum = float(support_fields["pre_projection_minimum"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("LightGBM candidate negative fraction is invalid")
        if (support_fields["pre_projection_negative_count"] == 0) != (
            minimum >= 0.0
        ):
            raise ValueError("LightGBM candidate negative minimum/count disagree")
        expected_config_sha256 = canonical_sha256(
            _lightgbm_config(master, model_name=model_name).to_dict(
                train_seed=42,
                target_scale=float(master["data"]["target_scale"]),
            )
        )
        if row["configuration_sha256"] != expected_config_sha256:
            raise ValueError(
                f"LightGBM selection candidate configuration mismatch: {model_name}"
            )
    return candidates


def freeze_lightgbm_convergence_selection(
    *,
    raw_run_dir: str | Path,
    log1p_run_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    expected_git_sha: str,
) -> Path:
    """Freeze the lower-MAE convergence candidate before seeds 43/44 run."""

    values = load_phase5a_config(config_path)
    artifacts = {
        "lightgbm_raw": load_stored_validation_artifact(raw_run_dir),
        "lightgbm_log1p": load_stored_validation_artifact(log1p_run_dir),
    }
    candidate_rows: dict[str, Any] = {}
    for model_name, artifact in artifacts.items():
        if artifact.model_name != model_name or artifact.train_seed != 42:
            raise ValueError("convergence selection received the wrong model or seed")
        if artifact.manifest.get("run_type") != "convergence":
            raise ValueError("LightGBM selection requires convergence-run artifacts")
        expected_config = canonical_sha256(
            _lightgbm_config(values, model_name=model_name).to_dict(
                train_seed=42,
                target_scale=float(values["data"]["target_scale"]),
            )
        )
        _validate_artifact_contract(
            artifact,
            values,
            expected_git_sha=expected_git_sha,
            expected_config_sha256=expected_config,
        )
        metrics_artifact = _read_json(artifact.metrics_path)
        metrics_actual_iterations = _selection_positive_iteration(
            metrics_artifact.get("actual_iterations"),
            field=f"{model_name}.validation_metrics.actual_iterations",
        )
        manifest_actual_iterations = _selection_positive_iteration(
            artifact.manifest.get("actual_iterations"),
            field=f"{model_name}.run_manifest.actual_iterations",
        )
        if metrics_actual_iterations != manifest_actual_iterations:
            raise ValueError(
                f"LightGBM actual_iterations mismatch between metrics and manifest: {model_name}"
            )
        candidate_rows[model_name] = {
            "validation_original_unit_mae": artifact.result["headline_metrics"][
                "original_unit"
            ]["mae"],
            "best_iteration": artifact.result["best_iteration"],
            "actual_iterations": manifest_actual_iterations,
            "maximum_iteration_reached": artifact.manifest[
                "maximum_iteration_reached"
            ],
            "configuration_sha256": artifact.result["config_sha256"],
            "manifest_sha256": file_sha256(artifact.manifest_path),
            "prediction_sha256": file_sha256(artifact.prediction_path),
            **_support_fields(artifact.result),
        }
    selected_model = min(
        LIGHTGBM_CANDIDATES,
        key=lambda name: (
            float(candidate_rows[name]["validation_original_unit_mae"]),
            LIGHTGBM_CANDIDATES.index(name),
        ),
    )
    output = Path(output_path)
    payload = {
        "schema_version": PHASE5A_SELECTION_SCHEMA_VERSION,
        "stage": "validation_only",
        "selection_metric": "validation_original_unit_mae",
        "prediction_support_policy": NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
        "tie_break_order": list(LIGHTGBM_CANDIDATES),
        "selected_model_name": selected_model,
        "selected_parameters": values["lightgbm_convergence"]["parameters"],
        "candidate_results": candidate_rows,
        "required_hashes": values["required_hashes"],
        "phase5a_config_sha256": file_sha256(config_path),
        "git_commit_sha": expected_git_sha,
        "frozen_before_new_seed_training": True,
        "allowed_new_train_seeds": list(NEW_SEEDS),
        "test_subset_materialized": False,
        "test_evaluation_performed": False,
        "created_at_utc": _utc_now(),
    }
    write_metrics_json(output, payload)
    return output


def load_lightgbm_selection(
    path: str | Path,
    *,
    config_path: str | Path,
) -> dict[str, Any]:
    selection = _read_json(path)
    master = load_phase5a_config(config_path)
    if selection.get("schema_version") != PHASE5A_SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported LightGBM convergence selection schema")
    if selection.get("selected_model_name") not in LIGHTGBM_CANDIDATES:
        raise ValueError("LightGBM convergence selection has an invalid model")
    if selection.get("frozen_before_new_seed_training") is not True:
        raise ValueError("LightGBM selection was not frozen before seed repetition")
    if selection.get("allowed_new_train_seeds") != list(NEW_SEEDS):
        raise ValueError("LightGBM selection seed contract changed")
    if selection.get("test_subset_materialized") is not False:
        raise ValueError("LightGBM selection does not prove a closed test gate")
    if selection.get("test_evaluation_performed") is not False:
        raise ValueError("LightGBM selection indicates a test evaluation")
    if selection.get("selection_metric") != "validation_original_unit_mae":
        raise ValueError("LightGBM selection metric changed")
    if (
        selection.get("prediction_support_policy")
        != NONNEGATIVE_PREDICTION_SUPPORT_POLICY
    ):
        raise ValueError("LightGBM selection prediction support policy changed")
    if selection.get("phase5a_config_sha256") != file_sha256(config_path):
        raise ValueError("LightGBM selection/config SHA-256 mismatch")
    if selection.get("required_hashes") != master["required_hashes"]:
        raise ValueError("LightGBM selection provenance hashes changed")
    if selection.get("selected_parameters") != master["lightgbm_convergence"][
        "parameters"
    ]:
        raise ValueError("LightGBM selection parameters changed")
    if selection.get("tie_break_order") != list(LIGHTGBM_CANDIDATES):
        raise ValueError("LightGBM selection tie-break order changed")
    candidates = _validated_lightgbm_candidate_results(selection, master=master)
    selected_model = min(
        LIGHTGBM_CANDIDATES,
        key=lambda name: (
            float(candidates[name]["validation_original_unit_mae"]),
            LIGHTGBM_CANDIDATES.index(name),
        ),
    )
    if selection.get("selected_model_name") != selected_model:
        raise ValueError(
            "LightGBM selected_model_name does not match the candidate validation "
            "original-unit MAE winner"
        )
    return selection


def run_phase5a_cell_repeat(
    verified: VerifiedV1Archive,
    *,
    config_path: str | Path,
    runtime_config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_git_sha: str,
    variant: str,
    train_seed: int,
    device: str,
    allow_full_validation: bool = False,
    git_identity: GitIdentity | None = None,
) -> Path:
    """Run only the two predeclared neural variants at seed 43 or 44."""

    values = load_phase5a_config(config_path)
    _ensure_full_validation_allowed(values, allow_full_validation)
    if train_seed not in NEW_SEEDS:
        raise ValueError("Phase 5A neural repeats are fixed to seeds 43 and 44")
    if variant not in {"bidirectional", "token_no_attention"}:
        raise ValueError("Phase 5A neural repeats exclude this variant")
    _validate_provenance(verified, values, train_seed=train_seed)
    runtime_values = load_kaggle_validation_config(runtime_config_path)
    if int(runtime_values["frozen_train_seed"]) != train_seed:
        raise ValueError("Phase 5A runtime config train seed mismatch")
    referenced = values["cell_runtime_configs"].get(str(train_seed))
    if Path(runtime_config_path).name != Path(str(referenced)).name:
        raise ValueError("Phase 5A runtime config is not the frozen seed config")
    if runtime_values["required_hashes"] != values["cell_required_hashes_by_seed"][
        str(train_seed)
    ]:
        raise ValueError("Phase 5A runtime config hashes differ from the master config")
    _validate_phase5a_output_path(output_root, kaggle=verified.kaggle)
    artifacts = run_kaggle_validation(
        variant=variant,
        train_seed=train_seed,
        config_path=runtime_config_path,
        data_path=verified.data_dir,
        output_path=output_root,
        device=device,
        expected_git_sha=expected_git_sha,
        kaggle=verified.kaggle and verified.input_mode == "expanded_npz",
        repository_root=repository_root,
        git_identity=git_identity,
    )
    return artifacts.output_dir


def _aligned_prediction_arrays(
    artifacts: Sequence[StoredValidationArtifact],
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float64]]:
    if not artifacts:
        raise ValueError("at least one validation artifact is required")
    reference = artifacts[0].predictions
    for artifact in artifacts[1:]:
        current = artifact.predictions
        if not np.array_equal(current["cell_id"], reference["cell_id"]):
            raise ValueError("validation prediction row/cell_id alignment mismatch")
        if not np.array_equal(current["true_original"], reference["true_original"]):
            raise ValueError("validation true_original alignment mismatch")
        if not np.array_equal(current["true_log"], reference["true_log"]):
            raise ValueError("validation true_log alignment mismatch")
    return (
        np.asarray(reference["cell_id"], dtype=str),
        np.asarray(reference["true_original"], dtype=np.float64),
        np.asarray(reference["true_log"], dtype=np.float64),
    )


def paired_cell_cluster_mae_difference(
    *,
    y_true_original: ArrayLike,
    bidirectional_prediction: ArrayLike,
    comparator_prediction: ArrayLike,
    cell_ids: ArrayLike,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    rows_per_cell: int = 36,
) -> dict[str, Any]:
    """Bootstrap paired MAE(bidirectional)-MAE(comparator) by complete cell."""

    true = np.asarray(y_true_original, dtype=np.float64).ravel()
    bidirectional = np.asarray(bidirectional_prediction, dtype=np.float64).ravel()
    comparator = np.asarray(comparator_prediction, dtype=np.float64).ravel()
    cells = np.asarray(cell_ids, dtype=str).ravel()
    shapes = {true.shape, bidirectional.shape, comparator.shape, cells.shape}
    if len(shapes) != 1 or true.size == 0:
        raise ValueError("paired bootstrap arrays must have one non-empty shape")
    if not all(np.all(np.isfinite(array)) for array in (true, bidirectional, comparator)):
        raise ValueError("paired bootstrap arrays must contain only finite values")
    if not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if rows_per_cell <= 0:
        raise ValueError("rows_per_cell must be positive")

    unique_cells, inverse, counts = np.unique(
        cells,
        return_inverse=True,
        return_counts=True,
    )
    if not np.all(counts == rows_per_cell):
        count_values = sorted(set(int(value) for value in counts))
        raise ValueError(
            "paired bootstrap requires complete cells with exactly "
            f"{rows_per_cell} rows; observed counts={count_values}"
        )
    row_difference = np.abs(bidirectional - true) - np.abs(comparator - true)
    cell_difference = np.zeros(unique_cells.size, dtype=np.float64)
    np.add.at(cell_difference, inverse, row_difference)
    cell_difference /= counts
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        drawn = rng.integers(0, unique_cells.size, size=unique_cells.size)
        bootstrap_values[iteration] = float(np.mean(cell_difference[drawn]))
    lower, upper = np.percentile(
        bootstrap_values,
        [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)],
    )
    return {
        "difference_orientation": "mae_bidirectional_minus_mae_comparator",
        "estimate": float(np.mean(row_difference)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": float(1.0 - alpha),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "unique_cell_count": int(unique_cells.size),
        "rows_per_cell": int(rows_per_cell),
        "complete_cell_resampling": True,
    }


def _seed_metric_summary(
    artifacts: Sequence[StoredValidationArtifact],
    *,
    ddof: int,
) -> dict[str, Any]:
    ordered = sorted(artifacts, key=lambda artifact: artifact.train_seed)
    if [artifact.train_seed for artifact in ordered] != list(REPEATED_SEEDS):
        raise ValueError("model aggregation requires exactly seeds 42, 43, and 44")
    values_by_metric: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    values_by_metric["spearman"] = []
    per_seed: dict[str, Any] = {}
    for artifact in ordered:
        original = artifact.result["headline_metrics"]["original_unit"]
        secondary = artifact.result["secondary_metrics"]
        row = {name: float(original[name]) for name in METRIC_NAMES}
        row["spearman"] = float(secondary["spearman"])
        per_seed[str(artifact.train_seed)] = row
        for name, value in row.items():
            values_by_metric[name].append(value)
    aggregate = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=ddof)),
        }
        for name, values in values_by_metric.items()
    }
    return {"per_seed": per_seed, "aggregate": aggregate, "std_ddof": ddof}


def _expected_cell_config_hash(
    values: Mapping[str, Any],
    *,
    repository_root: Path,
    variant: str,
    seed: int,
    device: str,
) -> str:
    if seed == 42:
        path = repository_root / str(values["seed42_cell_config"])
    else:
        path = repository_root / str(values["cell_runtime_configs"][str(seed)])
    runtime = load_kaggle_validation_config(path)
    key = f"{variant}:{device}"
    return str(
        runtime["required_hashes"]["configuration_sha256_by_variant_and_device"][
            key
        ]
    )


def aggregate_phase5a_results(
    *,
    seed42_package_root: str | Path,
    phase5a_results_root: str | Path,
    lightgbm_selection_path: str | Path,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path,
    expected_new_git_sha: str,
) -> Path:
    """Verify saved predictions, summarize seeds, and run paired cell bootstrap."""

    values = load_phase5a_config(config_path)
    repository = Path(repository_root).resolve()
    selection = load_lightgbm_selection(
        lightgbm_selection_path,
        config_path=config_path,
    )
    if selection.get("git_commit_sha") != str(expected_new_git_sha).lower():
        raise ValueError("LightGBM selection Git SHA differs from new results")
    seed42 = discover_validation_artifacts(seed42_package_root)
    new = discover_validation_artifacts(phase5a_results_root)
    expected_seed42_git = str(values["required_base_git_sha"])
    reference_mae = values["seed42_reference_mae_rounded_6dp"]
    seed42_required = (
        "cell_msca_bidirectional",
        "cell_msca_token_no_attention",
        "lightgbm_raw",
        "lightgbm_log1p",
    )
    for model_name in seed42_required:
        artifact = seed42.get((model_name, 42))
        if artifact is None:
            raise ValueError(f"seed 42 package is missing {model_name}")
        _validate_artifact_contract(
            artifact,
            values,
            expected_git_sha=expected_seed42_git,
            expected_config_sha256=values[
                "seed42_expected_configuration_sha256"
            ][model_name],
        )
        actual_mae = float(
            artifact.result["headline_metrics"]["original_unit"]["mae"]
        )
        if round(actual_mae, 6) != float(reference_mae[model_name]):
            raise ValueError(f"seed 42 reference MAE mismatch for {model_name}")
        if artifact.manifest.get("projection_identity_verified") is not True:
            raise ValueError(
                f"seed 42 artifact did not pass projection identity: {model_name}"
            )

    selected_lightgbm = str(selection["selected_model_name"])
    model_artifacts: dict[str, list[StoredValidationArtifact]] = {
        "bidirectional": [seed42[("cell_msca_bidirectional", 42)]],
        "token_no_attention": [
            seed42[("cell_msca_token_no_attention", 42)]
        ],
        "selected_lightgbm": [],
    }
    convergence_seed42 = new.get((selected_lightgbm, 42))
    if convergence_seed42 is None:
        raise ValueError("Phase 5A results are missing selected LightGBM seed 42")
    model_artifacts["selected_lightgbm"].append(convergence_seed42)
    for seed in NEW_SEEDS:
        required = {
            "bidirectional": ("cell_msca_bidirectional", seed),
            "token_no_attention": ("cell_msca_token_no_attention", seed),
            "selected_lightgbm": (selected_lightgbm, seed),
        }
        for group, key in required.items():
            artifact = new.get(key)
            if artifact is None:
                raise ValueError(f"Phase 5A results are missing {key}")
            model_artifacts[group].append(artifact)

    expected_new_configs: dict[tuple[str, int], str] = {}
    for seed in NEW_SEEDS:
        for variant in ("bidirectional", "token_no_attention"):
            expected_new_configs[(f"cell_msca_{variant}", seed)] = (
                _expected_cell_config_hash(
                    values,
                    repository_root=repository,
                    variant=variant,
                    seed=seed,
                    device="cuda",
                )
            )
        expected_new_configs[(selected_lightgbm, seed)] = canonical_sha256(
            _lightgbm_config(values, model_name=selected_lightgbm).to_dict(
                train_seed=seed,
                target_scale=float(values["data"]["target_scale"]),
            )
        )
    expected_new_configs[(selected_lightgbm, 42)] = canonical_sha256(
        _lightgbm_config(values, model_name=selected_lightgbm).to_dict(
            train_seed=42,
            target_scale=float(values["data"]["target_scale"]),
        )
    )
    for artifacts in model_artifacts.values():
        for artifact in artifacts:
            if artifact.train_seed == 42 and artifact.model_name.startswith("cell_msca"):
                continue
            _validate_artifact_contract(
                artifact,
                values,
                expected_git_sha=expected_new_git_sha,
                expected_config_sha256=expected_new_configs[
                    (artifact.model_name, artifact.train_seed)
                ],
            )

    all_artifacts = [
        artifact
        for artifacts in model_artifacts.values()
        for artifact in artifacts
    ]
    cell_ids, true_original, _ = _aligned_prediction_arrays(all_artifacts)
    ddof = int(values["aggregation"]["standard_deviation_ddof"])
    summaries = {
        name: _seed_metric_summary(artifacts, ddof=ddof)
        for name, artifacts in model_artifacts.items()
    }
    mean_predictions = {
        name: np.mean(
            np.stack(
                [
                    np.asarray(artifact.predictions["pred_original"], dtype=np.float64)
                    for artifact in artifacts
                ]
            ),
            axis=0,
        )
        for name, artifacts in model_artifacts.items()
    }
    bootstrap = values["paired_cluster_bootstrap"]
    comparisons = {
        "bidirectional_vs_token_no_attention": paired_cell_cluster_mae_difference(
            y_true_original=true_original,
            bidirectional_prediction=mean_predictions["bidirectional"],
            comparator_prediction=mean_predictions["token_no_attention"],
            cell_ids=cell_ids,
            n_boot=int(bootstrap["n_boot"]),
            seed=int(bootstrap["seed"]),
            alpha=float(bootstrap["alpha"]),
            rows_per_cell=int(bootstrap["rows_per_cell"]),
        ),
        "bidirectional_vs_selected_lightgbm": paired_cell_cluster_mae_difference(
            y_true_original=true_original,
            bidirectional_prediction=mean_predictions["bidirectional"],
            comparator_prediction=mean_predictions["selected_lightgbm"],
            cell_ids=cell_ids,
            n_boot=int(bootstrap["n_boot"]),
            seed=int(bootstrap["seed"]),
            alpha=float(bootstrap["alpha"]),
            rows_per_cell=int(bootstrap["rows_per_cell"]),
        ),
    }

    destination = _validate_phase5a_output_path(output_dir, kaggle=False)
    summary_path = destination / "seed_metric_summary.json"
    bootstrap_path = destination / "paired_cell_cluster_bootstrap.json"
    manifest_path = destination / "aggregation_manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite aggregation output: {destination}")
    destination.mkdir(parents=True)
    write_metrics_json(
        summary_path,
        {
            "schema_version": PHASE5A_AGGREGATION_SCHEMA_VERSION,
            "artifact_classification": "validation-only",
            "selected_lightgbm": selected_lightgbm,
            "models": summaries,
        },
    )
    write_metrics_json(
        bootstrap_path,
        {
            "schema_version": PHASE5A_AGGREGATION_SCHEMA_VERSION,
            "artifact_classification": "validation-only",
            "prediction_basis": "rowwise_mean_of_three_frozen_seed_predictions",
            "comparisons": comparisons,
            "limitation": (
                "Cell-cluster resampling conditions on the three seed-mean "
                "predictions and does not incorporate training-seed uncertainty."
            ),
        },
    )
    input_files = [
        {
            "model_name": artifact.model_name,
            "train_seed": artifact.train_seed,
            "git_commit_sha": artifact.git_commit_sha,
            "manifest_sha256": file_sha256(artifact.manifest_path),
            "prediction_sha256": file_sha256(artifact.prediction_path),
        }
        for artifact in all_artifacts
    ]
    write_metrics_json(
        manifest_path,
        {
            "schema_version": PHASE5A_AGGREGATION_SCHEMA_VERSION,
            "artifact_classification": "validation-only",
            "status": "completed",
            "phase5a_config_sha256": file_sha256(config_path),
            "lightgbm_selection_sha256": file_sha256(lightgbm_selection_path),
            "required_hashes": values["required_hashes"],
            "seed42_git_sha": expected_seed42_git,
            "new_results_git_sha": expected_new_git_sha,
            "prediction_alignment_verified": True,
            "prediction_metrics_recalculated": True,
            "seed42_projection_identity_verified": {
                model_name: True for model_name in seed42_required
            },
            "complete_cell_rows": int(bootstrap["rows_per_cell"]),
            "test_subset_materialized": False,
            "test_evaluation_performed": False,
            "input_artifacts": input_files,
            "completed_at_utc": _utc_now(),
        },
    )
    return destination


@contextmanager
def _verified_from_arguments(
    arguments: argparse.Namespace,
    values: Mapping[str, Any],
    *,
    train_seed: int,
) -> Iterator[VerifiedV1Archive]:
    repository = Path(arguments.repository_root).resolve()
    manifest_path = repository / str(values["data"]["archive_manifest"])
    with verified_v1_input(
        input_root=arguments.input_root,
        working_root=arguments.working_root,
        manifest_path=manifest_path,
        repository_root=repository,
        kaggle=arguments.kaggle,
        train_seed=train_seed,
    ) as verified:
        yield verified


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--working-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--kaggle", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5A validation robustness runner")
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    integrity = commands.add_parser("integrity")
    convergence = commands.add_parser("lightgbm-convergence")
    repeat = commands.add_parser("lightgbm-repeat")
    cell = commands.add_parser("cell-repeat")
    for command in (integrity, convergence, repeat, cell):
        _add_input_arguments(command)
    integrity.add_argument("--train-seed", type=int, default=42)
    convergence.add_argument("--model", choices=LIGHTGBM_CANDIDATES, required=True)
    convergence.add_argument("--output-root", required=True, type=Path)
    convergence.add_argument("--expected-git-sha", required=True)
    convergence.add_argument("--allow-full-validation", action="store_true")
    repeat.add_argument("--seed", choices=NEW_SEEDS, required=True, type=int)
    repeat.add_argument("--selection", required=True, type=Path)
    repeat.add_argument("--output-root", required=True, type=Path)
    repeat.add_argument("--expected-git-sha", required=True)
    repeat.add_argument("--allow-full-validation", action="store_true")
    cell.add_argument("--seed", choices=NEW_SEEDS, required=True, type=int)
    cell.add_argument(
        "--variant",
        choices=("bidirectional", "token_no_attention"),
        required=True,
    )
    cell.add_argument("--device", choices=("cpu", "cuda"), required=True)
    cell.add_argument("--output-root", required=True, type=Path)
    cell.add_argument("--expected-git-sha", required=True)
    cell.add_argument("--allow-full-validation", action="store_true")

    freeze = commands.add_parser("freeze-lightgbm-selection")
    freeze.add_argument("--raw-run-dir", required=True, type=Path)
    freeze.add_argument("--log1p-run-dir", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--expected-git-sha", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--seed42-package-root", required=True, type=Path)
    aggregate.add_argument("--phase5a-results-root", required=True, type=Path)
    aggregate.add_argument("--selection", required=True, type=Path)
    aggregate.add_argument("--repository-root", required=True, type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.add_argument("--expected-new-git-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    values = load_phase5a_config(arguments.config)
    if arguments.command == "freeze-lightgbm-selection":
        output = freeze_lightgbm_convergence_selection(
            raw_run_dir=arguments.raw_run_dir,
            log1p_run_dir=arguments.log1p_run_dir,
            config_path=arguments.config,
            output_path=arguments.output,
            expected_git_sha=arguments.expected_git_sha,
        )
        print(json.dumps({"selection": str(output)}, indent=2))
        return 0
    if arguments.command == "aggregate":
        output = aggregate_phase5a_results(
            seed42_package_root=arguments.seed42_package_root,
            phase5a_results_root=arguments.phase5a_results_root,
            lightgbm_selection_path=arguments.selection,
            config_path=arguments.config,
            repository_root=arguments.repository_root,
            output_dir=arguments.output_dir,
            expected_new_git_sha=arguments.expected_new_git_sha,
        )
        print(json.dumps({"aggregation_output": str(output)}, indent=2))
        return 0

    seed = int(getattr(arguments, "train_seed", getattr(arguments, "seed", 42)))
    with _verified_from_arguments(arguments, values, train_seed=seed) as verified:
        if arguments.command == "integrity":
            print(json.dumps(verified.summary, indent=2, allow_nan=False))
            return 0
        if arguments.command == "lightgbm-convergence":
            output = run_phase5a_lightgbm(
                verified,
                config_path=arguments.config,
                output_root=arguments.output_root,
                repository_root=arguments.repository_root,
                expected_git_sha=arguments.expected_git_sha,
                model_name=arguments.model,
                train_seed=42,
                run_type="convergence",
                allow_full_validation=arguments.allow_full_validation,
            )
        elif arguments.command == "lightgbm-repeat":
            selection = load_lightgbm_selection(
                arguments.selection,
                config_path=arguments.config,
            )
            model_name = str(selection["selected_model_name"])
            output = run_phase5a_lightgbm(
                verified,
                config_path=arguments.config,
                output_root=arguments.output_root,
                repository_root=arguments.repository_root,
                expected_git_sha=arguments.expected_git_sha,
                model_name=model_name,
                train_seed=arguments.seed,
                run_type="repeat",
                selection_path=arguments.selection,
                allow_full_validation=arguments.allow_full_validation,
            )
        else:
            runtime_path = Path(arguments.repository_root) / str(
                values["cell_runtime_configs"][str(arguments.seed)]
            )
            output = run_phase5a_cell_repeat(
                verified,
                config_path=arguments.config,
                runtime_config_path=runtime_path,
                output_root=arguments.output_root,
                repository_root=arguments.repository_root,
                expected_git_sha=arguments.expected_git_sha,
                variant=arguments.variant,
                train_seed=arguments.seed,
                device=arguments.device,
                allow_full_validation=arguments.allow_full_validation,
            )
        print(json.dumps({"output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
