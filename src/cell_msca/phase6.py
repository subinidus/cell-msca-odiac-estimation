"""Locked, one-pass final-test evaluation for validation-frozen models.

The module contains no training or validation-selection entry point.  It first
verifies the two immutable validation packages, every frozen artifact, the
project-data provenance, the output destination, and an explicit command-line
authorization.  Only a successful gate may materialize the persistent test
split, exactly once in one runner invocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from . import baselines as _baseline_module
from .baselines import (
    BaselineArraySplit,
    BaselineProvenance,
    FittedLightGBM,
    ModelContract,
    TestEvaluationAuthorization,
    TestEvaluationBlockedError,
    evaluate_fitted_baseline,
)
from .data import file_sha256
from .evaluate import (
    calculate_prediction_metrics_from_csv,
    read_prediction_csv,
    verify_baseline_result_from_prediction_csv,
    write_metrics_json,
    write_prediction_csv,
    write_prediction_support_diagnostic_csv,
)
from .metrics import spearman_correlation
from .phase5a import paired_cell_cluster_mae_difference
from .target import NONNEGATIVE_PREDICTION_SUPPORT_POLICY
from .v1_validation import verified_v1_input

PHASE6_CONFIG_SCHEMA_VERSION = "cell_msca.phase6_locked_test_protocol.v1"
PHASE6_GATE_SCHEMA_VERSION = "cell_msca.phase6_final_test_gate_audit.v1"
PHASE6_METRICS_SCHEMA_VERSION = "cell_msca.phase6_test_metrics.v1"
PHASE6_SUMMARY_SCHEMA_VERSION = "cell_msca.phase6_test_seed_summary.v1"
PHASE6_BOOTSTRAP_SCHEMA_VERSION = "cell_msca.phase6_test_bootstrap.v1"
PHASE6_MANIFEST_SCHEMA_VERSION = "cell_msca.phase6_test_manifest.v1"
PHASE6_VALIDATION_CODE_GIT_SHA = "35696c7e8c251e47e12c095f4e9e648a43f97a42"

PHASE6_MODELS = (
    "lightgbm_raw",
    "cell_msca_token_no_attention",
    "cell_msca_bidirectional",
)
PHASE6_SEEDS = (42, 43, 44)
PHASE6_ARTIFACT_KEYS = frozenset(
    (model_name, seed) for model_name in PHASE6_MODELS for seed in PHASE6_SEEDS
)
_SHA256_LENGTH = 64
_GIT_SHA_LENGTH = 40
_PHASE6_GATE_AUTHORITY = object()


@dataclass(frozen=True)
class FrozenArtifactSpec:
    model_name: str
    train_seed: int
    source_package: str
    artifact_directory: str
    manifest_path: str
    manifest_sha256: str
    validation_prediction_path: str
    validation_prediction_sha256: str
    model_artifact_path: str
    model_artifact_sha256: str
    model_artifact_type: str
    configuration_sha256: str
    git_commit_sha: str

    @property
    def key(self) -> tuple[str, int]:
        return self.model_name, self.train_seed


@dataclass(frozen=True)
class LoadedFrozenModel:
    spec: FrozenArtifactSpec
    model: Any
    validation_result: Mapping[str, Any]


@dataclass(frozen=True)
class FinalTestGate:
    config: Mapping[str, Any]
    config_sha256: str
    loaded_models: Mapping[tuple[str, int], LoadedFrozenModel]
    audit: Mapping[str, Any]
    _authority: object = field(repr=False, compare=False)


ModelLoader = Callable[
    [FrozenArtifactSpec, Path, Mapping[str, Any], str], Any
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return values


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validate_git_sha(value: Any, *, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != _GIT_SHA_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a 40-character Git SHA")
    return digest


def _artifact_spec(values: Mapping[str, Any], *, index: int) -> FrozenArtifactSpec:
    required = {
        "model_name",
        "train_seed",
        "source_package",
        "artifact_directory",
        "manifest_path",
        "manifest_sha256",
        "validation_prediction_path",
        "validation_prediction_sha256",
        "model_artifact_path",
        "model_artifact_sha256",
        "model_artifact_type",
        "configuration_sha256",
        "git_commit_sha",
    }
    if set(values) != required:
        raise ValueError(
            f"frozen artifact {index} fields differ: "
            f"missing={sorted(required - set(values))}, "
            f"unexpected={sorted(set(values) - required)}"
        )
    model_name = str(values["model_name"])
    train_seed = int(values["train_seed"])
    source_package = str(values["source_package"])
    artifact_type = str(values["model_artifact_type"])
    if model_name not in PHASE6_MODELS or train_seed not in PHASE6_SEEDS:
        raise ValueError(f"unplanned final-test artifact: {(model_name, train_seed)}")
    expected_package = (
        "seed42" if model_name.startswith("cell_msca_") and train_seed == 42
        else "phase5a"
    )
    if source_package != expected_package:
        raise ValueError(f"wrong source package for {(model_name, train_seed)}")
    expected_type = "lightgbm_text" if model_name == "lightgbm_raw" else "torch_weights"
    if artifact_type != expected_type:
        raise ValueError(f"wrong model artifact type for {(model_name, train_seed)}")
    paths = {
        name: str(values[name])
        for name in (
            "artifact_directory",
            "manifest_path",
            "validation_prediction_path",
            "model_artifact_path",
        )
    }
    for name, path in paths.items():
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe frozen artifact path {name}: {path!r}")
    directory = paths["artifact_directory"].rstrip("/")
    if any(
        not (path == directory or path.startswith(directory + "/"))
        for name, path in paths.items()
        if name != "artifact_directory"
    ):
        raise ValueError("artifact member is outside its frozen artifact directory")
    return FrozenArtifactSpec(
        model_name=model_name,
        train_seed=train_seed,
        source_package=source_package,
        artifact_directory=directory,
        manifest_path=paths["manifest_path"],
        manifest_sha256=_validate_sha256(
            values["manifest_sha256"], name=f"artifacts[{index}].manifest_sha256"
        ),
        validation_prediction_path=paths["validation_prediction_path"],
        validation_prediction_sha256=_validate_sha256(
            values["validation_prediction_sha256"],
            name=f"artifacts[{index}].validation_prediction_sha256",
        ),
        model_artifact_path=paths["model_artifact_path"],
        model_artifact_sha256=_validate_sha256(
            values["model_artifact_sha256"],
            name=f"artifacts[{index}].model_artifact_sha256",
        ),
        model_artifact_type=artifact_type,
        configuration_sha256=_validate_sha256(
            values["configuration_sha256"],
            name=f"artifacts[{index}].configuration_sha256",
        ),
        git_commit_sha=_validate_git_sha(
            values["git_commit_sha"], name=f"artifacts[{index}].git_commit_sha"
        ),
    )


def load_phase6_protocol(path: str | Path) -> dict[str, Any]:
    """Load the frozen protocol and reject any expanded test scope."""

    values = _read_json(path)
    if values.get("schema_version") != PHASE6_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 6 protocol schema_version")
    if values.get("stage") != "locked_final_test":
        raise ValueError("Phase 6 stage must be locked_final_test")
    if tuple(values.get("models", ())) != PHASE6_MODELS:
        raise ValueError("Phase 6 model set or order changed")
    if tuple(values.get("train_seeds", ())) != PHASE6_SEEDS:
        raise ValueError("Phase 6 train seeds must be [42, 43, 44]")
    if int(values.get("split_seed", -1)) != 42:
        raise ValueError("Phase 6 split_seed must remain 42")
    validation_samples = values.get("validation_evidence_expected_samples")
    if isinstance(validation_samples, bool) or not isinstance(validation_samples, int):
        raise ValueError("validation_evidence_expected_samples must be a positive integer")
    if validation_samples <= 0:
        raise ValueError("validation_evidence_expected_samples must be a positive integer")
    if (
        values.get("prediction_support_policy")
        != NONNEGATIVE_PREDICTION_SUPPORT_POLICY
    ):
        raise ValueError("Phase 6 prediction support policy changed")
    validation_git_sha = _validate_git_sha(
        values.get("validation_code_git_sha"), name="validation_code_git_sha"
    )
    if validation_git_sha != PHASE6_VALIDATION_CODE_GIT_SHA:
        raise ValueError("Phase 6 validation code Git SHA changed")
    if values.get("data") != {
        "data_version": "v1_legacy",
        "archive_manifest": "configs/v1_legacy_archive_manifest.json",
        "target_definition": "v1_legacy ODIAC-derived zonal mean",
        "target_scale": 1.0,
    }:
        raise ValueError("Phase 6 data contract changed")

    required_hashes = values.get("required_hashes")
    if not isinstance(required_hashes, Mapping) or set(required_hashes) != {
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    }:
        raise ValueError("Phase 6 required_hashes are incomplete")
    for name, digest in required_hashes.items():
        _validate_sha256(digest, name=f"required_hashes.{name}")

    packages = values.get("validation_packages")
    if not isinstance(packages, Mapping) or set(packages) != {"phase5a", "seed42"}:
        raise ValueError("Phase 6 requires exactly the Phase 5A and seed-42 packages")
    for package_name, package in packages.items():
        if not isinstance(package, Mapping) or set(package) != {
            "file_name",
            "file_sha256",
        }:
            raise ValueError(f"invalid validation package contract: {package_name}")
        _validate_sha256(
            package["file_sha256"], name=f"validation_packages.{package_name}"
        )

    evidence = values.get("phase5a_evidence")
    required_evidence = {
        "progress_path",
        "progress_sha256",
        "aggregation_manifest_path",
        "aggregation_manifest_sha256",
        "selection_path",
        "selection_sha256",
        "phase5a_config_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_evidence:
        raise ValueError("Phase 5A evidence contract is incomplete")
    for name in (
        "progress_sha256",
        "aggregation_manifest_sha256",
        "selection_sha256",
        "phase5a_config_sha256",
    ):
        _validate_sha256(evidence[name], name=f"phase5a_evidence.{name}")

    selection = values.get("lightgbm_selection")
    expected_parameters = {
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "early_stopping_rounds": 200,
    }
    if not isinstance(selection, Mapping):
        raise ValueError("Phase 6 LightGBM selection is missing")
    if selection.get("selected_model_name") != "lightgbm_raw":
        raise ValueError("Phase 6 selected LightGBM must remain lightgbm_raw")
    if selection.get("parameters") != expected_parameters:
        raise ValueError("Phase 6 LightGBM parameters changed")
    if selection.get("selection_metric") != "validation_original_unit_mae":
        raise ValueError("Phase 6 LightGBM selection metric changed")

    metric_contract = values.get("metrics")
    if metric_contract != {
        "primary_original_unit": ["mae", "rmse", "r2"],
        "secondary": ["bias", "spearman", "log_unit_mae", "log_unit_rmse", "log_unit_r2"],
        "recalculation_source": "saved_test_prediction_csv_only",
        "seed_summary_standard_deviation_ddof": 1,
    }:
        raise ValueError("Phase 6 metric contract changed")
    bootstrap = values.get("paired_cell_cluster_bootstrap")
    if bootstrap != {
        "n_boot": 10000,
        "seed": 3407,
        "alpha": 0.05,
        "rows_per_cell": 36,
        "prediction_basis": "rowwise_mean_of_three_frozen_seed_predictions",
        "difference_orientation": "mae_bidirectional_minus_mae_comparator",
        "comparators": ["cell_msca_token_no_attention", "lightgbm_raw"],
        "training_seed_uncertainty_included": False,
    }:
        raise ValueError("Phase 6 bootstrap contract changed")
    test_contract = values.get("test_contract")
    if test_contract != {
        "required_cli_flag": "--allow-final-test",
        "maximum_materializations_per_invocation": 1,
        "expected_unique_cells": 2574,
        "expected_rows_per_cell": 36,
        "expected_samples": 92664,
        "output_overwrite": False,
        "training_performed": False,
        "model_selection_performed": False,
        "hyperparameter_change_performed": False,
    }:
        raise ValueError("Phase 6 final-test access contract changed")

    prohibited = set(values.get("prohibited_operations", ()))
    if prohibited != {
        "training",
        "early_stopping",
        "checkpoint_reselection",
        "hyperparameter_change",
        "test_based_model_selection",
        "prediction_policy_change",
        "split_regeneration",
        "target_change",
        "feature_change",
        "architecture_or_attention_change",
    }:
        raise ValueError("Phase 6 prohibited-operation contract changed")
    artifacts = values.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 9:
        raise ValueError("Phase 6 requires exactly nine frozen model artifacts")
    specs = [_artifact_spec(item, index=index) for index, item in enumerate(artifacts)]
    keys = [spec.key for spec in specs]
    if len(keys) != len(set(keys)) or set(keys) != PHASE6_ARTIFACT_KEYS:
        raise ValueError("Phase 6 frozen artifact keys are missing or duplicated")
    return values


def frozen_artifact_specs(config: Mapping[str, Any]) -> dict[tuple[str, int], FrozenArtifactSpec]:
    specs = {
        spec.key: spec
        for index, item in enumerate(config["artifacts"])
        for spec in (_artifact_spec(item, index=index),)
    }
    if set(specs) != PHASE6_ARTIFACT_KEYS:
        raise ValueError("Phase 6 frozen artifact mapping is incomplete")
    return specs


def _validate_zip_inventory(archive: zipfile.ZipFile, *, package_name: str) -> set[str]:
    names: list[str] = []
    for entry in archive.infolist():
        relative = PurePosixPath(entry.filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe ZIP member in {package_name}: {entry.filename!r}")
        mode = (entry.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError(f"symbolic-link ZIP member in {package_name}: {entry.filename!r}")
        if not entry.is_dir():
            names.append(relative.as_posix())
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate ZIP members in {package_name}")
    return set(names)


def _zip_bytes(
    archive: zipfile.ZipFile,
    inventory: set[str],
    path: str,
    *,
    expected_sha256: str | None = None,
) -> bytes:
    if path not in inventory:
        raise ValueError(f"frozen ZIP member is missing: {path}")
    values = archive.read(path)
    if expected_sha256 is not None and _sha256_bytes(values) != expected_sha256:
        raise ValueError(f"frozen ZIP member SHA-256 mismatch: {path}")
    return values


def _zip_json(
    archive: zipfile.ZipFile,
    inventory: set[str],
    path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        values = json.loads(
            _zip_bytes(
                archive,
                inventory,
                path,
                expected_sha256=expected_sha256,
            ).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in frozen ZIP member: {path}") from error
    if not isinstance(values, dict):
        raise ValueError(f"frozen JSON member must be an object: {path}")
    return values


def _extract_verified_member(
    archive: zipfile.ZipFile,
    inventory: set[str],
    spec: FrozenArtifactSpec,
    extraction_root: Path,
) -> Path:
    suffix = ".txt" if spec.model_artifact_type == "lightgbm_text" else ".pt"
    destination = extraction_root / f"{spec.model_name}-seed{spec.train_seed}{suffix}"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite extracted model: {destination}")
    values = _zip_bytes(
        archive,
        inventory,
        spec.model_artifact_path,
        expected_sha256=spec.model_artifact_sha256,
    )
    destination.write_bytes(values)
    if file_sha256(destination) != spec.model_artifact_sha256:
        raise ValueError("extracted model artifact SHA-256 mismatch")
    return destination


def _model_contract_from_result(result: Mapping[str, Any]) -> ModelContract:
    return ModelContract(
        model_name=str(result["model_name"]),
        config_sha256=str(result["config_sha256"]),
        target_transform=str(result["target_transform"]),
        target_scale=float(result["target_scale"]),
        loss_objective=str(result["loss_objective"]),
        inverse_mode=str(result["inverse_mode"]),
        smearing_factor=(
            None if result.get("smearing_factor") is None else float(result["smearing_factor"])
        ),
        best_iteration=(
            None if result.get("best_iteration") is None else int(result["best_iteration"])
        ),
        best_epoch=None if result.get("best_epoch") is None else int(result["best_epoch"]),
    )


def _load_frozen_model(
    spec: FrozenArtifactSpec,
    artifact_path: Path,
    validation_result: Mapping[str, Any],
    device: str,
) -> Any:
    expected_hashes = {
        "data_sha256": str(validation_result["data_sha256"]),
        "split_sha256": str(validation_result["split_sha256"]),
        "split_config_sha256": str(validation_result["split_config_sha256"]),
        "preprocessing_sha256": str(validation_result["preprocessing_sha256"]),
        "configuration_sha256": spec.configuration_sha256,
        "train_seed": spec.train_seed,
        "git_commit_sha": spec.git_commit_sha,
    }
    if spec.model_artifact_type == "lightgbm_text":
        try:
            lightgbm = importlib.import_module("lightgbm")
        except (ImportError, OSError) as error:
            raise ImportError("Phase 6 LightGBM model loading requires lightgbm>=4.0") from error
        booster = lightgbm.Booster(model_file=str(artifact_path))
        if int(booster.num_feature()) != 7:
            raise ValueError("frozen LightGBM model must have exactly seven features")
        expected_iteration = int(validation_result["best_iteration"])
        if int(booster.current_iteration()) != expected_iteration:
            raise ValueError("frozen LightGBM model iteration count differs from validation")
        return FittedLightGBM(
            estimator=booster,
            target_space="original",
            contract=_model_contract_from_result(validation_result),
        )

    train_module = importlib.import_module("cell_msca.train")
    checkpoint = train_module.load_selected_checkpoint(
        artifact_path,
        device=device,
        expected_hashes=expected_hashes,
    )
    if checkpoint.contract != _model_contract_from_result(validation_result):
        raise ValueError("frozen neural checkpoint contract differs from validation result")
    return train_module.FittedCellMSCA(
        model=checkpoint.model,
        contract=checkpoint.contract,
        parameter_count=checkpoint.parameter_count,
        device=str(next(checkpoint.model.parameters()).device),
        checkpoint_path=artifact_path,
        checkpoint_provenance=checkpoint.provenance,
    )


def _validation_result(
    archive: zipfile.ZipFile,
    inventory: set[str],
    spec: FrozenArtifactSpec,
    manifest: Mapping[str, Any],
    *,
    expected_samples: int,
) -> Mapping[str, Any]:
    result = manifest.get("validation_metrics")
    if not isinstance(result, Mapping):
        raise ValueError(f"validation metrics missing from {spec.manifest_path}")
    if result.get("model_name") != spec.model_name or int(result.get("train_seed", -1)) != (
        spec.train_seed
    ):
        raise ValueError("frozen validation result model/seed mismatch")
    if result.get("evaluation_split") != "validation":
        raise ValueError("frozen model selection evidence must be validation-only")
    if result.get("config_sha256") != spec.configuration_sha256:
        raise ValueError("frozen validation result configuration SHA-256 mismatch")
    for name, expected in {
        "data_sha256": manifest["data_sha256"],
        "split_sha256": manifest["split_sha256"],
        "split_config_sha256": manifest["split_config_sha256"],
        "preprocessing_sha256": manifest["preprocessing_sha256"],
        "split_seed": 42,
    }.items():
        if result.get(name) != expected:
            raise ValueError(f"frozen validation result {name} mismatch")
    support_policy = result.get("prediction_support_policy")
    if support_policy not in (None, NONNEGATIVE_PREDICTION_SUPPORT_POLICY):
        raise ValueError("frozen validation result prediction support policy changed")
    if support_policy is None and spec.source_package != "seed42":
        raise ValueError("new validation result is missing prediction support policy")
    # Reading and hashing the immutable validation prediction is part of the
    # identity gate.  The test metrics are never combined with these values.
    prediction_bytes = _zip_bytes(
        archive,
        inventory,
        spec.validation_prediction_path,
        expected_sha256=spec.validation_prediction_sha256,
    )
    try:
        rows = csv.DictReader(prediction_bytes.decode("utf-8").splitlines())
        minimum = float("inf")
        count = 0
        for row in rows:
            prediction = float(row["pred_original"])
            if not np.isfinite(prediction) or prediction < 0.0:
                raise ValueError("frozen validation pred_original violates its support policy")
            minimum = min(minimum, prediction)
            count += 1
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid frozen validation prediction CSV: {spec.validation_prediction_path}"
        ) from error
    if count != expected_samples or not np.isfinite(minimum):
        raise ValueError("frozen validation prediction row count changed")
    return result


def _validate_manifest(
    manifest: Mapping[str, Any],
    spec: FrozenArtifactSpec,
    config: Mapping[str, Any],
) -> None:
    required_hashes = config["required_hashes"]
    if manifest.get("status") != "completed" or manifest.get("stage") != "validation_only":
        raise ValueError(f"frozen validation run is not completed: {spec.manifest_path}")
    if manifest.get("model_name") != spec.model_name or int(
        manifest.get("train_seed", -1)
    ) != spec.train_seed:
        raise ValueError("frozen run manifest model/seed mismatch")
    if int(manifest.get("split_seed", -1)) != 42:
        raise ValueError("frozen run manifest split_seed mismatch")
    if manifest.get("git_commit_sha") != spec.git_commit_sha:
        raise ValueError("frozen run manifest Git SHA mismatch")
    if manifest.get("configuration_sha256") != spec.configuration_sha256:
        raise ValueError("frozen run manifest configuration SHA-256 mismatch")
    for name, expected in required_hashes.items():
        if manifest.get(name) != expected:
            raise ValueError(f"frozen run manifest {name} mismatch")
    if manifest.get("test_subset_materialized") is not False:
        raise ValueError("validation artifact materialized the test subset")
    manifest_support = manifest.get("prediction_support_policy")
    if manifest_support not in (None, NONNEGATIVE_PREDICTION_SUPPORT_POLICY):
        raise ValueError("frozen run manifest prediction support policy changed")
    if manifest_support is None and spec.source_package != "seed42":
        raise ValueError("new validation manifest is missing prediction support policy")
    if spec.model_artifact_type == "lightgbm_text":
        if manifest.get("model_sha256") != spec.model_artifact_sha256:
            raise ValueError("frozen LightGBM manifest model SHA-256 mismatch")
        if manifest.get("model_artifact") != PurePosixPath(
            spec.model_artifact_path
        ).name:
            raise ValueError("frozen LightGBM manifest model path mismatch")
    else:
        artifacts = manifest.get("artifacts")
        checkpoint = artifacts.get("selected_checkpoint.pt") if isinstance(
            artifacts, Mapping
        ) else None
        if not isinstance(checkpoint, Mapping) or checkpoint.get("path") != (
            PurePosixPath(spec.model_artifact_path).name
        ):
            raise ValueError("frozen neural manifest checkpoint path mismatch")
    test_evaluated = manifest.get("test_evaluation_performed", "missing")
    if test_evaluated is not False:
        legacy_missing = (
            test_evaluated == "missing"
            and spec.source_package == "seed42"
            and spec.train_seed == 42
        )
        if not legacy_missing:
            raise ValueError("validation artifact test-evaluation state is not false")


def _verify_phase5a_evidence(
    archive: zipfile.ZipFile,
    inventory: set[str],
    config: Mapping[str, Any],
    specs: Mapping[tuple[str, int], FrozenArtifactSpec],
) -> dict[str, Any]:
    evidence = config["phase5a_evidence"]
    progress = _zip_json(
        archive,
        inventory,
        str(evidence["progress_path"]),
        expected_sha256=str(evidence["progress_sha256"]),
    )
    stages = progress.get("stages")
    if progress.get("status") != "completed" or not isinstance(stages, Mapping):
        raise ValueError("Phase 5A progress is not completed")
    failed_stages = sorted(
        str(name) for name, stage in stages.items()
        if not isinstance(stage, Mapping) or stage.get("status") != "completed"
    )
    if failed_stages:
        raise ValueError(f"Phase 5A has incomplete or failed stages: {failed_stages}")
    if progress.get("test_subset_materialized") is not False or progress.get(
        "test_evaluation_performed"
    ) is not False:
        raise ValueError("Phase 5A progress does not keep the test gate closed")

    aggregation = _zip_json(
        archive,
        inventory,
        str(evidence["aggregation_manifest_path"]),
        expected_sha256=str(evidence["aggregation_manifest_sha256"]),
    )
    if aggregation.get("status") != "completed":
        raise ValueError("Phase 5A aggregation is not completed")
    if aggregation.get("phase5a_config_sha256") != evidence["phase5a_config_sha256"]:
        raise ValueError("Phase 5A config SHA-256 changed")
    if aggregation.get("lightgbm_selection_sha256") != evidence["selection_sha256"]:
        raise ValueError("Phase 5A selection SHA-256 changed")
    if aggregation.get("required_hashes") != config["required_hashes"]:
        raise ValueError("Phase 5A aggregation provenance changed")
    if aggregation.get("test_subset_materialized") is not False or aggregation.get(
        "test_evaluation_performed"
    ) is not False:
        raise ValueError("Phase 5A aggregation opened the test gate")

    aggregation_keys: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in aggregation.get("input_artifacts", ()):
        if not isinstance(row, Mapping):
            raise ValueError("invalid Phase 5A input artifact identity")
        key = str(row.get("model_name")), int(row.get("train_seed", -1))
        if key in aggregation_keys:
            raise ValueError(f"duplicate Phase 5A aggregation input: {key}")
        aggregation_keys[key] = row
    if set(aggregation_keys) != PHASE6_ARTIFACT_KEYS:
        raise ValueError("Phase 5A aggregation input set differs from the nine frozen runs")
    for key, spec in specs.items():
        row = aggregation_keys[key]
        if row.get("git_commit_sha") != spec.git_commit_sha:
            raise ValueError(f"Phase 5A aggregation Git identity mismatch: {key}")
        if row.get("manifest_sha256") != spec.manifest_sha256:
            raise ValueError(f"Phase 5A aggregation manifest identity mismatch: {key}")
        if row.get("prediction_sha256") != spec.validation_prediction_sha256:
            raise ValueError(f"Phase 5A aggregation prediction identity mismatch: {key}")

    selection = _zip_json(
        archive,
        inventory,
        str(evidence["selection_path"]),
        expected_sha256=str(evidence["selection_sha256"]),
    )
    if selection.get("selected_model_name") != "lightgbm_raw":
        raise ValueError("frozen Phase 5A LightGBM winner changed")
    if selection.get("prediction_support_policy") != NONNEGATIVE_PREDICTION_SUPPORT_POLICY:
        raise ValueError("frozen Phase 5A selection support policy changed")
    selected_config = config["lightgbm_selection"]
    if selection.get("selection_metric") != selected_config["selection_metric"]:
        raise ValueError("frozen LightGBM selection metric changed")
    if selection.get("selected_parameters") != selected_config["parameters"]:
        raise ValueError("frozen LightGBM parameters changed")
    candidates = selection.get("candidate_results")
    tie_break = selection.get("tie_break_order")
    if not isinstance(candidates, Mapping) or set(candidates) != {
        "lightgbm_raw",
        "lightgbm_log1p",
    }:
        raise ValueError("frozen LightGBM candidates changed")
    if tie_break != ["lightgbm_raw", "lightgbm_log1p"]:
        raise ValueError("frozen LightGBM tie-break order changed")
    maes: dict[str, float] = {}
    for model_name in tie_break:
        row = candidates[model_name]
        if not isinstance(row, Mapping):
            raise ValueError("invalid frozen LightGBM candidate row")
        try:
            mae = float(row["validation_original_unit_mae"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid frozen LightGBM candidate MAE") from error
        if not np.isfinite(mae):
            raise ValueError("frozen LightGBM candidate MAE must be finite")
        maes[model_name] = mae
    recomputed_winner = min(tie_break, key=lambda name: maes[name])
    if recomputed_winner != selection["selected_model_name"]:
        raise ValueError("frozen LightGBM winner does not match candidate MAE")
    return {
        "progress_status": "completed",
        "completed_stage_count": len(stages),
        "failed_stage_count": 0,
        "aggregation_status": "completed",
        "aggregation_input_artifact_count": len(aggregation_keys),
        "selection_revalidated": True,
        "selection_winner_recomputed_from_validation_mae": recomputed_winner,
        "validation_test_subset_materialized": False,
        "validation_test_evaluation_performed": False,
    }


def _validate_protocol_provenance(protocol: Any, config: Mapping[str, Any]) -> None:
    provenance = protocol.provenance
    for name, expected in config["required_hashes"].items():
        if getattr(provenance, name) != expected:
            raise ValueError(f"verified project data {name} mismatch")
    if provenance.split_seed != 42:
        raise ValueError("verified project data split seed mismatch")


def verify_final_test_gate(
    *,
    config_path: str | Path,
    phase5a_zip: str | Path,
    seed42_zip: str | Path,
    output_dir: str | Path,
    extraction_root: str | Path,
    allow_final_test: bool,
    protocol: Any | None = None,
    device: str = "cpu",
    model_loader: ModelLoader | None = None,
    repository_root: str | Path | None = None,
    kaggle: bool = False,
) -> FinalTestGate:
    """Verify every frozen condition without materializing the test subset."""

    if not allow_final_test:
        raise TestEvaluationBlockedError(
            "locked final-test evaluation requires explicit --allow-final-test"
        )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6 output: {destination}")
    sibling_outputs = (
        destination.parent / f"{destination.name}.zip",
        destination.parent / f"{destination.name}.sha256.json",
        destination.parent / f"{destination.name}.zip.tmp",
    )
    if any(path.exists() for path in sibling_outputs):
        raise FileExistsError("refusing to overwrite a Phase 6 bundle, receipt, or temporary ZIP")
    if kaggle:
        output_posix = PurePosixPath(str(destination).replace("\\", "/"))
        try:
            output_posix.relative_to(PurePosixPath("/kaggle/working"))
        except ValueError as error:
            raise ValueError("Kaggle Phase 6 output must be below /kaggle/working") from error
    config = load_phase6_protocol(config_path)
    config_sha256 = file_sha256(config_path)
    runner_git_sha: str | None = None
    if repository_root is not None:
        repository = Path(repository_root).resolve()
        runner_git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        _validate_git_sha(runner_git_sha, name="runner_git_sha")
        ancestor = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(config["validation_code_git_sha"]),
                runner_git_sha,
            ],
            cwd=repository,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError("runner source does not descend from the frozen validation code")
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty.strip():
            raise RuntimeError("Phase 6 runner requires a clean tracked-and-untracked source tree")
    if protocol is not None:
        _validate_protocol_provenance(protocol, config)
    package_paths = {
        "phase5a": Path(phase5a_zip),
        "seed42": Path(seed42_zip),
    }
    for package_name, path in package_paths.items():
        expected = config["validation_packages"][package_name]
        if path.name != expected["file_name"]:
            raise ValueError(f"{package_name} validation package file name mismatch")
        if file_sha256(path) != expected["file_sha256"]:
            raise ValueError(f"{package_name} validation package SHA-256 mismatch")

    extraction = Path(extraction_root)
    extraction.mkdir(parents=True, exist_ok=True)
    if any(extraction.iterdir()):
        raise FileExistsError("model extraction_root must be empty")
    specs = frozen_artifact_specs(config)
    archives: dict[str, zipfile.ZipFile] = {}
    inventories: dict[str, set[str]] = {}
    loaded: dict[tuple[str, int], LoadedFrozenModel] = {}
    package_audit: dict[str, Any] = {}
    loader = model_loader or _load_frozen_model
    try:
        for package_name, path in package_paths.items():
            archive = zipfile.ZipFile(path, "r")
            archives[package_name] = archive
            inventories[package_name] = _validate_zip_inventory(
                archive, package_name=package_name
            )
            package_audit[package_name] = {
                "file_name": path.name,
                "file_sha256": file_sha256(path),
                "member_count": len(inventories[package_name]),
            }
        evidence_audit = _verify_phase5a_evidence(
            archives["phase5a"], inventories["phase5a"], config, specs
        )
        legacy_missing_test_field: list[str] = []
        for key in sorted(specs):
            spec = specs[key]
            archive = archives[spec.source_package]
            inventory = inventories[spec.source_package]
            manifest = _zip_json(
                archive,
                inventory,
                spec.manifest_path,
                expected_sha256=spec.manifest_sha256,
            )
            _validate_manifest(manifest, spec, config)
            if "test_evaluation_performed" not in manifest:
                legacy_missing_test_field.append(spec.manifest_path)
            validation_result = _validation_result(
                archive,
                inventory,
                spec,
                manifest,
                expected_samples=int(config["validation_evidence_expected_samples"]),
            )
            model_path = _extract_verified_member(
                archive, inventory, spec, extraction
            )
            model = loader(spec, model_path, validation_result, device)
            loaded[key] = LoadedFrozenModel(spec, model, validation_result)
    finally:
        for archive in archives.values():
            archive.close()
    if set(loaded) != PHASE6_ARTIFACT_KEYS:
        raise ValueError("safe loading did not produce exactly nine frozen models")
    audit = {
        "schema_version": PHASE6_GATE_SCHEMA_VERSION,
        "artifact_classification": "final-test-gate",
        "status": "passed",
        "checked_at_utc": _utc_now(),
        "config_sha256": config_sha256,
        "explicit_allow_final_test": True,
        "output_path_was_absent": True,
        "output_sibling_bundle_paths_were_absent": True,
        "runner_git_sha": runner_git_sha,
        "validation_code_git_sha_is_ancestor": (
            True if repository_root is not None else "not_checked"
        ),
        "git_dirty_state_policy": "tracked_and_untracked_files",
        "packages": package_audit,
        "phase5a_evidence": evidence_audit,
        "expected_artifact_count": 9,
        "loaded_artifact_count": len(loaded),
        "loaded_artifact_keys": [
            {"model_name": model_name, "train_seed": seed}
            for model_name, seed in sorted(loaded)
        ],
        "all_model_artifact_sha256_verified": True,
        "all_models_safely_loaded_before_test_materialization": True,
        "legacy_seed42_missing_test_evaluation_field_paths": legacy_missing_test_field,
        "legacy_seed42_missing_field_policy": (
            "accepted only for the exact immutable seed-42 neural manifests; "
            "Phase 5A aggregation and package gate record effective false"
        ),
        "test_subset_materialized": False,
        "test_evaluation_performed": False,
        "training_performed": False,
        "model_selection_performed": False,
        "hyperparameter_change_performed": False,
    }
    return FinalTestGate(
        config=config,
        config_sha256=config_sha256,
        loaded_models=loaded,
        audit=audit,
        _authority=_PHASE6_GATE_AUTHORITY,
    )


def _materialize_test_once(protocol: Any, gate: FinalTestGate) -> BaselineArraySplit:
    if gate._authority is not _PHASE6_GATE_AUTHORITY:
        raise TestEvaluationBlockedError("invalid Phase 6 final-test gate authority")
    authorization = TestEvaluationAuthorization(
        data_sha256=protocol.provenance.data_sha256,
        split_sha256=protocol.provenance.split_sha256,
        split_config_sha256=protocol.provenance.split_config_sha256,
        preprocessing_sha256=protocol.provenance.preprocessing_sha256,
        selected_config_sha256=gate.config_sha256,
        model_name="phase6_locked_three_model_protocol",
        _authority=_baseline_module._FINAL_TEST_AUTHORITY,
    )
    return protocol.test_data(authorization)


def validate_test_structure(
    split: BaselineArraySplit,
    *,
    expected_cells: int,
    rows_per_cell: int,
    expected_samples: int,
) -> dict[str, Any]:
    if split.name != "test":
        raise ValueError("locked final-test runner requires a test split")
    if split.n_samples != expected_samples:
        raise ValueError(
            f"test sample count mismatch: expected={expected_samples}, actual={split.n_samples}"
        )
    cells, counts = np.unique(split.cell_ids, return_counts=True)
    if cells.size != expected_cells or not np.all(counts == rows_per_cell):
        raise ValueError("test cells are not complete fixed-length monthly clusters")
    expected_months = set(np.unique(split.month_ids).tolist())
    if len(expected_months) != rows_per_cell:
        raise ValueError("test split does not contain the expected number of months")
    for cell_id in cells:
        actual_months = set(split.month_ids[split.cell_ids == cell_id].tolist())
        if actual_months != expected_months:
            raise ValueError(f"test cell has incomplete month membership: {cell_id}")
    return {
        "sample_count": int(split.n_samples),
        "unique_cell_count": int(cells.size),
        "rows_per_cell": int(rows_per_cell),
        "unique_month_count": int(len(expected_months)),
        "complete_cell_month_structure_verified": True,
    }


def _phase6_metrics_from_csv(
    prediction_path: Path,
    full_result: Mapping[str, Any],
) -> dict[str, Any]:
    recalculated = calculate_prediction_metrics_from_csv(prediction_path)
    predictions = read_prediction_csv(prediction_path)
    spearman = spearman_correlation(
        predictions["true_original"], predictions["pred_original"]
    )
    verify_baseline_result_from_prediction_csv(prediction_path, full_result)
    original = recalculated["original_unit"]
    log_space = recalculated["log_space"]
    return {
        "primary_original_unit": {
            "mae": float(original["mae"]),
            "rmse": float(original["rmse"]),
            "r2": float(original["r2"]),
        },
        "secondary": {
            "bias": float(original["bias"]),
            "spearman": float(spearman),
            "log_unit_mae": float(log_space["mae"]),
            "log_unit_rmse": float(log_space["rmse"]),
            "log_unit_r2": float(log_space["r2"]),
        },
        "metric_recalculation_from_saved_prediction_csv": True,
    }


def _aggregate_seed_metrics(
    run_metrics: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    ddof: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_paths = {
        "mae": ("primary_original_unit", "mae"),
        "rmse": ("primary_original_unit", "rmse"),
        "r2": ("primary_original_unit", "r2"),
        "bias": ("secondary", "bias"),
        "spearman": ("secondary", "spearman"),
        "log_unit_mae": ("secondary", "log_unit_mae"),
        "log_unit_rmse": ("secondary", "log_unit_rmse"),
        "log_unit_r2": ("secondary", "log_unit_r2"),
    }
    for model_name in PHASE6_MODELS:
        rows = [run_metrics[(model_name, seed)] for seed in PHASE6_SEEDS]
        per_seed = {str(seed): rows[index] for index, seed in enumerate(PHASE6_SEEDS)}
        aggregate: dict[str, Any] = {}
        for metric_name, (namespace, key) in metric_paths.items():
            values = [float(row[namespace][key]) for row in rows]
            aggregate[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=ddof)),
            }
        output[model_name] = {
            "per_seed": per_seed,
            "aggregate": aggregate,
            "standard_deviation_ddof": ddof,
        }
    return output


def _verify_prediction_alignment(
    prediction_paths: Mapping[tuple[str, int], Path],
) -> tuple[NDArray[Any], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    reference_cells: NDArray[Any] | None = None
    reference_true: NDArray[np.float64] | None = None
    by_model: dict[str, list[NDArray[np.float64]]] = {
        model_name: [] for model_name in PHASE6_MODELS
    }
    for model_name in PHASE6_MODELS:
        for seed in PHASE6_SEEDS:
            values = read_prediction_csv(prediction_paths[(model_name, seed)])
            cells = np.asarray(values["cell_id"], dtype=str)
            true = np.asarray(values["true_original"], dtype=np.float64)
            if reference_cells is None:
                reference_cells = cells
                reference_true = true
            elif not np.array_equal(cells, reference_cells) or not np.array_equal(
                true, reference_true
            ):
                raise ValueError("test prediction row/cell/target alignment mismatch")
            by_model[model_name].append(
                np.asarray(values["pred_original"], dtype=np.float64)
            )
    assert reference_cells is not None and reference_true is not None
    means = {
        model_name: np.mean(np.stack(predictions), axis=0)
        for model_name, predictions in by_model.items()
    }
    return reference_cells, reference_true, means


def _write_final_zip(output_dir: Path) -> tuple[Path, Path, str]:
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    receipt_path = output_dir.parent / f"{output_dir.name}.sha256.json"
    if zip_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite final Phase 6 bundle or receipt")
    temporary = zip_path.with_suffix(".zip.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary final ZIP already exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir).as_posix())
        os.replace(temporary, zip_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = file_sha256(zip_path)
    write_metrics_json(
        receipt_path,
        {
            "artifact_classification": "final-test",
            "file_name": zip_path.name,
            "size_bytes": zip_path.stat().st_size,
            "sha256": digest,
        },
    )
    return zip_path, receipt_path, digest


def execute_locked_final_test(
    *,
    protocol: Any,
    gate: FinalTestGate,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize test once, predict with frozen models, and save final artifacts."""

    if gate._authority is not _PHASE6_GATE_AUTHORITY:
        raise TestEvaluationBlockedError("Phase 6 execution requires a verified gate")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6 output: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    protocol_copy = {
        **dict(gate.config),
        "source_config_sha256": gate.config_sha256,
        "artifact_classification": "final-test-protocol",
    }
    write_metrics_json(destination / "PHASE6_LOCKED_TEST_PROTOCOL.json", protocol_copy)
    write_metrics_json(destination / "final_test_gate_audit.json", gate.audit)
    log_path = destination / "execution.log"
    log_path.write_text(
        f"{_utc_now()} gate_passed; materializing test exactly once\n",
        encoding="utf-8",
    )

    test_split = _materialize_test_once(protocol, gate)
    contract = gate.config["test_contract"]
    structure = validate_test_structure(
        test_split,
        expected_cells=int(contract["expected_unique_cells"]),
        rows_per_cell=int(contract["expected_rows_per_cell"]),
        expected_samples=int(contract["expected_samples"]),
    )
    run_metrics: dict[tuple[str, int], Mapping[str, Any]] = {}
    prediction_paths: dict[tuple[str, int], Path] = {}
    run_artifacts: list[dict[str, Any]] = []
    for key in sorted(gate.loaded_models):
        loaded = gate.loaded_models[key]
        spec = loaded.spec
        run_dir = destination / "runs" / f"{spec.model_name}_seed{spec.train_seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        run_provenance = BaselineProvenance(
            data_version=protocol.provenance.data_version,
            data_sha256=protocol.provenance.data_sha256,
            split_sha256=protocol.provenance.split_sha256,
            split_config_sha256=protocol.provenance.split_config_sha256,
            preprocessing_sha256=protocol.provenance.preprocessing_sha256,
            split_seed=protocol.provenance.split_seed,
            train_seed=spec.train_seed,
            target_scale=protocol.provenance.target_scale,
        )
        evaluation = evaluate_fitted_baseline(loaded.model, test_split, run_provenance)
        prediction_path = write_prediction_csv(
            run_dir / "test_predictions.csv",
            y_true_original=test_split.target_original,
            y_pred_original=evaluation.predictions.pred_original,
            y_true_log=test_split.target_log,
            y_pred_log=evaluation.predictions.pred_log,
            cell_ids=test_split.cell_ids,
        )
        diagnostic_path = write_prediction_support_diagnostic_csv(
            run_dir / "test_negative_predictions.csv",
            y_true_original=test_split.target_original,
            unprojected_prediction=evaluation.predictions.unprojected_original,
            final_prediction=evaluation.predictions.pred_original,
            cell_ids=test_split.cell_ids,
            month_ids=test_split.month_ids,
        )
        metrics = _phase6_metrics_from_csv(prediction_path, evaluation.result)
        support = evaluation.predictions.support_diagnostics
        assert support is not None
        prediction_sha256 = file_sha256(prediction_path)
        payload = {
            "schema_version": PHASE6_METRICS_SCHEMA_VERSION,
            "artifact_classification": "final-test",
            "model_name": spec.model_name,
            "train_seed": spec.train_seed,
            "split_seed": 42,
            "data_sha256": protocol.provenance.data_sha256,
            "split_sha256": protocol.provenance.split_sha256,
            "split_config_sha256": protocol.provenance.split_config_sha256,
            "preprocessing_sha256": protocol.provenance.preprocessing_sha256,
            "configuration_sha256": spec.configuration_sha256,
            "source_model_artifact_sha256": spec.model_artifact_sha256,
            "prediction_sha256": prediction_sha256,
            **metrics,
            **support.to_dict(),
            "test_subset_materialized": True,
            "test_evaluation_performed": True,
            "training_performed": False,
            "model_selection_performed": False,
            "hyperparameter_change_performed": False,
        }
        write_metrics_json(run_dir / "test_metrics.json", payload)
        write_metrics_json(
            run_dir / "run_manifest.json",
            {
                **payload,
                "schema_version": PHASE6_MANIFEST_SCHEMA_VERSION,
                "test_prediction_file": prediction_path.name,
                "negative_prediction_diagnostic_file": diagnostic_path.name,
                "source_validation_manifest_sha256": spec.manifest_sha256,
                "source_validation_prediction_sha256": spec.validation_prediction_sha256,
                "completed_at_utc": _utc_now(),
            },
        )
        run_metrics[key] = metrics
        prediction_paths[key] = prediction_path
        run_artifacts.append(
            {
                "model_name": spec.model_name,
                "train_seed": spec.train_seed,
                "prediction_sha256": prediction_sha256,
                "model_artifact_sha256": spec.model_artifact_sha256,
            }
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{_utc_now()} completed {spec.model_name} seed {spec.train_seed}\n")

    ddof = int(gate.config["metrics"]["seed_summary_standard_deviation_ddof"])
    summaries = _aggregate_seed_metrics(run_metrics, ddof=ddof)
    write_metrics_json(
        destination / "test_seed_metric_summary.json",
        {
            "schema_version": PHASE6_SUMMARY_SCHEMA_VERSION,
            "artifact_classification": "final-test",
            "models": summaries,
        },
    )
    cell_ids, true_original, means = _verify_prediction_alignment(prediction_paths)
    bootstrap = gate.config["paired_cell_cluster_bootstrap"]
    comparisons = {
        "bidirectional_vs_token_no_attention": paired_cell_cluster_mae_difference(
            y_true_original=true_original,
            bidirectional_prediction=means["cell_msca_bidirectional"],
            comparator_prediction=means["cell_msca_token_no_attention"],
            cell_ids=cell_ids,
            n_boot=int(bootstrap["n_boot"]),
            seed=int(bootstrap["seed"]),
            alpha=float(bootstrap["alpha"]),
            rows_per_cell=int(bootstrap["rows_per_cell"]),
        ),
        "bidirectional_vs_lightgbm_raw": paired_cell_cluster_mae_difference(
            y_true_original=true_original,
            bidirectional_prediction=means["cell_msca_bidirectional"],
            comparator_prediction=means["lightgbm_raw"],
            cell_ids=cell_ids,
            n_boot=int(bootstrap["n_boot"]),
            seed=int(bootstrap["seed"]),
            alpha=float(bootstrap["alpha"]),
            rows_per_cell=int(bootstrap["rows_per_cell"]),
        ),
    }
    write_metrics_json(
        destination / "paired_test_cell_cluster_bootstrap.json",
        {
            "schema_version": PHASE6_BOOTSTRAP_SCHEMA_VERSION,
            "artifact_classification": "final-test",
            "prediction_basis": bootstrap["prediction_basis"],
            "comparisons": comparisons,
            "limitation": (
                "Complete-cell resampling conditions on rowwise mean predictions "
                "from three frozen training seeds; it does not include training-seed uncertainty."
            ),
        },
    )
    manifest = {
        "schema_version": PHASE6_MANIFEST_SCHEMA_VERSION,
        "artifact_classification": "final-test",
        "status": "completed",
        "protocol_config_sha256": gate.config_sha256,
        "validation_code_git_sha": gate.config["validation_code_git_sha"],
        "runner_git_sha": gate.audit.get("runner_git_sha"),
        "data_sha256": protocol.provenance.data_sha256,
        "split_sha256": protocol.provenance.split_sha256,
        "split_config_sha256": protocol.provenance.split_config_sha256,
        "preprocessing_sha256": protocol.provenance.preprocessing_sha256,
        "test_structure": structure,
        "test_materialization_count": 1,
        "test_subset_materialized": True,
        "test_evaluation_performed": True,
        "training_performed": False,
        "model_selection_performed": False,
        "hyperparameter_change_performed": False,
        "prediction_support_policy": NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
        "prediction_metrics_recalculated_from_saved_csv": True,
        "prediction_alignment_verified": True,
        "run_artifacts": run_artifacts,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pytorch": _distribution_version("torch"),
            "lightgbm": _distribution_version("lightgbm"),
        },
        "completed_at_utc": _utc_now(),
    }
    write_metrics_json(destination / "phase6_test_manifest.json", manifest)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} aggregation_and_bootstrap_completed\n")
    zip_path, receipt_path, zip_sha256 = _write_final_zip(destination)
    return {
        "output_dir": str(destination),
        "final_zip": str(zip_path),
        "final_zip_sha256_receipt": str(receipt_path),
        "final_zip_sha256": zip_sha256,
        "manifest": manifest,
    }


def run_locked_final_test(
    *,
    config_path: str | Path,
    input_root: str | Path,
    working_root: str | Path,
    repository_root: str | Path,
    phase5a_zip: str | Path,
    seed42_zip: str | Path,
    output_dir: str | Path,
    allow_final_test: bool,
    device: str,
    kaggle: bool = False,
    model_loader: ModelLoader | None = None,
) -> dict[str, Any]:
    """Run the frozen protocol; no fit or selection function is reachable here."""

    if not allow_final_test:
        raise TestEvaluationBlockedError(
            "locked final-test evaluation requires explicit --allow-final-test"
        )
    if Path(output_dir).exists():
        raise FileExistsError(f"refusing to overwrite Phase 6 output: {output_dir}")
    config = load_phase6_protocol(config_path)
    repository = Path(repository_root).resolve()
    manifest_path = repository / str(config["data"]["archive_manifest"])
    with verified_v1_input(
        input_root=input_root,
        working_root=working_root,
        manifest_path=manifest_path,
        repository_root=repository,
        kaggle=kaggle,
        train_seed=42,
    ) as verified, TemporaryDirectory(
        prefix="cell-msca-phase6-models-", dir=Path(working_root)
    ) as temporary:
        gate = verify_final_test_gate(
            config_path=config_path,
            phase5a_zip=phase5a_zip,
            seed42_zip=seed42_zip,
            output_dir=output_dir,
            extraction_root=temporary,
            allow_final_test=allow_final_test,
            protocol=verified.protocol,
            device=device,
            model_loader=model_loader,
            repository_root=repository,
            kaggle=kaggle,
        )
        return execute_locked_final_test(
            protocol=verified.protocol,
            gate=gate,
            output_dir=output_dir,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate nine validation-frozen models on final test exactly once",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--working-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--phase5a-zip", required=True, type=Path)
    parser.add_argument("--seed42-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = run_locked_final_test(
        config_path=arguments.config,
        input_root=arguments.input_root,
        working_root=arguments.working_root,
        repository_root=arguments.repository_root,
        phase5a_zip=arguments.phase5a_zip,
        seed42_zip=arguments.seed42_zip,
        output_dir=arguments.output_dir,
        allow_final_test=arguments.allow_final_test,
        device=arguments.device,
        kaggle=arguments.kaggle,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
