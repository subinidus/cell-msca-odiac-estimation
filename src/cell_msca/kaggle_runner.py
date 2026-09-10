"""Thin validation-only execution layer for local and Kaggle Cell-MSCA runs."""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .baselines import (
    BaselineDataProtocol,
    TestEvaluationBlockedError,
    TuningData,
    evaluate_fitted_baseline,
)
from .data import (
    STREAM_A_FEATURES,
    STREAM_B_FEATURES,
    CellDataset,
    canonical_sha256,
    file_sha256,
)
from .evaluate import (
    verify_baseline_result_from_prediction_csv,
    write_metrics_json,
    write_prediction_csv,
)
from .splits import CellFixedSplitConfig, create_persistent_cell_fixed_split

KAGGLE_VALIDATION_SCHEMA_VERSION = "cell_msca.kaggle_validation.v1"
KAGGLE_MANIFEST_SCHEMA_VERSION = "cell_msca.kaggle_run_manifest.v1"
EXPERIMENT_ASSIGNMENT_SCHEMA_VERSION = "cell_msca.experiment_assignment.v1"
SYNTHETIC_DATA_SCHEMA_VERSION = "cell_msca.synthetic_validation.v1"
SOURCE_TREE_MANIFEST_SCHEMA_VERSION = "cell_msca.source_tree_manifest.v1"
ATTACHED_SOURCE_POLICY = "attached_code_dataset_manifest_verified_no_git_worktree"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
SOURCE_MANIFEST_DIRECTORIES = ("src/cell_msca", "configs")
SOURCE_MANIFEST_FILES = (
    "notebooks/cell_msca_kaggle_validation.ipynb",
    "pyproject.toml",
    "requirements.txt",
)
SOURCE_MANIFEST_IGNORED_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
}
FORBIDDEN_CONFIG_FIELDS = {
    "access_token",
    "api_key",
    "credential",
    "credentials",
    "password",
    "private_url",
    "secret",
    "token",
}
FORBIDDEN_TEST_CONFIG_FIELDS = {
    "evaluate_test",
    "test",
    "test_evaluation",
    "test_path",
    "test_split",
    "test_subset",
}
ARTIFACT_CLASSIFICATIONS = {
    "resolved_config.json": "engineering-only",
    "run_manifest.json": "engineering-only",
    "environment.json": "engineering-only",
    "validation_metrics.json": "validation-only",
    "validation_predictions.csv": "validation-only",
    "selected_checkpoint.pt": "validation-only",
    "execution.log": "engineering-only",
}


@dataclass(frozen=True)
class GitIdentity:
    commit_sha: str
    worktree_dirty: bool
    dirty_state_policy: str
    source: str
    source_manifest_sha256: str | None = None
    verified_source_file_count: int | None = None

    def __post_init__(self) -> None:
        commit = self.commit_sha.lower()
        if not GIT_SHA_PATTERN.fullmatch(commit):
            raise ValueError("Git commit SHA must contain 40-64 lowercase hex characters")
        object.__setattr__(self, "commit_sha", commit)
        if not self.dirty_state_policy:
            raise ValueError("Git dirty-state policy must not be empty")
        if not self.source:
            raise ValueError("Git identity source must not be empty")
        if self.source == "attached_private_code_dataset":
            manifest_sha256 = _validate_sha256(
                self.source_manifest_sha256,
                name="source_manifest_sha256",
            )
            object.__setattr__(self, "source_manifest_sha256", manifest_sha256)
            if (
                self.verified_source_file_count is None
                or self.verified_source_file_count <= 0
            ):
                raise ValueError(
                    "attached source identity requires verified source files"
                )


@dataclass(frozen=True)
class ValidationRunArtifacts:
    experiment_id: str
    output_dir: Path
    resolved_config_json: Path
    run_manifest_json: Path
    environment_json: Path
    validation_metrics_json: Path
    validation_predictions_csv: Path
    selected_checkpoint: Path
    execution_log: Path


class _ExecutionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, message: str) -> None:
        timestamp = _utc_now()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} {message}\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_sha256(value: Any, *, name: str) -> str:
    digest = str(value).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} must be an exact lowercase SHA-256 digest")
    return digest


def _walk_config_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_walk_config_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_config_keys(nested))
    return keys


def load_kaggle_validation_config(path: str | Path) -> dict[str, Any]:
    """Load a frozen validation-only configuration without resolving runtime paths."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("Kaggle validation config root must be an object")
    if values.get("schema_version") != KAGGLE_VALIDATION_SCHEMA_VERSION:
        raise ValueError("unsupported Kaggle validation config schema_version")
    if values.get("stage") != "validation_only":
        raise TestEvaluationBlockedError(
            "Kaggle runner stage must be validation_only; test access is closed"
        )
    if values.get("allowed_materialized_splits") != ["train", "validation"]:
        raise TestEvaluationBlockedError(
            "allowed_materialized_splits must be exactly train and validation"
        )
    forbidden = [
        key
        for key in _walk_config_keys(values)
        if key.lower() in FORBIDDEN_CONFIG_FIELDS
    ]
    if forbidden:
        raise ValueError(f"configuration contains forbidden sensitive fields: {forbidden}")
    forbidden_test = [
        key for key in _walk_config_keys(values) if key.lower() in FORBIDDEN_TEST_CONFIG_FIELDS
    ]
    if forbidden_test:
        raise TestEvaluationBlockedError(
            f"configuration contains forbidden test access fields: {forbidden_test}"
        )
    for key in ("owner", "model_family"):
        if not str(values.get(key, "")).strip():
            raise ValueError(f"configuration {key} must not be empty")
    if values["model_family"] != "cell_msca":
        raise ValueError("model_family must be cell_msca")
    if not isinstance(values.get("training"), dict):
        raise ValueError("configuration requires a training object")
    if not isinstance(values.get("data"), dict):
        raise ValueError("configuration requires a data object")
    data_mode = values["data"].get("mode")
    if data_mode not in {"synthetic", "npz"}:
        raise ValueError("data.mode must be 'synthetic' or 'npz'")
    if (
        data_mode == "synthetic"
        and values["data"].get("synthetic_schema_version")
        != SYNTHETIC_DATA_SCHEMA_VERSION
    ):
        raise ValueError("unsupported synthetic data schema version")
    if (
        data_mode == "synthetic"
        and values["data"].get("data_hash_mode") != "canonical_npz_content"
    ):
        raise ValueError(
            "synthetic data requires data_hash_mode='canonical_npz_content'"
        )
    if not isinstance(values.get("split"), dict):
        raise ValueError("configuration requires a split object")
    required_hashes = values.get("required_hashes")
    if not isinstance(required_hashes, dict):
        raise ValueError("configuration requires a required_hashes object")
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    ):
        _validate_sha256(required_hashes.get(name), name=name)
    configuration_hashes = required_hashes.get(
        "configuration_sha256_by_variant_and_device"
    )
    if not isinstance(configuration_hashes, dict) or not configuration_hashes:
        raise ValueError(
            "required_hashes.configuration_sha256_by_variant_and_device is required"
        )
    for key, digest in configuration_hashes.items():
        if not isinstance(key, str) or ":" not in key:
            raise ValueError("configuration hash keys must be '<variant>:<device>'")
        _validate_sha256(digest, name=f"configuration_sha256[{key}]")
    return values


def validate_experiment_assignments(values: Mapping[str, Any]) -> None:
    """Validate assignment ownership and reject duplicate experiment identifiers."""

    if values.get("schema_version") != EXPERIMENT_ASSIGNMENT_SCHEMA_VERSION:
        raise ValueError("unsupported experiment assignment schema_version")
    if values.get("hyperparameter_policy") != "frozen_no_independent_changes":
        raise ValueError("experiment owners must execute frozen configurations")
    assignments = values.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("experiment assignment template requires assignments")
    required = {
        "owner",
        "experiment_id",
        "model_family",
        "variant",
        "loss",
        "seed",
        "status",
        "output_location",
        "notes",
    }
    experiment_ids: list[str] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or set(assignment) != required:
            raise ValueError(f"assignment {index} fields must be exactly {sorted(required)}")
        experiment_id = str(assignment["experiment_id"]).strip()
        if not experiment_id:
            raise ValueError(f"assignment {index} experiment_id must not be empty")
        experiment_ids.append(experiment_id)
    duplicates = sorted(
        identifier
        for identifier in set(experiment_ids)
        if experiment_ids.count(identifier) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate experiment IDs are not allowed: {duplicates}")


def load_experiment_assignments(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("experiment assignment root must be an object")
    validate_experiment_assignments(values)
    return values


def deterministic_experiment_id(
    *,
    owner: str,
    variant: str,
    train_seed: int,
    split_seed: int,
    device: str,
    data_sha256: str,
    split_sha256: str,
    split_config_sha256: str,
    preprocessing_sha256: str,
    configuration_sha256: str,
    git_commit_sha: str,
) -> str:
    """Create a stable identifier from the frozen execution contract."""

    payload = {
        "schema_version": KAGGLE_VALIDATION_SCHEMA_VERSION,
        "owner": str(owner),
        "model_family": "cell_msca",
        "variant": str(variant),
        "train_seed": int(train_seed),
        "split_seed": int(split_seed),
        "device": str(device),
        "data_sha256": _validate_sha256(data_sha256, name="data_sha256"),
        "split_sha256": _validate_sha256(split_sha256, name="split_sha256"),
        "split_config_sha256": _validate_sha256(
            split_config_sha256,
            name="split_config_sha256",
        ),
        "preprocessing_sha256": _validate_sha256(
            preprocessing_sha256,
            name="preprocessing_sha256",
        ),
        "configuration_sha256": _validate_sha256(
            configuration_sha256,
            name="configuration_sha256",
        ),
        "git_commit_sha": str(git_commit_sha).lower(),
    }
    if not GIT_SHA_PATTERN.fullmatch(payload["git_commit_sha"]):
        raise ValueError("git_commit_sha must be an exact 40-64 character hex digest")
    digest = canonical_sha256(payload)[:16]
    safe_variant = re.sub(r"[^a-z0-9_-]+", "-", str(variant).lower())
    return f"cell-msca-{safe_variant}-s{int(train_seed)}-{digest}"


def _posix_path(value: str | Path) -> PurePosixPath:
    return PurePosixPath(str(value).replace("\\", "/"))


def _is_posix_relative_to(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_execution_paths(
    *,
    data_path: str | Path,
    output_path: str | Path,
    data_mode: str,
    kaggle: bool,
) -> None:
    """Enforce read-only inputs and Kaggle's writable working directory."""

    output_posix = _posix_path(output_path)
    kaggle_input = PurePosixPath("/kaggle/input")
    kaggle_working = PurePosixPath("/kaggle/working")
    if _is_posix_relative_to(output_posix, kaggle_input):
        raise ValueError("output_path must never be under read-only /kaggle/input")
    if kaggle and not _is_posix_relative_to(output_posix, kaggle_working):
        raise ValueError("Kaggle output_path must be under /kaggle/working")
    if data_mode == "npz":
        data_posix = _posix_path(data_path)
        if kaggle and not _is_posix_relative_to(data_posix, kaggle_input):
            raise ValueError("Kaggle NPZ data_path must be under /kaggle/input")
        data_resolved = Path(data_path).resolve()
        output_resolved = Path(output_path).resolve()
        try:
            output_resolved.relative_to(data_resolved)
        except ValueError:
            pass
        else:
            raise ValueError("output_path must not be inside the read-only data_path")
    elif data_mode == "synthetic":
        if str(data_path) not in {"generated", "synthetic", "generated://synthetic"}:
            raise ValueError("synthetic data_path must explicitly be 'generated'")
    else:
        raise ValueError("data.mode must be 'synthetic' or 'npz'")


def _source_manifest_paths(repository_root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for relative_directory in SOURCE_MANIFEST_DIRECTORIES:
        directory = repository_root / relative_directory
        if not directory.is_dir():
            raise FileNotFoundError(
                f"source manifest scope is missing directory: {relative_directory}"
            )
        for candidate in directory.rglob("*"):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(repository_root)
            if any(part in SOURCE_MANIFEST_IGNORED_PARTS for part in relative.parts):
                continue
            if candidate.suffix.lower() in {".pyc", ".pyo"}:
                continue
            if candidate.is_symlink():
                raise ValueError(
                    f"source manifest scope rejects symbolic links: {relative.as_posix()}"
                )
            paths.add(relative)
    for relative_file in SOURCE_MANIFEST_FILES:
        candidate = repository_root / relative_file
        if not candidate.is_file():
            raise FileNotFoundError(
                f"source manifest scope is missing file: {relative_file}"
            )
        if candidate.is_symlink():
            raise ValueError(
                f"source manifest scope rejects symbolic links: {relative_file}"
            )
        paths.add(Path(relative_file))
    return tuple(sorted(paths, key=lambda value: value.as_posix()))


def build_source_tree_manifest(
    repository_root: str | Path,
    *,
    git_commit_sha: str,
) -> dict[str, Any]:
    """Build the attached-code integrity manifest for the frozen source scope."""

    root = Path(repository_root).resolve()
    commit = str(git_commit_sha).lower()
    if not GIT_SHA_PATTERN.fullmatch(commit):
        raise ValueError("git_commit_sha must be an exact Git commit SHA")
    files = [
        {
            "path": relative.as_posix(),
            "sha256": file_sha256(root / relative),
        }
        for relative in _source_manifest_paths(root)
    ]
    return {
        "schema_version": SOURCE_TREE_MANIFEST_SCHEMA_VERSION,
        "git_commit_sha": commit,
        "scope": {
            "directories": list(SOURCE_MANIFEST_DIRECTORIES),
            "files": list(SOURCE_MANIFEST_FILES),
        },
        "files": files,
    }


def write_source_tree_manifest(
    repository_root: str | Path,
    destination: str | Path,
    *,
    git_commit_sha: str,
) -> str:
    """Write a source manifest without overwriting and return its file SHA-256."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_source_tree_manifest(
        repository_root,
        git_commit_sha=git_commit_sha,
    )
    _atomic_write_json(output, manifest)
    return file_sha256(output)


def verify_source_tree_manifest(
    repository_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_git_sha: str,
    expected_manifest_sha256: str,
) -> int:
    """Fail closed unless every required attached-source file matches its manifest."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path)
    expected_manifest = _validate_sha256(
        expected_manifest_sha256,
        name="expected_source_manifest_sha256",
    )
    if not path.is_file():
        raise FileNotFoundError(f"attached source manifest is missing: {path}")
    actual_manifest = file_sha256(path)
    if actual_manifest != expected_manifest:
        raise ValueError(
            "attached source manifest SHA-256 mismatch: "
            f"expected={expected_manifest}, actual={actual_manifest}"
        )
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("attached source manifest root must be an object")
    if manifest.get("schema_version") != SOURCE_TREE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported attached source manifest schema_version")
    expected_commit = str(expected_git_sha).lower()
    if manifest.get("git_commit_sha") != expected_commit:
        raise ValueError("attached source manifest Git SHA mismatch")
    if manifest.get("scope") != {
        "directories": list(SOURCE_MANIFEST_DIRECTORIES),
        "files": list(SOURCE_MANIFEST_FILES),
    }:
        raise ValueError("attached source manifest scope does not match the runner contract")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("attached source manifest requires file entries")
    recorded: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"invalid attached source manifest entry at index {index}")
        relative_text = str(entry["path"])
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_text != relative.as_posix()
        ):
            raise ValueError(f"unsafe repository-relative manifest path: {relative_text!r}")
        if relative_text in recorded:
            raise ValueError(f"duplicate attached source manifest path: {relative_text}")
        recorded[relative_text] = _validate_sha256(
            entry["sha256"],
            name=f"source manifest sha256[{relative_text}]",
        )
    actual_paths = {
        relative.as_posix(): relative for relative in _source_manifest_paths(root)
    }
    if set(recorded) != set(actual_paths):
        missing = sorted(set(actual_paths) - set(recorded))
        unexpected = sorted(set(recorded) - set(actual_paths))
        raise ValueError(
            "attached source manifest coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative_text, relative in actual_paths.items():
        actual_sha256 = file_sha256(root / relative)
        if actual_sha256 != recorded[relative_text]:
            raise ValueError(
                f"attached source file SHA-256 mismatch: {relative_text}"
            )
    return len(recorded)


def collect_environment() -> dict[str, Any]:
    """Collect versions and fail clearly for incompatible required packages."""

    python_version = platform.python_version()
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >=3.10 is required; found {python_version}")
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise RuntimeError("compatible PyTorch is required and must be preinstalled") from error
    torch_version = str(torch.__version__)
    version_match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if version_match is None or tuple(map(int, version_match.groups())) < (2, 1):
        raise RuntimeError(f"PyTorch >=2.1 is required; found {torch_version}")
    numpy_version = str(np.__version__)
    numpy_match = re.match(r"^(\d+)\.(\d+)", numpy_version)
    if numpy_match is None or tuple(map(int, numpy_match.groups())) < (1, 24):
        raise RuntimeError(f"NumPy >=1.24 is required; found {numpy_version}")
    try:
        lightgbm_version = importlib.metadata.version("lightgbm")
    except importlib.metadata.PackageNotFoundError:
        lightgbm_version = "not-installed"
    cuda_available = bool(torch.cuda.is_available())
    return {
        "schema_version": "cell_msca.runtime_environment.v1",
        "artifact_classification": "engineering-only",
        "python": python_version,
        "platform": platform.platform(),
        "pytorch": torch_version,
        "numpy": numpy_version,
        "lightgbm": lightgbm_version,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "compatibility": {
            "python_required": ">=3.10",
            "pytorch_required": ">=2.1",
            "numpy_required": ">=1.24",
            "lightgbm_required_for_cell_msca_run": False,
        },
    }


def resolve_git_identity(
    repository_root: str | Path,
    *,
    expected_git_sha: str,
    source_git_sha_file: str | Path | None = None,
    source_tree_manifest: str | Path | None = None,
    expected_source_manifest_sha256: str | None = None,
) -> GitIdentity:
    """Verify a clean checkout or a manifest-bound attached source tree."""

    expected = str(expected_git_sha).lower()
    if not GIT_SHA_PATTERN.fullmatch(expected):
        raise ValueError("expected_git_sha must be an exact Git commit SHA")
    root = Path(repository_root).resolve()
    attached_values = (
        source_git_sha_file,
        source_tree_manifest,
        expected_source_manifest_sha256,
    )
    if any(value is not None for value in attached_values):
        if any(value is None for value in attached_values):
            raise RuntimeError(
                "attached code requires --source-git-sha-file, "
                "--source-tree-manifest, and --expected-source-manifest-sha256"
            )
        assert source_git_sha_file is not None
        assert source_tree_manifest is not None
        assert expected_source_manifest_sha256 is not None
        declaration = Path(source_git_sha_file)
        if not declaration.is_file():
            raise FileNotFoundError(
                f"attached source Git SHA declaration is missing: {declaration}"
            )
        commit = declaration.read_text(encoding="utf-8").strip().lower()
        if not GIT_SHA_PATTERN.fullmatch(commit):
            raise ValueError("source Git SHA file does not contain an exact commit SHA")
        if commit != expected:
            raise ValueError(
                f"Git SHA mismatch: expected={expected}, declared={commit}"
            )
        verified_file_count = verify_source_tree_manifest(
            root,
            source_tree_manifest,
            expected_git_sha=commit,
            expected_manifest_sha256=expected_source_manifest_sha256,
        )
        identity = GitIdentity(
            commit,
            False,
            ATTACHED_SOURCE_POLICY,
            "attached_private_code_dataset",
            source_manifest_sha256=expected_source_manifest_sha256.lower(),
            verified_source_file_count=verified_file_count,
        )
    else:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
            status = subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Git metadata is unavailable; attached code requires a verified "
                "source manifest"
            ) from error
        if not GIT_SHA_PATTERN.fullmatch(commit):
            raise RuntimeError("git rev-parse returned an invalid commit SHA")
        identity = GitIdentity(
            commit,
            bool(status.strip()),
            "tracked_and_untracked_files",
            "git_checkout",
        )
    if identity.commit_sha != expected:
        raise ValueError(
            f"Git SHA mismatch: expected={expected}, actual={identity.commit_sha}"
        )
    if identity.worktree_dirty:
        raise RuntimeError("validation runner requires a clean tracked-and-untracked tree")
    return identity


def _build_training_config(
    values: Mapping[str, Any],
    *,
    variant: str,
    device: str,
) -> Any:
    from .model import CELL_MSCA_VARIANTS, CellMSCAConfig
    from .train import CellMSCATrainingConfig

    if variant not in CELL_MSCA_VARIANTS:
        raise ValueError(f"variant must be one of {CELL_MSCA_VARIANTS}")
    training = dict(values["training"])
    architecture_values = training.pop("architecture", None)
    if not isinstance(architecture_values, dict):
        raise ValueError("training.architecture must be an object")
    architecture = CellMSCAConfig(variant=variant, **architecture_values)
    return CellMSCATrainingConfig(
        architecture=architecture,
        device=device,
        **training,
    )


def _write_synthetic_month(path: Path, *, month_index: int) -> None:
    height, width = 4, 5
    grid = np.arange(height * width, dtype=np.float64).reshape(height, width)
    stream_a = np.stack(
        [grid / 10.0 + channel + month_index / 10.0 for channel in range(3)]
    )
    stream_b = np.stack(
        [grid / 20.0 + 2.0 * channel + month_index / 10.0 for channel in range(4)]
    )
    target = np.square(grid / 5.0 + 0.2 * month_index) + 0.5
    np.savez_compressed(
        path,
        stream_a=stream_a,
        stream_b=stream_b,
        label_reg=target,
        mask=np.ones((height, width), dtype=np.uint8),
        month_id=np.asarray(f"2026-{month_index + 1:02d}"),
        stream_a_feature_names=np.asarray(STREAM_A_FEATURES),
        stream_b_feature_names=np.asarray(STREAM_B_FEATURES),
        data_version=np.asarray("synthetic_kaggle_smoke_v1"),
        target_unit=np.asarray("synthetic_unit"),
    )


def _synthetic_tuning_data(
    directory: Path,
    *,
    split_config: CellFixedSplitConfig,
    train_seed: int,
    target_scale: float,
) -> TuningData:
    data_dir = directory / "generated_data"
    data_dir.mkdir(parents=True)
    paths: list[Path] = []
    for month_index in range(2):
        path = data_dir / f"month_{month_index:02d}.npz"
        _write_synthetic_month(path, month_index=month_index)
        paths.append(path)
    dataset = CellDataset(
        paths,
        target_scale=target_scale,
        data_hash_mode="canonical_npz_content",
    )
    split_csv = directory / "synthetic_split.csv"
    split_metadata = directory / "synthetic_split.metadata.json"
    create_persistent_cell_fixed_split(
        dataset,
        split_csv,
        split_metadata,
        config=split_config,
    )
    protocol = BaselineDataProtocol.from_dataset(
        dataset,
        split_csv,
        split_metadata,
        split_config=split_config,
        train_seed=train_seed,
    )
    return protocol.tuning_data()


def _npz_tuning_data(
    values: Mapping[str, Any],
    *,
    config_path: Path,
    data_path: str | Path,
    split_config: CellFixedSplitConfig,
    train_seed: int,
) -> TuningData:
    data_values = values["data"]
    pattern = str((Path(data_path) / str(data_values["npz_glob"])).resolve())
    paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"data path matched no NPZ files: {pattern}")
    dataset = CellDataset(
        paths,
        target_scale=float(data_values.get("target_scale", 1.0)),
        legacy_data_version=data_values.get("legacy_data_version"),
        legacy_target_unit=data_values.get("legacy_target_unit"),
        data_hash_mode=str(data_values.get("data_hash_mode", "file_bytes")),
    )
    project_root = (config_path.parent / values.get("project_root", "..")).resolve()
    split_values = values["split"]
    protocol = BaselineDataProtocol.from_dataset(
        dataset,
        project_root / str(split_values["csv"]),
        project_root / str(split_values["metadata_json"]),
        split_config=split_config,
        train_seed=train_seed,
    )
    return protocol.tuning_data()


def _validated_tuning_data(
    values: Mapping[str, Any],
    *,
    config_path: Path,
    data_path: str | Path,
    output_root: Path,
    split_config: CellFixedSplitConfig,
    train_seed: int,
) -> TuningData:
    data_values = values["data"]
    mode = str(data_values["mode"])
    if mode == "synthetic":
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".cell-msca-synthetic-",
            dir=output_root,
        ) as directory:
            data = _synthetic_tuning_data(
                Path(directory),
                split_config=split_config,
                train_seed=train_seed,
                target_scale=float(data_values.get("target_scale", 1.0)),
            )
    else:
        data = _npz_tuning_data(
            values,
            config_path=config_path,
            data_path=data_path,
            split_config=split_config,
            train_seed=train_seed,
        )
    data.subset("train")
    data.subset("validation")
    return data


def _verify_data_hashes(data: TuningData, expected: Mapping[str, Any]) -> None:
    actual = {
        "data_sha256": data.provenance.data_sha256,
        "split_sha256": data.provenance.split_sha256,
        "split_config_sha256": data.provenance.split_config_sha256,
        "preprocessing_sha256": data.provenance.preprocessing_sha256,
    }
    for name, actual_digest in actual.items():
        expected_digest = _validate_sha256(expected[name], name=name)
        if actual_digest != expected_digest:
            raise ValueError(
                f"{name} mismatch: expected={expected_digest}, actual={actual_digest}"
            )


def _atomic_write_json(path: Path, values: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(name)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _artifact_manifest() -> dict[str, dict[str, str]]:
    return {
        name: {"path": name, "artifact_classification": classification}
        for name, classification in ARTIFACT_CLASSIFICATIONS.items()
    }


def run_kaggle_validation(
    *,
    variant: str,
    train_seed: int,
    config_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    device: str,
    expected_git_sha: str,
    kaggle: bool = False,
    source_git_sha_file: str | Path | None = None,
    source_tree_manifest: str | Path | None = None,
    expected_source_manifest_sha256: str | None = None,
    repository_root: str | Path | None = None,
    git_identity: GitIdentity | None = None,
) -> ValidationRunArtifacts:
    """Run one frozen Cell-MSCA variant using train and validation only."""

    config_path = Path(config_path).resolve()
    output_root = Path(output_path).resolve()
    values = load_kaggle_validation_config(config_path)
    if int(values.get("frozen_train_seed")) != int(train_seed):
        raise ValueError("train_seed differs from the frozen configuration")
    data_mode = str(values["data"]["mode"])
    validate_execution_paths(
        data_path=data_path,
        output_path=output_root,
        data_mode=data_mode,
        kaggle=kaggle,
    )
    environment = collect_environment()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    identity = git_identity or resolve_git_identity(
        root,
        expected_git_sha=expected_git_sha,
        source_git_sha_file=source_git_sha_file,
        source_tree_manifest=source_tree_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if identity.commit_sha != str(expected_git_sha).lower():
        raise ValueError("injected Git identity does not match expected_git_sha")
    if identity.worktree_dirty:
        raise RuntimeError("validation runner requires a clean source state")

    split_values = values["split"]
    split_config = CellFixedSplitConfig(
        split_seed=int(split_values["split_seed"]),
        validation_ratio=float(split_values["validation_ratio"]),
        test_ratio=float(split_values["test_ratio"]),
    )
    training_config = _build_training_config(values, variant=variant, device=device)
    # Keep the frozen JSON scalar representation in the configuration contract.
    # In particular, canonical JSON distinguishes ``1`` from ``1.0``.
    target_scale = values["data"].get("target_scale", 1.0)
    configuration_sha256 = canonical_sha256(
        training_config.to_dict(train_seed=train_seed, target_scale=target_scale)
    )
    configuration_key = f"{variant}:{device}"
    configuration_hashes = values["required_hashes"][
        "configuration_sha256_by_variant_and_device"
    ]
    if configuration_key not in configuration_hashes:
        raise ValueError(f"missing frozen configuration hash for {configuration_key}")
    expected_configuration_sha256 = _validate_sha256(
        configuration_hashes[configuration_key],
        name=f"configuration_sha256[{configuration_key}]",
    )
    if configuration_sha256 != expected_configuration_sha256:
        raise ValueError(
            "configuration_sha256 mismatch: "
            f"expected={expected_configuration_sha256}, "
            f"actual={configuration_sha256}"
        )

    expected_hashes = values["required_hashes"]
    experiment_id = deterministic_experiment_id(
        owner=str(values["owner"]),
        variant=variant,
        train_seed=train_seed,
        split_seed=split_config.split_seed,
        device=device,
        data_sha256=expected_hashes["data_sha256"],
        split_sha256=expected_hashes["split_sha256"],
        split_config_sha256=expected_hashes["split_config_sha256"],
        preprocessing_sha256=expected_hashes["preprocessing_sha256"],
        configuration_sha256=configuration_sha256,
        git_commit_sha=identity.commit_sha,
    )
    run_dir = output_root / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    artifacts = ValidationRunArtifacts(
        experiment_id=experiment_id,
        output_dir=run_dir,
        resolved_config_json=run_dir / "resolved_config.json",
        run_manifest_json=run_dir / "run_manifest.json",
        environment_json=run_dir / "environment.json",
        validation_metrics_json=run_dir / "validation_metrics.json",
        validation_predictions_csv=run_dir / "validation_predictions.csv",
        selected_checkpoint=run_dir / "selected_checkpoint.pt",
        execution_log=run_dir / "execution.log",
    )
    artifacts.execution_log.touch(exist_ok=False)
    logger = _ExecutionLogger(artifacts.execution_log)
    start_time = _utc_now()
    base_manifest: dict[str, Any] = {
        "schema_version": KAGGLE_MANIFEST_SCHEMA_VERSION,
        "artifact_classification": "engineering-only",
        "stage": "validation_only",
        "status": "running",
        "owner": values["owner"],
        "experiment_id": experiment_id,
        "model_family": "cell_msca",
        "model_name": f"cell_msca_{variant}",
        "variant": variant,
        "train_seed": train_seed,
        "split_seed": split_config.split_seed,
        "device": device,
        "data_sha256": expected_hashes["data_sha256"],
        "split_sha256": expected_hashes["split_sha256"],
        "split_config_sha256": expected_hashes["split_config_sha256"],
        "preprocessing_sha256": expected_hashes["preprocessing_sha256"],
        "configuration_sha256": configuration_sha256,
        "config_file_sha256": file_sha256(config_path),
        "git_commit_sha": identity.commit_sha,
        "git_identity_source": identity.source,
        "git_dirty_state_policy": identity.dirty_state_policy,
        "git_worktree_dirty": identity.worktree_dirty,
        "source_manifest_sha256": identity.source_manifest_sha256,
        "verified_source_file_count": identity.verified_source_file_count,
        "runtime_versions": {
            "python": environment["python"],
            "pytorch": environment["pytorch"],
            "numpy": environment["numpy"],
            "lightgbm": environment["lightgbm"],
        },
        "start_time_utc": start_time,
        "completion_time_utc": None,
        "allowed_materialized_splits": ["train", "validation"],
        "test_subset_materialized": False,
        "artifacts": _artifact_manifest(),
    }
    resolved_config = {
        "schema_version": KAGGLE_VALIDATION_SCHEMA_VERSION,
        "artifact_classification": "engineering-only",
        "experiment_id": experiment_id,
        "source_config": str(config_path),
        "data_path": str(data_path),
        "output_dir": str(run_dir),
        "owner": values["owner"],
        "model_family": "cell_msca",
        "variant": variant,
        "train_seed": train_seed,
        "split_seed": split_config.split_seed,
        "device": device,
        "training_config": training_config.to_dict(
            train_seed=train_seed,
            target_scale=target_scale,
        ),
        "required_hashes": {
            "data_sha256": expected_hashes["data_sha256"],
            "split_sha256": expected_hashes["split_sha256"],
            "split_config_sha256": expected_hashes["split_config_sha256"],
            "preprocessing_sha256": expected_hashes["preprocessing_sha256"],
            "configuration_sha256": configuration_sha256,
            "git_commit_sha": identity.commit_sha,
        },
        "allowed_materialized_splits": ["train", "validation"],
        "test_subset_materialized": False,
    }
    write_metrics_json(artifacts.resolved_config_json, resolved_config)
    write_metrics_json(artifacts.environment_json, environment)
    _atomic_write_json(artifacts.run_manifest_json, base_manifest)
    logger.write("run started; test gate is closed")

    try:
        data = _validated_tuning_data(
            values,
            config_path=config_path,
            data_path=data_path,
            output_root=output_root,
            split_config=split_config,
            train_seed=train_seed,
        )
        _verify_data_hashes(data, expected_hashes)
        logger.write("data, split, and train-only preprocessing hashes verified")
        from .train import (
            build_checkpoint_provenance,
            fit_cell_msca,
            load_selected_checkpoint,
            save_selected_checkpoint,
        )

        fitted = fit_cell_msca(data, training_config)
        if fitted.contract.config_sha256 != configuration_sha256:
            raise ValueError("trainer configuration hash differs from resolved config")
        evaluation = evaluate_fitted_baseline(
            fitted,
            data.validation,
            data.provenance,
        )
        checkpoint_provenance = build_checkpoint_provenance(
            data,
            training_config,
            configuration_sha256=configuration_sha256,
            device=fitted.device,
            git_commit_sha=identity.commit_sha,
            git_worktree_dirty=identity.worktree_dirty,
            git_dirty_state_policy=identity.dirty_state_policy,
        )
        validation_mae = evaluation.result["headline_metrics"]["original_unit"][
            "mae"
        ]
        save_selected_checkpoint(
            artifacts.selected_checkpoint,
            model=fitted.model,
            config=training_config,
            contract=fitted.contract,
            provenance=checkpoint_provenance,
            best_epoch=fitted.contract.best_epoch or 1,
            validation_original_mae=float(validation_mae),
            parameter_count=fitted.parameter_count,
        )
        load_selected_checkpoint(
            artifacts.selected_checkpoint,
            device="cpu",
            expected_hashes={
                "data_sha256": data.provenance.data_sha256,
                "split_sha256": data.provenance.split_sha256,
                "preprocessing_sha256": data.provenance.preprocessing_sha256,
                "configuration_sha256": configuration_sha256,
            },
        )
        write_prediction_csv(
            artifacts.validation_predictions_csv,
            y_true_original=data.validation.target_original,
            y_pred_original=evaluation.predictions.pred_original,
            y_true_log=data.validation.target_log,
            y_pred_log=evaluation.predictions.pred_log,
            cell_ids=data.validation.cell_ids,
        )
        verify_baseline_result_from_prediction_csv(
            artifacts.validation_predictions_csv,
            evaluation.result,
        )
        metrics_payload = {
            "schema_version": "cell_msca.validation_metrics_artifact.v1",
            "artifact_classification": "validation-only",
            "experiment_id": experiment_id,
            "prediction_metric_verification": "passed_including_spearman",
            "result": evaluation.result,
        }
        write_metrics_json(artifacts.validation_metrics_json, metrics_payload)
        completed_manifest = {
            **base_manifest,
            "status": "completed",
            "completion_time_utc": _utc_now(),
            "parameter_count": fitted.parameter_count,
            "best_epoch": fitted.contract.best_epoch,
            "inverse_mode": fitted.contract.inverse_mode,
            "smearing_factor": fitted.contract.smearing_factor,
            "validation_metrics": evaluation.result,
        }
        _atomic_write_json(artifacts.run_manifest_json, completed_manifest)
        logger.write("run completed; validation metrics round-trip verified")
    except BaseException as error:
        failed_manifest = {
            **base_manifest,
            "status": "failed",
            "completion_time_utc": _utc_now(),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "test_subset_materialized": False,
        }
        _atomic_write_json(artifacts.run_manifest_json, failed_manifest)
        logger.write(f"run failed: {type(error).__name__}: {error}")
        raise
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen Cell-MSCA validation-only experiment",
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda", "auto"))
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--source-git-sha-file", type=Path)
    parser.add_argument("--source-tree-manifest", type=Path)
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--kaggle", action="store_true")
    arguments = parser.parse_args(argv)
    artifacts = run_kaggle_validation(
        variant=arguments.variant,
        train_seed=arguments.train_seed,
        config_path=arguments.config,
        data_path=arguments.data_path,
        output_path=arguments.output_path,
        device=arguments.device,
        expected_git_sha=arguments.expected_git_sha,
        kaggle=arguments.kaggle,
        source_git_sha_file=arguments.source_git_sha_file,
        source_tree_manifest=arguments.source_tree_manifest,
        expected_source_manifest_sha256=(
            arguments.expected_source_manifest_sha256
        ),
        repository_root=arguments.repository_root,
    )
    print(
        json.dumps(
            {
                "experiment_id": artifacts.experiment_id,
                "output_dir": str(artifacts.output_dir),
                "run_manifest": str(artifacts.run_manifest_json),
                "test_subset_materialized": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
