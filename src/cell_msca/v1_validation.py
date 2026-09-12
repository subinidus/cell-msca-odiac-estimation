"""Frozen v1_legacy archive verification and validation-only adapters.

This module orchestrates existing Phase 3/4 data, model, selection, and
evaluation APIs. It does not create data, alter persistent splits, or expose a
test-materialization path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping

from .baselines import (
    LIGHTGBM_MODEL_NAMES,
    BaselineDataProtocol,
    LightGBMConfig,
    TrainMeanConfig,
    fit_lightgbm_baseline,
    fit_train_mean,
    tune_on_validation,
)
from .data import (
    CellDataset,
    canonical_npz_content_sha256,
    file_sha256,
)
from .evaluate import (
    verify_baseline_result_from_prediction_csv,
    write_metrics_json,
    write_prediction_csv,
)
from .experiment import BaselineExperimentConfig, load_baseline_experiment_config
from .kaggle_runner import (
    load_experiment_assignments,
    resolve_git_identity,
    run_kaggle_validation,
)
from .neural_baselines import ConcatMLPConfig, fit_concat_mlp
from .splits import CellFixedSplitConfig, load_persistent_split

V1_ARCHIVE_MANIFEST_SCHEMA_VERSION = "cell_msca.v1_legacy_archive_manifest.v1"
BASELINE_ADAPTER_SCHEMA_VERSION = "cell_msca.baseline_experiment.v1"
V1_ARCHIVE_NAME = "2022_2024_npz.zip"
V1_BASELINE_MODELS = (
    "train_mean",
    "lightgbm_raw",
    "lightgbm_log1p",
    "lightgbm_tweedie",
    "concat_mlp_log1p",
)
_MONTH_FILE_PATTERN = re.compile(
    r"^ms_mca_tensor_delhi_ncr_(?P<year>202[2-4])_(?P<month>0[1-9]|1[0-2])\.npz$"
)


@dataclass(frozen=True)
class VerifiedV1Archive:
    archive_path: Path | None
    data_dir: Path
    protocol: BaselineDataProtocol
    summary: Mapping[str, Any]
    kaggle: bool
    input_mode: str = "archive"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return values


def discover_v1_archive(input_root: str | Path) -> Path:
    """Find exactly one legacy archive without assuming a Kaggle dataset slug."""

    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {root}")
    matches = sorted(path for path in root.rglob(V1_ARCHIVE_NAME) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {V1_ARCHIVE_NAME} below {root}; found {len(matches)}"
        )
    return matches[0]


def _safe_archive_entries(
    archive: zipfile.ZipFile,
    expected_files: Mapping[str, Mapping[str, Any]],
) -> list[zipfile.ZipInfo]:
    entries = [entry for entry in archive.infolist() if not entry.is_dir()]
    names: list[str] = []
    for entry in entries:
        relative = PurePosixPath(entry.filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"unsafe or nested ZIP entry: {entry.filename!r}")
        names.append(relative.name)
    if len(names) != len(set(names)):
        raise ValueError("archive contains duplicate file names")
    if set(names) != set(expected_files):
        missing = sorted(set(expected_files) - set(names))
        unexpected = sorted(set(names) - set(expected_files))
        raise ValueError(
            f"archive file set mismatch: missing={missing}, unexpected={unexpected}"
        )
    for entry in entries:
        name = PurePosixPath(entry.filename.replace("\\", "/")).name
        expected_size = int(expected_files[name]["size_bytes"])
        if entry.file_size != expected_size:
            raise ValueError(
                f"ZIP entry size mismatch for {name}: "
                f"expected={expected_size}, actual={entry.file_size}"
            )
    return entries


def _validate_month_coverage(file_names: list[str]) -> None:
    months: list[str] = []
    for name in file_names:
        match = _MONTH_FILE_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"unexpected v1_legacy NPZ file name: {name!r}")
        months.append(f"{match.group('year')}-{match.group('month')}")
    expected = [
        f"{year}-{month:02d}"
        for year in range(2022, 2025)
        for month in range(1, 13)
    ]
    if sorted(months) != expected:
        missing = sorted(set(expected) - set(months))
        duplicates = sorted({month for month in months if months.count(month) > 1})
        raise ValueError(
            f"archive month coverage mismatch: missing={missing}, duplicates={duplicates}"
        )


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
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


def _validate_kaggle_roots(input_root: Path, working_root: Path) -> None:
    input_posix = PurePosixPath(str(input_root).replace("\\", "/"))
    working_posix = PurePosixPath(str(working_root).replace("\\", "/"))
    kaggle_input = PurePosixPath("/kaggle/input")
    kaggle_working = PurePosixPath("/kaggle/working")
    try:
        input_posix.relative_to(kaggle_input)
    except ValueError as error:
        raise ValueError("Kaggle input_root must be below /kaggle/input") from error
    try:
        working_posix.relative_to(kaggle_working)
    except ValueError as error:
        raise ValueError("Kaggle working_root must be below /kaggle/working") from error


def _validate_kaggle_output(output_root: Path) -> None:
    output_posix = PurePosixPath(str(output_root).replace("\\", "/"))
    try:
        output_posix.relative_to(PurePosixPath("/kaggle/working"))
    except ValueError as error:
        raise ValueError("Kaggle output_root must be below /kaggle/working") from error


def _load_v1_archive_manifest(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != V1_ARCHIVE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported v1_legacy archive manifest schema_version")
    archive_contract = manifest.get("archive")
    if not isinstance(archive_contract, dict):
        raise ValueError("archive manifest requires an archive object")
    file_entries = manifest.get("files")
    if not isinstance(file_entries, list) or len(file_entries) != 36:
        raise ValueError("archive manifest must contain exactly 36 NPZ entries")
    expected_files = {
        str(entry["file_name"]): entry
        for entry in file_entries
        if isinstance(entry, dict) and "file_name" in entry
    }
    if len(expected_files) != 36:
        raise ValueError("archive manifest contains invalid or duplicate NPZ entries")
    _validate_month_coverage(list(expected_files))
    return manifest, archive_contract, expected_files


def _verify_v1_npz_files(
    paths: list[Path],
    expected_files: Mapping[str, Mapping[str, Any]],
) -> None:
    actual_names = [path.name for path in paths]
    if len(actual_names) != len(set(actual_names)):
        raise ValueError("expanded v1_legacy input contains duplicate NPZ names")
    if set(actual_names) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(expected_files))
        raise ValueError(
            f"expanded NPZ file set mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    for path in paths:
        expected = expected_files[path.name]
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"NPZ size mismatch: {path.name}")
        if file_sha256(path) != expected["file_sha256"]:
            raise ValueError(f"NPZ file SHA-256 mismatch: {path.name}")
        if canonical_npz_content_sha256(path) != expected["canonical_content_sha256"]:
            raise ValueError(f"NPZ canonical content SHA-256 mismatch: {path.name}")


def _build_verified_v1(
    *,
    paths: list[Path],
    data_dir: Path,
    repository: Path,
    manifest: Mapping[str, Any],
    archive_contract: Mapping[str, Any],
    archive_path: Path | None,
    train_seed: int | None,
    kaggle: bool,
    input_mode: str,
) -> VerifiedV1Archive:
    data_contract = manifest["data_contract"]
    dataset = CellDataset(
        paths,
        target_scale=float(data_contract["target_scale"]),
        legacy_data_version=str(data_contract["data_version"]),
        legacy_target_unit=str(data_contract["target_unit"]),
    )
    split_contract = manifest["split_contract"]
    split_config = CellFixedSplitConfig(
        split_seed=int(split_contract["split_seed"]),
        validation_ratio=float(split_contract["validation_ratio"]),
        test_ratio=float(split_contract["test_ratio"]),
    )
    split_csv = repository / str(split_contract["csv"])
    split_metadata = repository / str(split_contract["metadata_json"])
    split_manifest = load_persistent_split(
        dataset,
        split_csv,
        metadata_json=split_metadata,
        config=split_config,
    )
    resolved_train_seed = (
        int(manifest["train_seed"]) if train_seed is None else int(train_seed)
    )
    protocol = BaselineDataProtocol.from_dataset(
        dataset,
        split_csv,
        split_metadata,
        split_config=split_config,
        train_seed=resolved_train_seed,
    )
    expected_hashes = manifest["required_hashes"]
    actual_hashes = {
        "data_sha256": protocol.provenance.data_sha256,
        "split_sha256": protocol.provenance.split_sha256,
        "split_config_sha256": protocol.provenance.split_config_sha256,
        "preprocessing_sha256": protocol.provenance.preprocessing_sha256,
    }
    for name, actual in actual_hashes.items():
        if actual != expected_hashes[name]:
            raise ValueError(
                f"{name} mismatch: expected={expected_hashes[name]}, actual={actual}"
            )
    summary = {
        "input_mode": input_mode,
        "archive_file_sha256": archive_contract["file_sha256"],
        "archive_file_sha256_verified": archive_path is not None,
        "per_file_sha256_verified": True,
        "per_file_canonical_content_sha256_verified": True,
        "npz_count": len(paths),
        "train_seed": resolved_train_seed,
        **actual_hashes,
        "cell_counts": {
            name: len(split_manifest.cell_ids(name))
            for name in ("train", "validation", "test")
        },
        "test_assignment_integrity_verified": True,
        "test_sample_arrays_materialized": False,
        "temporary_data_cleanup": (
            "automatic_on_context_exit" if input_mode == "archive" else "not_applicable"
        ),
    }
    return VerifiedV1Archive(
        archive_path,
        data_dir,
        protocol,
        summary,
        kaggle,
        input_mode,
    )


def _expanded_v1_directories(
    input_root: Path,
    expected_files: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    first_name = sorted(expected_files)[0]
    candidates = {
        path.parent.resolve()
        for path in input_root.rglob(first_name)
        if path.is_file()
    }
    matches: list[Path] = []
    for candidate in candidates:
        names = {path.name for path in candidate.glob("*.npz") if path.is_file()}
        if names == set(expected_files):
            matches.append(candidate)
    return sorted(matches)


@contextmanager
def verified_v1_archive(
    *,
    input_root: str | Path,
    working_root: str | Path,
    manifest_path: str | Path,
    repository_root: str | Path,
    kaggle: bool = False,
    train_seed: int | None = None,
) -> Iterator[VerifiedV1Archive]:
    """Verify, temporarily extract, and automatically clean the legacy archive."""

    input_path = Path(input_root).resolve()
    working_path = Path(working_root).resolve()
    repository = Path(repository_root).resolve()
    if kaggle:
        _validate_kaggle_roots(input_path, working_path)
    working_path.mkdir(parents=True, exist_ok=True)
    archive_path = discover_v1_archive(input_path)
    manifest, archive_contract, expected_files = _load_v1_archive_manifest(
        manifest_path
    )
    if archive_path.name != archive_contract.get("file_name"):
        raise ValueError("archive name does not match the frozen manifest")
    if archive_path.stat().st_size != int(archive_contract.get("size_bytes", -1)):
        raise ValueError("archive size does not match the frozen manifest")
    if file_sha256(archive_path) != archive_contract.get("file_sha256"):
        raise ValueError("archive SHA-256 does not match the frozen manifest")
    with TemporaryDirectory(prefix="cell-msca-v1-", dir=working_path) as directory:
        data_dir = Path(directory)
        with zipfile.ZipFile(archive_path, "r") as archive:
            entries = _safe_archive_entries(archive, expected_files)
            for entry in entries:
                destination = data_dir / PurePosixPath(
                    entry.filename.replace("\\", "/")
                ).name
                with archive.open(entry, "r") as source, destination.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
        paths = sorted(data_dir.glob("*.npz"))
        _verify_v1_npz_files(paths, expected_files)
        yield _build_verified_v1(
            paths=paths,
            data_dir=data_dir,
            repository=repository,
            manifest=manifest,
            archive_contract=archive_contract,
            archive_path=archive_path,
            train_seed=train_seed,
            kaggle=kaggle,
            input_mode="archive",
        )


@contextmanager
def verified_v1_input(
    *,
    input_root: str | Path,
    working_root: str | Path,
    manifest_path: str | Path,
    repository_root: str | Path,
    kaggle: bool = False,
    train_seed: int | None = None,
) -> Iterator[VerifiedV1Archive]:
    """Verify one archive or one Kaggle-expanded 36-NPZ directory."""

    input_path = Path(input_root).resolve()
    working_path = Path(working_root).resolve()
    repository = Path(repository_root).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_path}")
    if kaggle:
        _validate_kaggle_roots(input_path, working_path)
    manifest, archive_contract, expected_files = _load_v1_archive_manifest(
        manifest_path
    )
    archives = sorted(
        path.resolve()
        for path in input_path.rglob(V1_ARCHIVE_NAME)
        if path.is_file()
    )
    expanded = _expanded_v1_directories(input_path, expected_files)
    available_modes = int(bool(archives)) + int(bool(expanded))
    if len(archives) > 1 or len(expanded) > 1 or available_modes != 1:
        raise RuntimeError(
            "expected exactly one v1_legacy input mode; "
            f"archives={len(archives)}, expanded_directories={len(expanded)}"
        )
    if archives:
        with verified_v1_archive(
            input_root=input_path,
            working_root=working_path,
            manifest_path=manifest_path,
            repository_root=repository,
            kaggle=kaggle,
            train_seed=train_seed,
        ) as verified:
            yield verified
        return

    data_dir = expanded[0]
    paths = sorted(data_dir.glob("*.npz"))
    _verify_v1_npz_files(paths, expected_files)
    yield _build_verified_v1(
        paths=paths,
        data_dir=data_dir,
        repository=repository,
        manifest=manifest,
        archive_contract=archive_contract,
        archive_path=None,
        train_seed=train_seed,
        kaggle=kaggle,
        input_mode="expanded_npz",
    )


def _load_frozen_baseline_values(
    path: str | Path,
) -> tuple[BaselineExperimentConfig, dict[str, Any]]:
    config = load_baseline_experiment_config(path)
    values = _read_json(path)
    if values.get("schema_version") != BASELINE_ADAPTER_SCHEMA_VERSION:
        raise ValueError("unsupported frozen baseline configuration")
    if values.get("allowed_materialized_splits") != ["train", "validation"]:
        raise ValueError("frozen baseline config must allow train and validation only")
    required_hashes = values.get("required_hashes")
    if not isinstance(required_hashes, dict):
        raise ValueError("frozen baseline config requires exact hashes")
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(required_hashes.get(name, ""))):
            raise ValueError(f"invalid frozen hash: {name}")
    experiment_ids = values.get("experiment_ids")
    if not isinstance(experiment_ids, dict) or set(experiment_ids) != set(
        V1_BASELINE_MODELS
    ):
        raise ValueError("frozen baseline config has incomplete experiment IDs")
    owners = values.get("owners_by_model")
    if not isinstance(owners, dict) or set(owners) != set(V1_BASELINE_MODELS):
        raise ValueError("frozen baseline config has incomplete model owners")
    if not all(str(owner).strip() for owner in owners.values()):
        raise ValueError("frozen baseline model owners must not be empty")
    for model_name, candidates in config.models.items():
        if len(candidates) != 1:
            raise ValueError(f"frozen baseline {model_name} requires one candidate")
    return config, values


def load_frozen_v1_assignments(path: str | Path) -> dict[str, Any]:
    """Load the ten frozen assignments and reject duplicate output locations."""

    values = load_experiment_assignments(path)
    assignments = values["assignments"]
    expected_ids = {
        "train_mean_seed42",
        "lightgbm_raw_seed42",
        "lightgbm_log1p_seed42",
        "lightgbm_tweedie_seed42",
        "concat_mlp_raw_huber_seed42",
        "concat_mlp_log1p_seed42",
        "token_no_attention_seed42",
        "cell_msca_forward_seed42",
        "cell_msca_reverse_seed42",
        "cell_msca_bidirectional_seed42",
    }
    actual_ids = {str(assignment["experiment_id"]) for assignment in assignments}
    if actual_ids != expected_ids:
        raise ValueError("v1_legacy assignment IDs differ from the frozen contract")
    output_locations = [str(item["output_location"]) for item in assignments]
    duplicates = sorted(
        location
        for location in set(output_locations)
        if output_locations.count(location) > 1
    )
    if duplicates:
        raise ValueError(
            f"duplicate assignment output locations are not allowed: {duplicates}"
        )
    unsupported = [
        item for item in assignments if str(item["status"]) == "unsupported"
    ]
    if len(unsupported) != 1 or unsupported[0]["experiment_id"] != (
        "concat_mlp_raw_huber_seed42"
    ):
        raise ValueError(
            "only concat_mlp_raw_huber_seed42 may be unsupported in this package"
        )
    return values


def _fit_single_baseline(data: Any, config: Any, model_name: str) -> Any:
    if model_name == "train_mean":
        candidates = [TrainMeanConfig(**config.models["train_mean"][0])]
        return tune_on_validation(data, candidates, fit_train_mean)
    if model_name in LIGHTGBM_MODEL_NAMES:
        candidates = [
            LightGBMConfig(model_name=model_name, **config.models[model_name][0])
        ]
        return tune_on_validation(data, candidates, fit_lightgbm_baseline)
    if model_name == "concat_mlp_log1p":
        candidates = [ConcatMLPConfig(**config.models["concat_mlp"][0])]
        return tune_on_validation(data, candidates, fit_concat_mlp)
    raise ValueError(f"unsupported baseline adapter model: {model_name!r}")


def run_frozen_baseline_validation(
    verified: VerifiedV1Archive,
    *,
    model_name: str,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_git_sha: str,
    allow_full_validation: bool = False,
) -> Path:
    """Run one existing Phase 3 model on train/validation only."""

    if model_name not in V1_BASELINE_MODELS:
        raise ValueError(f"model_name must be one of {V1_BASELINE_MODELS}")
    config, values = _load_frozen_baseline_values(config_path)
    execution = values.get("execution_policy", {})
    if execution.get("default_action") == "integrity_only" and not allow_full_validation:
        raise RuntimeError(
            "full validation is disabled by default; pass allow_full_validation=True"
        )
    expected_hashes = values["required_hashes"]
    provenance = verified.protocol.provenance
    for name in (
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
    ):
        if getattr(provenance, name) != expected_hashes[name]:
            raise ValueError(f"verified data differs from baseline config: {name}")
    identity = resolve_git_identity(
        repository_root,
        expected_git_sha=expected_git_sha,
    )
    if identity.worktree_dirty:
        raise RuntimeError("baseline adapter requires a clean Git source tree")
    experiment_id = str(values["experiment_ids"][model_name])
    resolved_output_root = Path(output_root).resolve()
    if verified.kaggle:
        _validate_kaggle_output(resolved_output_root)
    if resolved_output_root == verified.data_dir or verified.data_dir in (
        resolved_output_root.parents
    ):
        raise ValueError("output_root must not be inside the temporary data directory")
    output_dir = resolved_output_root / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    data = verified.protocol.tuning_data()
    selection = _fit_single_baseline(data, config, model_name)
    evaluation = selection.validation_evaluation
    prediction_path = output_dir / "validation_predictions.csv"
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
    verify_baseline_result_from_prediction_csv(prediction_path, evaluation.result)
    write_metrics_json(
        metrics_path,
        {
            "artifact_classification": "validation-only",
            "prediction_metric_verification": "passed_including_spearman",
            "result": evaluation.result,
        },
    )
    environment = _runtime_environment()
    write_metrics_json(environment_path, environment)
    write_metrics_json(
        manifest_path,
        {
            "schema_version": "cell_msca.v1_baseline_run_manifest.v1",
            "artifact_classification": "validation-only",
            "status": "completed",
            "experiment_id": experiment_id,
            "owner": str(values["owners_by_model"][model_name]),
            "model_name": model_name,
            "data_version": "v1_legacy",
            "data_sha256": provenance.data_sha256,
            "split_sha256": provenance.split_sha256,
            "split_config_sha256": provenance.split_config_sha256,
            "preprocessing_sha256": provenance.preprocessing_sha256,
            "config_sha256": evaluation.result["config_sha256"],
            "config_file_sha256": file_sha256(config_path),
            "git_commit_sha": identity.commit_sha,
            "git_dirty_state_policy": identity.dirty_state_policy,
            "split_seed": provenance.split_seed,
            "train_seed": provenance.train_seed,
            "selection_metric": "validation_original_unit_mae",
            "allowed_materialized_splits": ["train", "validation"],
            "test_subset_materialized": False,
            "runtime_versions": environment,
            "completion_time_utc": _utc_now(),
            "validation_metrics": evaluation.result,
            "model_artifact_status": (
                "not_saved_by_existing_phase3_baseline_checkpoint_contract"
            ),
        },
    )
    return output_dir


def run_frozen_cell_validation(
    verified: VerifiedV1Archive,
    *,
    variant: str,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_git_sha: str,
    device: str,
    allow_full_validation: bool = False,
) -> Path:
    """Run the existing Phase 4.3 runner against a verified temporary extract."""

    values = _read_json(config_path)
    execution = values.get("execution_policy", {})
    if execution.get("default_action") == "integrity_only" and not allow_full_validation:
        raise RuntimeError(
            "full validation is disabled by default; pass allow_full_validation=True"
        )
    if verified.kaggle:
        _validate_kaggle_output(Path(output_root).resolve())
    artifacts = run_kaggle_validation(
        variant=variant,
        train_seed=int(values["frozen_train_seed"]),
        config_path=config_path,
        data_path=verified.data_dir,
        output_path=output_root,
        device=device,
        expected_git_sha=expected_git_sha,
        kaggle=False,
        repository_root=repository_root,
    )
    return artifacts.output_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and orchestrate frozen v1_legacy validation inputs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    integrity = subparsers.add_parser("integrity")
    baseline = subparsers.add_parser("baseline")
    cell = subparsers.add_parser("cell")
    for command in (integrity, baseline, cell):
        command.add_argument("--input-root", required=True, type=Path)
        command.add_argument("--working-root", required=True, type=Path)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--repository-root", required=True, type=Path)
        command.add_argument("--kaggle", action="store_true")
    baseline.add_argument("--model", required=True, choices=V1_BASELINE_MODELS)
    baseline.add_argument("--config", required=True, type=Path)
    baseline.add_argument("--output-root", required=True, type=Path)
    baseline.add_argument("--expected-git-sha", required=True)
    baseline.add_argument("--allow-full-validation", action="store_true")
    cell.add_argument("--variant", required=True)
    cell.add_argument("--config", required=True, type=Path)
    cell.add_argument("--output-root", required=True, type=Path)
    cell.add_argument("--expected-git-sha", required=True)
    cell.add_argument("--device", required=True, choices=("cpu", "cuda", "auto"))
    cell.add_argument("--allow-full-validation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    with verified_v1_archive(
        input_root=arguments.input_root,
        working_root=arguments.working_root,
        manifest_path=arguments.manifest,
        repository_root=arguments.repository_root,
        kaggle=arguments.kaggle,
    ) as verified:
        if arguments.command == "integrity":
            print(json.dumps(verified.summary, indent=2, allow_nan=False))
            return 0
        if arguments.command == "baseline":
            output_dir = run_frozen_baseline_validation(
                verified,
                model_name=arguments.model,
                config_path=arguments.config,
                output_root=arguments.output_root,
                repository_root=arguments.repository_root,
                expected_git_sha=arguments.expected_git_sha,
                allow_full_validation=arguments.allow_full_validation,
            )
        else:
            output_dir = run_frozen_cell_validation(
                verified,
                variant=arguments.variant,
                config_path=arguments.config,
                output_root=arguments.output_root,
                repository_root=arguments.repository_root,
                expected_git_sha=arguments.expected_git_sha,
                device=arguments.device,
                allow_full_validation=arguments.allow_full_validation,
            )
        print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
