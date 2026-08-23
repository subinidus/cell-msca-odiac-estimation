"""Persistent cell-fixed splits and isolated buffered-block robustness splits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import CellDataset, CellLocation, canonical_sha256, file_sha256

PRIMARY_SPLITS = ("train", "validation", "test")
BLOCK_SPLITS = (*PRIMARY_SPLITS, "dropped")
SPLIT_COLUMNS = ("row", "col", "cell_id", "split")
SPLIT_SCHEMA_VERSION = "cell_msca.split.v2"
BUFFERED_BLOCK_RESEARCH_STATUS = {
    "protocol_status": "development_only_not_research_frozen",
    "unfrozen_default_parameters": {
        "block_size": 24,
        "buffer_cells": 16,
    },
    "note": (
        "block_size=24 and buffer_cells=16 are development settings; "
        "they are not yet research-frozen protocol choices"
    ),
}


@dataclass(frozen=True)
class SeedConfig:
    """Keep persistent data assignment and stochastic training seeds separate."""

    split_seed: int = 42
    train_seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
        if not isinstance(self.train_seed, int):
            raise ValueError("train_seed must be an integer")


@dataclass(frozen=True)
class CellFixedSplitConfig:
    split_seed: int = 42
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        _validate_ratios(self.validation_ratio, self.test_ratio)
        if not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")

    @classmethod
    def from_seeds(
        cls,
        seeds: SeedConfig,
        *,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> CellFixedSplitConfig:
        """Build split configuration from split_seed only, excluding train_seed."""

        return cls(
            split_seed=seeds.split_seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_mode": "cell_fixed",
            "split_seed": self.split_seed,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
        }


@dataclass(frozen=True)
class BufferedBlockSplitConfig:
    """Robustness split configuration with development-only defaults.

    The default ``block_size=24`` and ``buffer_cells=16`` remain research-
    unfrozen development settings and must not be treated as final protocol
    choices.
    """

    split_seed: int = 42
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    block_size: int = 24
    buffer_cells: int = 16

    def __post_init__(self) -> None:
        _validate_ratios(self.validation_ratio, self.test_ratio)
        if not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
        if not isinstance(self.block_size, int) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not isinstance(self.buffer_cells, int) or self.buffer_cells < 0:
            raise ValueError("buffer_cells must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_mode": "buffered_block",
            "split_seed": self.split_seed,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "block_size": self.block_size,
            "buffer_cells": self.buffer_cells,
        }


@dataclass(frozen=True)
class SplitAssignment:
    row: int
    col: int
    cell_id: str
    split: str


@dataclass(frozen=True)
class SplitManifest:
    assignments: tuple[SplitAssignment, ...]
    split_mode: str
    split_sha256: str
    source_path: Path

    def cell_ids(self, split: str) -> frozenset[str]:
        return frozenset(
            assignment.cell_id
            for assignment in self.assignments
            if assignment.split == split
        )

    def sample_indices(self, dataset: CellDataset, split: str) -> list[int]:
        return dataset.sample_indices_for_cells(self.cell_ids(split))


class CellSubset:
    """Minimal dependency-free view over sample indices for one split."""

    def __init__(self, dataset: CellDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


def build_split_subsets(
    dataset: CellDataset,
    manifest: SplitManifest,
) -> dict[str, CellSubset]:
    """Create train/validation/test dataset views from a reloaded manifest."""

    return {
        split: CellSubset(dataset, manifest.sample_indices(dataset, split))
        for split in PRIMARY_SPLITS
    }


def _validate_ratios(validation_ratio: float, test_ratio: float) -> None:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1")


def _heldout_counts(
    total: int,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    if total < 3:
        raise ValueError("at least three cells or occupied blocks are required")
    n_validation = max(1, round(total * validation_ratio))
    n_test = max(1, round(total * test_ratio))
    if n_validation + n_test >= total:
        raise ValueError(
            "split ratios leave no training unit: "
            f"total={total}, validation={n_validation}, test={n_test}"
        )
    return n_validation, n_test


def _cell_fixed_assignments(
    dataset: CellDataset,
    config: CellFixedSplitConfig,
) -> tuple[SplitAssignment, ...]:
    cells = dataset.cell_locations
    n_validation, n_test = _heldout_counts(
        len(cells),
        config.validation_ratio,
        config.test_ratio,
    )
    rng = np.random.default_rng(config.split_seed)
    order = rng.permutation(len(cells))
    validation_indices = set(order[:n_validation].tolist())
    test_indices = set(order[n_validation : n_validation + n_test].tolist())
    assignments = []
    for index, cell in enumerate(cells):
        if index in validation_indices:
            split = "validation"
        elif index in test_indices:
            split = "test"
        else:
            split = "train"
        assignments.append(SplitAssignment(cell.row, cell.col, cell.cell_id, split))
    return tuple(sorted(assignments, key=lambda item: (item.row, item.col)))


def create_persistent_cell_fixed_split(
    dataset: CellDataset,
    split_csv: str | Path,
    metadata_json: str | Path,
    *,
    config: CellFixedSplitConfig | None = None,
) -> SplitManifest:
    """Create a persistent cell-fixed split without overwriting artifacts."""

    config = config or CellFixedSplitConfig()
    assignments = _cell_fixed_assignments(dataset, config)
    return _persist_split(
        dataset,
        assignments,
        split_csv=Path(split_csv),
        metadata_json=Path(metadata_json),
        config=config.to_dict(),
        split_mode="cell_fixed",
    )


def _block_of(cell: CellLocation, block_size: int) -> tuple[int, int]:
    return cell.row // block_size, cell.col // block_size


def _minimum_chebyshev_distance(
    first_cells: list[CellLocation],
    second_cells: list[CellLocation],
) -> int | None:
    if not first_cells or not second_cells:
        return None
    first = np.asarray(
        [(cell.row, cell.col) for cell in first_cells],
        dtype=np.int64,
    )
    second = np.asarray(
        [(cell.row, cell.col) for cell in second_cells],
        dtype=np.int64,
    )
    minimum: int | None = None
    for start in range(0, first.shape[0], 256):
        chunk = first[start : start + 256]
        distances = np.max(np.abs(chunk[:, None, :] - second[None, :, :]), axis=2)
        chunk_minimum = int(np.min(distances))
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
    return minimum


def _buffered_block_assignments(
    dataset: CellDataset,
    config: BufferedBlockSplitConfig,
) -> tuple[tuple[SplitAssignment, ...] | None, dict[str, Any]]:
    cells = dataset.cell_locations
    occupied_blocks = sorted({_block_of(cell, config.block_size) for cell in cells})
    try:
        n_validation, n_test = _heldout_counts(
            len(occupied_blocks),
            config.validation_ratio,
            config.test_ratio,
        )
    except ValueError as error:
        return None, {
            "feasible": False,
            "reason": str(error),
            "occupied_block_count": len(occupied_blocks),
        }

    rng = np.random.default_rng(config.split_seed)
    order = rng.permutation(len(occupied_blocks))
    validation_blocks = {
        occupied_blocks[index] for index in order[:n_validation].tolist()
    }
    test_blocks = {
        occupied_blocks[index]
        for index in order[n_validation : n_validation + n_test].tolist()
    }

    validation_cells = [
        cell for cell in cells if _block_of(cell, config.block_size) in validation_blocks
    ]
    test_cells = [
        cell for cell in cells if _block_of(cell, config.block_size) in test_blocks
    ]
    heldout_cells = validation_cells + test_cells
    candidate_train = [
        cell
        for cell in cells
        if _block_of(cell, config.block_size) not in validation_blocks | test_blocks
    ]

    max_row = max(cell.row for cell in cells)
    max_col = max(cell.col for cell in cells)
    heldout_buffer = np.zeros((max_row + 1, max_col + 1), dtype=bool)
    for cell in heldout_cells:
        row_start = max(0, cell.row - config.buffer_cells)
        row_end = min(max_row + 1, cell.row + config.buffer_cells + 1)
        col_start = max(0, cell.col - config.buffer_cells)
        col_end = min(max_col + 1, cell.col + config.buffer_cells + 1)
        heldout_buffer[row_start:row_end, col_start:col_end] = True

    train_cells = [
        cell for cell in candidate_train if not heldout_buffer[cell.row, cell.col]
    ]
    dropped_cells = [
        cell for cell in candidate_train if heldout_buffer[cell.row, cell.col]
    ]
    pairwise_minimum_distances = {
        "train_validation": _minimum_chebyshev_distance(
            train_cells,
            validation_cells,
        ),
        "train_test": _minimum_chebyshev_distance(train_cells, test_cells),
        "validation_test": _minimum_chebyshev_distance(
            validation_cells,
            test_cells,
        ),
    }
    train_heldout_distances = (
        pairwise_minimum_distances["train_validation"],
        pairwise_minimum_distances["train_test"],
    )
    minimum_distance = (
        min(distance for distance in train_heldout_distances if distance is not None)
        if all(distance is not None for distance in train_heldout_distances)
        else None
    )

    if not train_cells or not validation_cells or not test_cells:
        return None, {
            "feasible": False,
            "reason": "buffered assignment leaves an empty primary split",
            "occupied_block_count": len(occupied_blocks),
            "train_cell_count": len(train_cells),
            "validation_cell_count": len(validation_cells),
            "test_cell_count": len(test_cells),
            "dropped_cell_count": len(dropped_cells),
            "minimum_train_to_heldout_chebyshev_distance": minimum_distance,
            "minimum_chebyshev_distance_by_pair": pairwise_minimum_distances,
        }
    if minimum_distance is None or minimum_distance <= config.buffer_cells:
        return None, {
            "feasible": False,
            "reason": "minimum train-to-heldout distance does not exceed the buffer",
            "occupied_block_count": len(occupied_blocks),
            "minimum_train_to_heldout_chebyshev_distance": minimum_distance,
            "minimum_chebyshev_distance_by_pair": pairwise_minimum_distances,
        }

    split_of = {
        **{cell.cell_id: "train" for cell in train_cells},
        **{cell.cell_id: "validation" for cell in validation_cells},
        **{cell.cell_id: "test" for cell in test_cells},
        **{cell.cell_id: "dropped" for cell in dropped_cells},
    }
    assignments = tuple(
        SplitAssignment(cell.row, cell.col, cell.cell_id, split_of[cell.cell_id])
        for cell in cells
    )
    feasibility = {
        "feasible": True,
        "reason": "buffered block split has non-empty primary splits",
        "occupied_block_count": len(occupied_blocks),
        "train_cell_count": len(train_cells),
        "validation_cell_count": len(validation_cells),
        "test_cell_count": len(test_cells),
        "dropped_cell_count": len(dropped_cells),
        "minimum_train_to_heldout_chebyshev_distance": minimum_distance,
        "minimum_chebyshev_distance_by_pair": pairwise_minimum_distances,
    }
    return assignments, feasibility


def assess_buffered_block_split(
    dataset: CellDataset,
    *,
    config: BufferedBlockSplitConfig | None = None,
) -> dict[str, Any]:
    """Check whether a separate buffered-block robustness split is feasible."""

    _, feasibility = _buffered_block_assignments(
        dataset,
        config or BufferedBlockSplitConfig(),
    )
    return feasibility


def create_persistent_buffered_block_split(
    dataset: CellDataset,
    split_csv: str | Path,
    metadata_json: str | Path,
    *,
    config: BufferedBlockSplitConfig | None = None,
) -> SplitManifest:
    """Persist the robustness split through a path separate from cell-fixed."""

    config = config or BufferedBlockSplitConfig()
    assignments, feasibility = _buffered_block_assignments(dataset, config)
    if assignments is None:
        raise ValueError(f"buffered block split is not feasible: {feasibility}")
    return _persist_split(
        dataset,
        assignments,
        split_csv=Path(split_csv),
        metadata_json=Path(metadata_json),
        config=config.to_dict(),
        split_mode="buffered_block",
        extra_metadata={
            "buffer_feasibility": feasibility,
            "research_status": BUFFERED_BLOCK_RESEARCH_STATUS,
        },
    )


def _assignment_sha256(assignments: tuple[SplitAssignment, ...], split: str) -> str:
    rows = [
        {
            "row": assignment.row,
            "col": assignment.col,
            "cell_id": assignment.cell_id,
        }
        for assignment in assignments
        if assignment.split == split
    ]
    return canonical_sha256(rows)


def _split_metadata(
    dataset: CellDataset,
    assignments: tuple[SplitAssignment, ...],
    *,
    split_mode: str,
    split_sha256: str,
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    split_names = BLOCK_SPLITS if split_mode == "buffered_block" else PRIMARY_SPLITS
    split_counts = {}
    for split in split_names:
        cell_ids = frozenset(
            assignment.cell_id
            for assignment in assignments
            if assignment.split == split
        )
        split_counts[split] = {
            "cell_count": len(cell_ids),
            "sample_count": dataset.sample_count_for_cells(cell_ids),
            "assignment_sha256": _assignment_sha256(assignments, split),
        }
    total_cells = len(dataset.cell_locations)
    total_samples = len(dataset)
    ratio_split_names = BLOCK_SPLITS
    actual_cell_ratios = {
        split: sum(
            assignment.split == split for assignment in assignments
        ) / total_cells
        for split in ratio_split_names
    }
    dropped_cells = sum(
        assignment.split == "dropped" for assignment in assignments
    )
    metadata: dict[str, Any] = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_mode": split_mode,
        "split_seed": config["split_seed"],
        "split_sha256": split_sha256,
        "data_sha256": dataset.data_sha256,
        "config_sha256": canonical_sha256(config),
        "config": config,
        "feature_metadata": dataset.feature_metadata.to_dict(),
        "feature_metadata_sha256": dataset.feature_metadata.sha256,
        "data_manifest": list(dataset.data_manifest),
        "counts": {
            "total_cells": total_cells,
            "total_samples": total_samples,
            "by_split": split_counts,
            "actual_cell_ratios": actual_cell_ratios,
            "dropped_cell_ratio": dropped_cells / total_cells,
            "dropped_sample_ratio": (
                dataset.sample_count_for_cells(
                    frozenset(
                        assignment.cell_id
                        for assignment in assignments
                        if assignment.split == "dropped"
                    )
                )
                / total_samples
            ),
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _persist_split(
    dataset: CellDataset,
    assignments: tuple[SplitAssignment, ...],
    *,
    split_csv: Path,
    metadata_json: Path,
    config: dict[str, Any],
    split_mode: str,
    extra_metadata: dict[str, Any] | None = None,
) -> SplitManifest:
    for path in (split_csv, metadata_json):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing split artifact: {path}")
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPLIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for assignment in sorted(assignments, key=lambda item: (item.row, item.col)):
            writer.writerow(
                {
                    "row": assignment.row,
                    "col": assignment.col,
                    "cell_id": assignment.cell_id,
                    "split": assignment.split,
                }
            )
    split_sha256 = file_sha256(split_csv)
    metadata = _split_metadata(
        dataset,
        assignments,
        split_mode=split_mode,
        split_sha256=split_sha256,
        config=config,
        extra_metadata=extra_metadata,
    )
    with metadata_json.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return SplitManifest(assignments, split_mode, split_sha256, split_csv)


def _config_contract(
    config: CellFixedSplitConfig | BufferedBlockSplitConfig,
) -> tuple[str, dict[str, Any]]:
    if isinstance(config, CellFixedSplitConfig):
        return "cell_fixed", config.to_dict()
    if isinstance(config, BufferedBlockSplitConfig):
        return "buffered_block", config.to_dict()
    raise TypeError(
        "config must be CellFixedSplitConfig or BufferedBlockSplitConfig; "
        f"got {type(config).__name__}"
    )


def _expected_assignments_and_extra(
    dataset: CellDataset,
    config: CellFixedSplitConfig | BufferedBlockSplitConfig,
) -> tuple[tuple[SplitAssignment, ...], dict[str, Any] | None]:
    if isinstance(config, CellFixedSplitConfig):
        return _cell_fixed_assignments(dataset, config), None
    assignments, feasibility = _buffered_block_assignments(dataset, config)
    if assignments is None:
        raise ValueError(
            "current dataset/config cannot reproduce the buffered split: "
            f"{feasibility}"
        )
    return assignments, {
        "buffer_feasibility": feasibility,
        "research_status": BUFFERED_BLOCK_RESEARCH_STATUS,
    }


def _assignment_map(
    assignments: tuple[SplitAssignment, ...],
) -> dict[str, tuple[int, int, str]]:
    return {
        assignment.cell_id: (
            assignment.row,
            assignment.col,
            assignment.split,
        )
        for assignment in assignments
    }


def _require_config_assignments(
    actual: tuple[SplitAssignment, ...],
    expected: tuple[SplitAssignment, ...],
) -> None:
    if _assignment_map(actual) != _assignment_map(expected):
        raise ValueError(
            "split assignments do not match the current dataset and split config"
        )


def write_split_metadata_for_existing_split(
    dataset: CellDataset,
    manifest: SplitManifest,
    metadata_json: str | Path,
    *,
    config: CellFixedSplitConfig | BufferedBlockSplitConfig,
) -> Path:
    """Write a v2 sidecar for an existing, unchanged split CSV.

    The assignment is regenerated from the current dataset/config and compared
    before a new sidecar is written. Existing metadata is never replaced.
    """

    metadata_path = Path(metadata_json)
    if metadata_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing split artifact: {metadata_path}"
        )
    split_mode, config_dict = _config_contract(config)
    if manifest.split_mode != split_mode:
        raise ValueError(
            "manifest split mode does not match config: "
            f"manifest={manifest.split_mode!r}, config={split_mode!r}"
        )
    current_split_sha256 = file_sha256(manifest.source_path)
    if current_split_sha256 != manifest.split_sha256:
        raise ValueError("manifest split checksum does not match its source CSV")
    verify_split_integrity(dataset, manifest)
    expected_assignments, extra_metadata = _expected_assignments_and_extra(
        dataset,
        config,
    )
    _require_config_assignments(manifest.assignments, expected_assignments)
    metadata = _split_metadata(
        dataset,
        manifest.assignments,
        split_mode=split_mode,
        split_sha256=current_split_sha256,
        config=config_dict,
        extra_metadata=extra_metadata,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return metadata_path


def _validate_loaded_metadata(
    dataset: CellDataset,
    manifest: SplitManifest,
    metadata: dict[str, Any],
    config: CellFixedSplitConfig | BufferedBlockSplitConfig,
) -> None:
    split_mode, config_dict = _config_contract(config)
    checks = (
        (
            "schema_version",
            metadata.get("schema_version"),
            SPLIT_SCHEMA_VERSION,
        ),
        ("split_mode", metadata.get("split_mode"), split_mode),
        ("split_seed", metadata.get("split_seed"), config.split_seed),
        ("config", metadata.get("config"), config_dict),
        (
            "config_sha256",
            metadata.get("config_sha256"),
            canonical_sha256(config_dict),
        ),
        (
            "feature_metadata",
            metadata.get("feature_metadata"),
            dataset.feature_metadata.to_dict(),
        ),
        (
            "feature_metadata_sha256",
            metadata.get("feature_metadata_sha256"),
            dataset.feature_metadata.sha256,
        ),
        ("data_sha256", metadata.get("data_sha256"), dataset.data_sha256),
        ("data_manifest", metadata.get("data_manifest"), list(dataset.data_manifest)),
        ("split_sha256", metadata.get("split_sha256"), manifest.split_sha256),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"split metadata {field} does not match the current "
                "dataset/config"
            )

    expected_assignments, extra_metadata = _expected_assignments_and_extra(
        dataset,
        config,
    )
    _require_config_assignments(manifest.assignments, expected_assignments)
    expected_metadata = _split_metadata(
        dataset,
        manifest.assignments,
        split_mode=split_mode,
        split_sha256=manifest.split_sha256,
        config=config_dict,
        extra_metadata=extra_metadata,
    )
    if metadata.get("counts") != expected_metadata["counts"]:
        raise ValueError(
            "split metadata counts do not match the current dataset/config"
        )
    if split_mode == "buffered_block":
        for field in ("buffer_feasibility", "research_status"):
            if metadata.get(field) != expected_metadata[field]:
                raise ValueError(
                    f"split metadata {field} does not match the current "
                    "dataset/config"
                )


def load_persistent_split(
    dataset: CellDataset,
    split_csv: str | Path,
    *,
    metadata_json: str | Path | None = None,
    config: CellFixedSplitConfig | BufferedBlockSplitConfig | None = None,
    unsafe_allow_missing_metadata: bool = False,
) -> SplitManifest:
    """Reload a split through the safe paper-facing provenance contract.

    Safe loading requires both ``metadata_json`` and the caller's current
    ``config``. Metadata-free diagnostic loading is available only when
    ``unsafe_allow_missing_metadata=True`` is stated explicitly.
    """

    split_path = Path(split_csv)
    metadata: dict[str, Any] | None = None
    if metadata_json is None:
        if not unsafe_allow_missing_metadata:
            raise ValueError(
                "metadata_json is required for safe persistent split loading; "
                "set unsafe_allow_missing_metadata=True only for explicit "
                "diagnostic use"
            )
    else:
        if config is None:
            raise ValueError(
                "config is required with metadata_json so the saved split can "
                "be validated against the current split configuration"
            )
        with Path(metadata_json).open("r", encoding="utf-8") as stream:
            loaded_metadata = json.load(stream)
        if not isinstance(loaded_metadata, dict):
            raise ValueError("split metadata JSON root must be an object")
        metadata = loaded_metadata
    assignments: list[SplitAssignment] = []
    with split_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SPLIT_COLUMNS:
            raise ValueError(
                f"split CSV columns must be {SPLIT_COLUMNS}; got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                parsed_row = int(row["row"])
                parsed_col = int(row["col"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid row/col at split CSV row {row_number}") from error
            split = row["split"]
            if split not in BLOCK_SPLITS:
                raise ValueError(f"invalid split {split!r} at CSV row {row_number}")
            assignments.append(
                SplitAssignment(parsed_row, parsed_col, row["cell_id"], split)
            )

    ids = [assignment.cell_id for assignment in assignments]
    if len(ids) != len(set(ids)):
        raise ValueError("split CSV contains duplicate cell_id values")
    dataset_ids = set(dataset.cell_ids)
    if set(ids) != dataset_ids:
        missing = sorted(dataset_ids - set(ids))
        unexpected = sorted(set(ids) - dataset_ids)
        raise ValueError(
            "split CSV cell set differs from dataset: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for assignment in assignments:
        location = dataset.cell_location(assignment.cell_id)
        if (assignment.row, assignment.col) != (location.row, location.col):
            raise ValueError(
                f"cell location mismatch for {assignment.cell_id}: "
                f"CSV={(assignment.row, assignment.col)}, "
                f"dataset={(location.row, location.col)}"
            )

    split_sha256 = file_sha256(split_path)
    if metadata is not None:
        assert config is not None
        split_mode, _ = _config_contract(config)
    else:
        split_mode = "buffered_block" if "dropped" in {
            assignment.split for assignment in assignments
        } else "cell_fixed"
    manifest = SplitManifest(
        tuple(assignments),
        split_mode,
        split_sha256,
        split_path,
    )
    verify_split_integrity(dataset, manifest)

    if metadata is not None:
        assert config is not None
        _validate_loaded_metadata(dataset, manifest, metadata, config)
    return manifest


def verify_split_integrity(
    dataset: CellDataset,
    manifest: SplitManifest,
) -> dict[str, Any]:
    """Verify disjoint cells and one split assignment across every month."""

    cell_sets = {split: manifest.cell_ids(split) for split in PRIMARY_SPLITS}
    overlaps = {
        "train_validation": len(cell_sets["train"] & cell_sets["validation"]),
        "train_test": len(cell_sets["train"] & cell_sets["test"]),
        "validation_test": len(cell_sets["validation"] & cell_sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"cell overlap detected: {overlaps}")
    split_of = {
        assignment.cell_id: assignment.split for assignment in manifest.assignments
    }
    sample_split_sets: dict[str, set[str]] = {}
    for sample in dataset.samples:
        sample_split_sets.setdefault(sample.cell_id, set()).add(split_of[sample.cell_id])
    invalid = {
        cell_id: splits
        for cell_id, splits in sample_split_sets.items()
        if len(splits) != 1
    }
    if invalid:
        raise AssertionError(f"cells span multiple splits across months: {invalid}")
    return {
        "overlaps": overlaps,
        "cell_counts": {split: len(cells) for split, cells in cell_sets.items()},
        "sample_counts": {
            split: len(manifest.sample_indices(dataset, split))
            for split in PRIMARY_SPLITS
        },
    }
