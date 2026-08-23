"""Single-cell data loading and train-only feature preprocessing.

The dataset exposes seven scalar features for one cell and one month. Spatial
neighbours, coordinates, grid indices, masks, and classification labels are
kept out of every model-facing sample.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .target import target_transform

STREAM_A_FEATURES = ("no2_mean", "so2_mean", "co_mean")
STREAM_B_FEATURES = (
    "nightlight_mean",
    "urban_fraction",
    "power_plant_count",
    "fossil_capacity_mw",
)

REQUIRED_NPZ_ARRAYS = (
    "stream_a",
    "stream_b",
    "label_reg",
    "mask",
)

FEATURE_METADATA_KEYS = {
    "stream_a": ("stream_a_feature_names", "stream_a_features"),
    "stream_b": ("stream_b_feature_names", "stream_b_features"),
}

MODEL_SAMPLE_KEYS = {
    "stream_a",
    "stream_b",
    "target_original",
    "target_log",
    "cell_id",
    "month_id",
}

PREPROCESSING_SCHEMA_VERSION = "cell_msca.preprocessing.v2"


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value through a stable canonical encoding."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_cell_id(row: int, col: int) -> str:
    """Create the stable identifier used by datasets and persistent splits."""

    if row < 0 or col < 0:
        raise ValueError(f"row and col must be non-negative; got row={row}, col={col}")
    return f"r{row}_c{col}"


def _text_value(value: Any, *, name: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _scalar_text(data: np.lib.npyio.NpzFile, key: str, *, path: Path) -> str:
    raw = np.asarray(data[key])
    if raw.size != 1:
        raise ValueError(f"{path} metadata {key!r} must contain exactly one value")
    return _text_value(raw.reshape(-1)[0], name=f"{path}:{key}")


def _feature_names(
    data: np.lib.npyio.NpzFile,
    key: str,
    *,
    path: Path,
) -> tuple[str, ...]:
    raw = np.asarray(data[key]).ravel()
    return tuple(_text_value(value, name=f"{path}:{key}") for value in raw)


def _first_available_key(
    data: np.lib.npyio.NpzFile,
    candidates: tuple[str, ...],
    *,
    path: Path,
) -> str:
    for key in candidates:
        if key in data.files:
            return key
    raise ValueError(f"{path} missing feature metadata; expected one of {candidates}")


@dataclass(frozen=True)
class FeatureMetadata:
    """NPZ feature and target metadata shared by every month."""

    stream_a: tuple[str, ...]
    stream_b: tuple[str, ...]
    data_version: str
    target_unit: str

    def validate_normative_order(self, *, path: Path | None = None) -> None:
        location = f"{path} " if path is not None else ""
        if self.stream_a != STREAM_A_FEATURES:
            raise ValueError(
                f"{location}stream_a feature order mismatch: "
                f"expected={STREAM_A_FEATURES}, actual={self.stream_a}"
            )
        if self.stream_b != STREAM_B_FEATURES:
            raise ValueError(
                f"{location}stream_b feature order mismatch: "
                f"expected={STREAM_B_FEATURES}, actual={self.stream_b}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_a": list(self.stream_a),
            "stream_b": list(self.stream_b),
            "data_version": self.data_version,
            "target_unit": self.target_unit,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CellLocation:
    row: int
    col: int
    cell_id: str


@dataclass(frozen=True)
class SampleReference:
    month_index: int
    row: int
    col: int
    cell_id: str


@dataclass(frozen=True)
class MonthlyCellGrid:
    month_id: str
    stream_a: NDArray[np.float64]
    stream_b: NDArray[np.float64]
    target_original: NDArray[np.float64]
    target_log: NDArray[np.float64]
    mask: NDArray[np.bool_]


@dataclass(frozen=True)
class PreprocessingStats:
    """Mean imputation and standardization fitted on train cells only."""

    schema_version: str
    data_sha256: str
    split_sha256: str
    split_name: str
    stream_a_impute: tuple[float, ...]
    stream_a_mean: tuple[float, ...]
    stream_a_std: tuple[float, ...]
    stream_b_impute: tuple[float, ...]
    stream_b_mean: tuple[float, ...]
    stream_b_std: tuple[float, ...]
    epsilon: float
    n_train_cells: int
    n_train_samples: int
    train_cells_sha256: str
    feature_metadata_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_sha256": self.data_sha256,
            "split_sha256": self.split_sha256,
            "split_name": self.split_name,
            "stream_a_impute": list(self.stream_a_impute),
            "stream_a_mean": list(self.stream_a_mean),
            "stream_a_std": list(self.stream_a_std),
            "stream_b_impute": list(self.stream_b_impute),
            "stream_b_mean": list(self.stream_b_mean),
            "stream_b_std": list(self.stream_b_std),
            "epsilon": self.epsilon,
            "n_train_cells": self.n_train_cells,
            "n_train_samples": self.n_train_samples,
            "train_cells_sha256": self.train_cells_sha256,
            "feature_metadata_sha256": self.feature_metadata_sha256,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PreprocessingStats:
        """Restore JSON-compatible statistics for validation or inference."""

        required = {
            "schema_version",
            "data_sha256",
            "split_sha256",
            "split_name",
            "stream_a_impute",
            "stream_a_mean",
            "stream_a_std",
            "stream_b_impute",
            "stream_b_mean",
            "stream_b_std",
            "epsilon",
            "n_train_cells",
            "n_train_samples",
            "train_cells_sha256",
            "feature_metadata_sha256",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(
                "preprocessing statistics are missing required fields: "
                f"{sorted(missing)}"
            )
        return cls(
            schema_version=str(values["schema_version"]),
            data_sha256=str(values["data_sha256"]),
            split_sha256=str(values["split_sha256"]),
            split_name=str(values["split_name"]),
            stream_a_impute=tuple(float(value) for value in values["stream_a_impute"]),
            stream_a_mean=tuple(float(value) for value in values["stream_a_mean"]),
            stream_a_std=tuple(float(value) for value in values["stream_a_std"]),
            stream_b_impute=tuple(float(value) for value in values["stream_b_impute"]),
            stream_b_mean=tuple(float(value) for value in values["stream_b_mean"]),
            stream_b_std=tuple(float(value) for value in values["stream_b_std"]),
            epsilon=float(values["epsilon"]),
            n_train_cells=int(values["n_train_cells"]),
            n_train_samples=int(values["n_train_samples"]),
            train_cells_sha256=str(values["train_cells_sha256"]),
            feature_metadata_sha256=str(values["feature_metadata_sha256"]),
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def validate_for(
        self,
        metadata: FeatureMetadata,
        *,
        data_sha256: str,
        split_sha256: str,
        split_name: str,
    ) -> None:
        """Validate that fitted statistics belong to this data and split."""

        if self.schema_version != PREPROCESSING_SCHEMA_VERSION:
            raise ValueError(
                "unsupported preprocessing schema version: "
                f"stats={self.schema_version!r}, "
                f"expected={PREPROCESSING_SCHEMA_VERSION!r}"
            )
        if self.data_sha256 != data_sha256:
            raise ValueError(
                "preprocessing data SHA-256 does not match the dataset: "
                f"stats={self.data_sha256}, dataset={data_sha256}"
            )
        if self.split_sha256 != split_sha256:
            raise ValueError(
                "preprocessing split SHA-256 does not match the selected split: "
                f"stats={self.split_sha256}, split={split_sha256}"
            )
        if self.split_name != split_name:
            raise ValueError(
                "preprocessing split name does not match the selected split: "
                f"stats={self.split_name!r}, split={split_name!r}"
            )
        if self.feature_metadata_sha256 != metadata.sha256:
            raise ValueError(
                "preprocessing feature metadata does not match the dataset: "
                f"stats={self.feature_metadata_sha256}, dataset={metadata.sha256}"
            )
        expected_lengths = {
            "stream_a_impute": len(STREAM_A_FEATURES),
            "stream_a_mean": len(STREAM_A_FEATURES),
            "stream_a_std": len(STREAM_A_FEATURES),
            "stream_b_impute": len(STREAM_B_FEATURES),
            "stream_b_mean": len(STREAM_B_FEATURES),
            "stream_b_std": len(STREAM_B_FEATURES),
        }
        for name, expected in expected_lengths.items():
            actual = len(getattr(self, name))
            if actual != expected:
                raise ValueError(f"{name} length is {actual}; expected {expected}")


class CellDataset:
    """Monthly single-cell samples loaded from metadata-bearing grid NPZ files."""

    def __init__(
        self,
        npz_paths: list[str | Path] | tuple[str | Path, ...],
        *,
        target_scale: float = 1.0,
        preprocessing: PreprocessingStats | None = None,
        preprocessing_split_sha256: str | None = None,
        preprocessing_split_name: str = "train",
        legacy_data_version: str | None = None,
        legacy_target_unit: str | None = None,
    ) -> None:
        if not npz_paths:
            raise ValueError("npz_paths must not be empty")
        self.npz_paths = tuple(Path(path) for path in npz_paths)
        if len({path.name for path in self.npz_paths}) != len(self.npz_paths):
            raise ValueError("NPZ file names must be unique for a stable data manifest")
        self.target_scale = float(target_scale)
        if not np.isfinite(self.target_scale) or self.target_scale <= 0.0:
            raise ValueError("target_scale must be finite and positive")
        if (legacy_data_version is None) != (legacy_target_unit is None):
            raise ValueError(
                "legacy_data_version and legacy_target_unit must be supplied together"
            )
        self.legacy_data_version = legacy_data_version
        self.legacy_target_unit = legacy_target_unit

        months: list[MonthlyCellGrid] = []
        feature_metadata: FeatureMetadata | None = None
        reference_mask: NDArray[np.bool_] | None = None
        for path in self.npz_paths:
            month, metadata = self._load_month(path)
            if feature_metadata is None:
                feature_metadata = metadata
                reference_mask = month.mask
            else:
                if metadata != feature_metadata:
                    raise ValueError(
                        f"{path} feature/data metadata differs from the first NPZ: "
                        f"expected={feature_metadata.to_dict()}, actual={metadata.to_dict()}"
                    )
                if not np.array_equal(month.mask, reference_mask):
                    raise ValueError(f"{path} mask differs from the first NPZ")
            months.append(month)

        assert feature_metadata is not None
        assert reference_mask is not None
        month_ids = [month.month_id for month in months]
        if len(set(month_ids)) != len(month_ids):
            raise ValueError(f"month_id values must be unique; got {month_ids}")

        self.feature_metadata = feature_metadata
        self.months = tuple(months)
        rows, cols = np.where(reference_mask)
        self.cell_locations = tuple(
            CellLocation(int(row), int(col), stable_cell_id(int(row), int(col)))
            for row, col in zip(rows, cols, strict=True)
        )
        if not self.cell_locations:
            raise ValueError("dataset contains no valid cells")
        self._cell_by_id = {cell.cell_id: cell for cell in self.cell_locations}
        self.samples = tuple(
            SampleReference(month_index, cell.row, cell.col, cell.cell_id)
            for month_index in range(len(self.months))
            for cell in self.cell_locations
        )
        data_manifest = [
            {
                "index": index,
                "file_name": path.name,
                "file_sha256": file_sha256(path),
                "month_id": self.months[index].month_id,
            }
            for index, path in enumerate(self.npz_paths)
        ]
        self.data_manifest = tuple(data_manifest)
        self.data_sha256 = canonical_sha256(data_manifest)
        self.preprocessing: PreprocessingStats | None = None
        if preprocessing is not None:
            if preprocessing_split_sha256 is None:
                raise ValueError(
                    "preprocessing_split_sha256 is required when preprocessing "
                    "statistics are supplied"
                )
            self.set_preprocessing(
                preprocessing,
                split_sha256=preprocessing_split_sha256,
                split_name=preprocessing_split_name,
            )
        elif preprocessing_split_sha256 is not None:
            raise ValueError(
                "preprocessing statistics are required when "
                "preprocessing_split_sha256 is supplied"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        reference = self.samples[index]
        month = self.months[reference.month_index]
        stream_a = month.stream_a[:, reference.row, reference.col].copy()
        stream_b = month.stream_b[:, reference.row, reference.col].copy()
        if self.preprocessing is None:
            if np.isnan(stream_a).any() or np.isnan(stream_b).any():
                raise RuntimeError(
                    "sample contains missing features; fit train-only preprocessing "
                    "and call set_preprocessing() before reading model samples"
                )
        else:
            stream_a = _transform_stream(
                stream_a,
                impute=self.preprocessing.stream_a_impute,
                mean=self.preprocessing.stream_a_mean,
                std=self.preprocessing.stream_a_std,
                epsilon=self.preprocessing.epsilon,
            )
            stream_b = _transform_stream(
                stream_b,
                impute=self.preprocessing.stream_b_impute,
                mean=self.preprocessing.stream_b_mean,
                std=self.preprocessing.stream_b_std,
                epsilon=self.preprocessing.epsilon,
            )

        sample = {
            "stream_a": stream_a.astype(np.float32),
            "stream_b": stream_b.astype(np.float32),
            "target_original": float(
                month.target_original[reference.row, reference.col]
            ),
            "target_log": float(month.target_log[reference.row, reference.col]),
            "cell_id": reference.cell_id,
            "month_id": month.month_id,
        }
        if set(sample) != MODEL_SAMPLE_KEYS:
            raise AssertionError(f"model sample contract changed unexpectedly: {set(sample)}")
        return sample

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self.cell_locations)

    @property
    def n_months(self) -> int:
        return len(self.months)

    def cell_location(self, cell_id: str) -> CellLocation:
        try:
            return self._cell_by_id[cell_id]
        except KeyError as error:
            raise KeyError(f"unknown cell_id: {cell_id!r}") from error

    def sample_metadata(self, index: int) -> dict[str, Any]:
        """Return audit metadata separately from the model-facing sample."""

        reference = self.samples[index]
        return {
            "month_id": self.months[reference.month_index].month_id,
            "row": reference.row,
            "col": reference.col,
            "cell_id": reference.cell_id,
            "data_version": self.feature_metadata.data_version,
            "target_unit": self.feature_metadata.target_unit,
            "stream_a_feature_order": list(self.feature_metadata.stream_a),
            "stream_b_feature_order": list(self.feature_metadata.stream_b),
        }

    def sample_indices_for_cells(self, cell_ids: set[str] | frozenset[str]) -> list[int]:
        unknown = set(cell_ids) - set(self._cell_by_id)
        if unknown:
            raise ValueError(f"unknown cell_ids: {sorted(unknown)[:3]}")
        return [
            index
            for index, sample in enumerate(self.samples)
            if sample.cell_id in cell_ids
        ]

    def sample_count_for_cells(self, cell_ids: set[str] | frozenset[str]) -> int:
        return len(self.sample_indices_for_cells(cell_ids))

    def set_preprocessing(
        self,
        stats: PreprocessingStats,
        *,
        split_sha256: str,
        split_name: str = "train",
    ) -> None:
        """Attach already-fitted train-only statistics without recomputing them."""

        stats.validate_for(
            self.feature_metadata,
            data_sha256=self.data_sha256,
            split_sha256=_validated_sha256(split_sha256, name="split_sha256"),
            split_name=_nonempty_text(split_name, name="split_name"),
        )
        self.preprocessing = stats

    def raw_feature_matrices(
        self,
        cell_ids: set[str] | frozenset[str],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return raw sample matrices for an explicit set of cell identifiers."""

        indices = self.sample_indices_for_cells(cell_ids)
        if not indices:
            raise ValueError("cell_ids selected no samples")
        stream_a = np.empty((len(indices), len(STREAM_A_FEATURES)), dtype=np.float64)
        stream_b = np.empty((len(indices), len(STREAM_B_FEATURES)), dtype=np.float64)
        for output_index, sample_index in enumerate(indices):
            reference = self.samples[sample_index]
            month = self.months[reference.month_index]
            stream_a[output_index] = month.stream_a[:, reference.row, reference.col]
            stream_b[output_index] = month.stream_b[:, reference.row, reference.col]
        return stream_a, stream_b

    def _load_month(self, path: Path) -> tuple[MonthlyCellGrid, FeatureMetadata]:
        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_NPZ_ARRAYS if key not in data.files]
            if missing:
                raise ValueError(f"{path} missing required arrays/metadata: {missing}")
            stream_a_key = _first_available_key(
                data,
                FEATURE_METADATA_KEYS["stream_a"],
                path=path,
            )
            stream_b_key = _first_available_key(
                data,
                FEATURE_METADATA_KEYS["stream_b"],
                path=path,
            )
            if "data_version" in data.files:
                data_version = _scalar_text(data, "data_version", path=path)
            elif self.legacy_data_version is not None:
                data_version = _text_value(
                    self.legacy_data_version,
                    name="legacy_data_version",
                )
            else:
                raise ValueError(
                    f"{path} missing data_version metadata; "
                    "legacy files require an explicit legacy_data_version"
                )
            if "target_unit" in data.files:
                target_unit = _scalar_text(data, "target_unit", path=path)
            elif self.legacy_target_unit is not None:
                target_unit = _text_value(
                    self.legacy_target_unit,
                    name="legacy_target_unit",
                )
            else:
                raise ValueError(
                    f"{path} missing target_unit metadata; "
                    "legacy files require an explicit legacy_target_unit"
                )
            metadata = FeatureMetadata(
                stream_a=_feature_names(data, stream_a_key, path=path),
                stream_b=_feature_names(data, stream_b_key, path=path),
                data_version=data_version,
                target_unit=target_unit,
            )
            metadata.validate_normative_order(path=path)
            stream_a = np.asarray(data["stream_a"], dtype=np.float64)
            stream_b = np.asarray(data["stream_b"], dtype=np.float64)
            target_original = np.asarray(data["label_reg"], dtype=np.float64)
            raw_mask = np.asarray(data["mask"])
            if "month_id" in data.files:
                month_id = _scalar_text(data, "month_id", path=path)
            elif "date" in data.files:
                month_id = _scalar_text(data, "date", path=path)
            else:
                month_id = path.stem

        _validate_month_shapes(path, stream_a, stream_b, target_original, raw_mask)
        mask_values = set(np.unique(raw_mask).tolist())
        if not mask_values <= {0, 1, False, True}:
            raise ValueError(f"{path} mask must contain only 0/1 values")
        mask = raw_mask.astype(bool)
        if np.isinf(stream_a[:, mask]).any() or np.isinf(stream_b[:, mask]).any():
            raise ValueError(f"{path} valid-cell features must not contain infinity")
        target_log = np.zeros_like(target_original, dtype=np.float64)
        target_log[mask] = target_transform(
            target_original[mask],
            scale=self.target_scale,
        )
        return (
            MonthlyCellGrid(
                month_id=month_id,
                stream_a=stream_a,
                stream_b=stream_b,
                target_original=target_original,
                target_log=target_log,
                mask=mask,
            ),
            metadata,
        )


def _validate_month_shapes(
    path: Path,
    stream_a: NDArray[np.float64],
    stream_b: NDArray[np.float64],
    target: NDArray[np.float64],
    mask: NDArray[Any],
) -> None:
    if stream_a.ndim != 3 or stream_a.shape[0] != len(STREAM_A_FEATURES):
        raise ValueError(
            f"{path} stream_a must have shape [{len(STREAM_A_FEATURES)}, H, W]; "
            f"got {stream_a.shape}"
        )
    if stream_b.ndim != 3 or stream_b.shape[0] != len(STREAM_B_FEATURES):
        raise ValueError(
            f"{path} stream_b must have shape [{len(STREAM_B_FEATURES)}, H, W]; "
            f"got {stream_b.shape}"
        )
    spatial_shape = stream_a.shape[1:]
    for name, actual in (
        ("stream_b", stream_b.shape[1:]),
        ("label_reg", target.shape),
        ("mask", mask.shape),
    ):
        if actual != spatial_shape:
            raise ValueError(
                f"{path} {name} spatial shape is {actual}; expected {spatial_shape}"
            )
    if not np.any(mask == 1):
        raise ValueError(f"{path} contains no valid mask cells")


def _fit_stream(
    values: NDArray[np.float64],
    *,
    feature_names: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if np.isinf(values).any():
        raise ValueError("training features must not contain infinity")
    observed_counts = np.sum(~np.isnan(values), axis=0)
    missing_features = [
        feature_names[index]
        for index, count in enumerate(observed_counts)
        if count == 0
    ]
    if missing_features:
        raise ValueError(
            f"training cells have no observed values for features: {missing_features}"
        )
    impute = np.nanmean(values, axis=0)
    filled = np.where(np.isnan(values), impute[None, :], values)
    mean = np.mean(filled, axis=0)
    std = np.std(filled, axis=0)
    return (
        tuple(float(value) for value in impute),
        tuple(float(value) for value in mean),
        tuple(float(value) for value in std),
    )


def fit_train_preprocessing(
    dataset: CellDataset,
    train_cell_ids: Iterable[str],
    *,
    split_sha256: str,
    split_name: str = "train",
    epsilon: float = 1e-8,
) -> PreprocessingStats:
    """Fit imputation and normalization using only explicitly supplied train cells."""

    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if isinstance(train_cell_ids, (str, bytes)):
        raise ValueError("train_cell_ids must be an iterable of identifiers, not one string")
    split_sha256 = _validated_sha256(split_sha256, name="split_sha256")
    split_name = _nonempty_text(split_name, name="split_name")
    train_cells = frozenset(str(value) for value in train_cell_ids)
    if not train_cells:
        raise ValueError("train_cell_ids must not be empty")
    stream_a, stream_b = dataset.raw_feature_matrices(train_cells)
    a_impute, a_mean, a_std = _fit_stream(
        stream_a,
        feature_names=STREAM_A_FEATURES,
    )
    b_impute, b_mean, b_std = _fit_stream(
        stream_b,
        feature_names=STREAM_B_FEATURES,
    )
    return PreprocessingStats(
        schema_version=PREPROCESSING_SCHEMA_VERSION,
        data_sha256=dataset.data_sha256,
        split_sha256=split_sha256,
        split_name=split_name,
        stream_a_impute=a_impute,
        stream_a_mean=a_mean,
        stream_a_std=a_std,
        stream_b_impute=b_impute,
        stream_b_mean=b_mean,
        stream_b_std=b_std,
        epsilon=epsilon,
        n_train_cells=len(train_cells),
        n_train_samples=stream_a.shape[0],
        train_cells_sha256=canonical_sha256(sorted(train_cells)),
        feature_metadata_sha256=dataset.feature_metadata.sha256,
    )


def _validated_sha256(value: str, *, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return digest


def _nonempty_text(value: str, *, name: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _transform_stream(
    values: NDArray[np.float64],
    *,
    impute: tuple[float, ...],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    epsilon: float,
) -> NDArray[np.float64]:
    impute_array = np.asarray(impute, dtype=np.float64)
    mean_array = np.asarray(mean, dtype=np.float64)
    std_array = np.asarray(std, dtype=np.float64)
    filled = np.where(np.isnan(values), impute_array, values)
    transformed = (filled - mean_array) / (std_array + epsilon)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("preprocessing produced non-finite feature values")
    return transformed
