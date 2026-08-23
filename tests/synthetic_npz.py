"""Synthetic metadata-bearing NPZ fixtures for Phase 2 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cell_msca.data import STREAM_A_FEATURES, STREAM_B_FEATURES


def synthetic_arrays(
    month_index: int,
    *,
    height: int,
    width: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(height * width, dtype=np.float64).reshape(height, width)
    stream_a = np.stack(
        [grid + 10.0 * (channel + 1) + month_index for channel in range(3)]
    )
    stream_b = np.stack(
        [grid + 100.0 * (channel + 1) + month_index for channel in range(4)]
    )
    target = grid + 1.0 + month_index
    if mask is None:
        mask = np.ones((height, width), dtype=np.uint8)
    return stream_a, stream_b, target, np.asarray(mask, dtype=np.uint8)


def write_synthetic_npz(
    path: Path,
    *,
    month_id: str,
    stream_a: np.ndarray,
    stream_b: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    stream_a_features: tuple[str, ...] = STREAM_A_FEATURES,
    stream_b_features: tuple[str, ...] = STREAM_B_FEATURES,
    data_version: str = "v2_corrected_synthetic",
    target_unit: str = "synthetic_unit",
    legacy_metadata_keys: bool = False,
) -> Path:
    if path.exists():
        raise FileExistsError(f"test fixture already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = (
        {
            "stream_a_features": np.asarray(stream_a_features),
            "stream_b_features": np.asarray(stream_b_features),
        }
        if legacy_metadata_keys
        else {
            "stream_a_feature_names": np.asarray(stream_a_features),
            "stream_b_feature_names": np.asarray(stream_b_features),
            "data_version": np.asarray(data_version),
            "target_unit": np.asarray(target_unit),
        }
    )
    arrays = {
        "stream_a": np.asarray(stream_a, dtype=np.float64),
        "stream_b": np.asarray(stream_b, dtype=np.float64),
        "label_reg": np.asarray(target, dtype=np.float64),
        "mask": np.asarray(mask, dtype=np.uint8),
        "month_id": np.asarray(month_id),
        "label_cls": np.zeros_like(target, dtype=np.int64),
        "hotspot_label": np.zeros_like(target, dtype=np.int64),
    }
    np.savez_compressed(path, **arrays, **metadata)
    return path


def write_synthetic_months(
    directory: Path,
    *,
    n_months: int = 2,
    height: int = 4,
    width: int = 4,
    mask: np.ndarray | None = None,
) -> list[Path]:
    paths = []
    for month_index in range(n_months):
        arrays = synthetic_arrays(
            month_index,
            height=height,
            width=width,
            mask=mask,
        )
        paths.append(
            write_synthetic_npz(
                directory / f"month_{month_index:02d}.npz",
                month_id=f"2026-{month_index + 1:02d}",
                stream_a=arrays[0],
                stream_b=arrays[1],
                target=arrays[2],
                mask=arrays[3],
            )
        )
    return paths
