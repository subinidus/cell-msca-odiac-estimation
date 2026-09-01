from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import (
    ensure_parent_dir,
    final_dataset_path,
    tensor_index_path,
    tensor_path,
)
from src.utils.validation import check_file_exists, validate_required_columns


def build_tensor(config: dict, month: str) -> dict[str, np.ndarray]:
    input_csv = check_file_exists(final_dataset_path(config, month))
    output_npz = tensor_path(config, month)
    ensure_parent_dir(output_npz)

    df = pd.read_csv(input_csv)
    validate_required_columns(df, ["row", "col"])

    stream_a_columns = _feature_columns_for_stream(config, "stream_a")
    stream_b_columns = _feature_columns_for_stream(config, "stream_b")

    label_column = config["label"]["label_column"]
    class_column = config["label"]["class_column"]
    required = stream_a_columns + stream_b_columns + [label_column, class_column]
    validate_required_columns(df, required)

    height = int(df["row"].max()) + 1
    width = int(df["col"].max()) + 1

    stream_a = _build_feature_tensor(df, stream_a_columns, height, width)
    stream_b = _build_feature_tensor(df, stream_b_columns, height, width)
    label_cls = np.zeros((height, width), dtype=np.int64)
    label_reg = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    for _, row in df.iterrows():
        r = int(row["row"])
        c = int(row["col"])
        label_cls[r, c] = int(row[class_column])
        label_reg[r, c] = float(row[label_column])
        mask[r, c] = 1

    np.savez_compressed(
        output_npz,
        stream_a=stream_a,
        stream_b=stream_b,
        label_cls=label_cls,
        label_reg=label_reg,
        mask=mask,
        stream_a_features=np.array(stream_a_columns),
        stream_b_features=np.array(stream_b_columns),
        date=np.array(month),
        region=np.array(config["region"]["name"]),
        height=np.array(height),
        width=np.array(width),
    )

    print(f"Saved tensor sample: {output_npz}")
    print(f"stream_a: {stream_a.shape}")
    print(f"stream_b: {stream_b.shape}")
    print(f"label_cls: {label_cls.shape}")
    print(f"label_reg: {label_reg.shape}")
    print(f"mask: {mask.shape}")

    return {
        "stream_a": stream_a,
        "stream_b": stream_b,
        "label_cls": label_cls,
        "label_reg": label_reg,
        "mask": mask,
    }


def build_tensor_index(config: dict, months: list[str]) -> pd.DataFrame:
    rows = []
    for month in months:
        tensor_file = check_file_exists(tensor_path(config, month))
        final_file = check_file_exists(final_dataset_path(config, month))
        rows.append(
            {
                "date": month,
                "tensor_path": str(tensor_file),
                "final_dataset_path": str(final_file),
            }
        )

    index_df = pd.DataFrame(rows)
    output_csv = tensor_index_path(config, months)
    ensure_parent_dir(output_csv)
    index_df.to_csv(output_csv, index=False)
    print(f"Saved tensor index: {output_csv}")
    return index_df


def _feature_columns_for_stream(config: dict, stream_name: str) -> list[str]:
    columns = []
    for feature in config["features"].values():
        if feature["stream"] != stream_name:
            continue
        columns.extend(_feature_output_columns(feature))
    return columns


def _feature_output_columns(feature: dict) -> list[str]:
    if "output_columns" in feature:
        return list(feature["output_columns"])
    return [feature["output_column"]]


def _build_feature_tensor(
    df: pd.DataFrame,
    feature_columns: list[str],
    height: int,
    width: int,
) -> np.ndarray:
    tensor = np.zeros((len(feature_columns), height, width), dtype=np.float32)
    for channel, feature_column in enumerate(feature_columns):
        for _, row in df.iterrows():
            r = int(row["row"])
            c = int(row["col"])
            tensor[channel, r, c] = float(row[feature_column])
    return tensor
