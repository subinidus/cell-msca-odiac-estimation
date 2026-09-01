from __future__ import annotations

import pandas as pd

from src.utils.paths import (
    combined_dataset_path,
    ensure_parent_dir,
    feature_clean_path,
    final_dataset_path,
    odiac_label_cls_path,
)
from src.utils.validation import (
    check_file_exists,
    print_label_distribution,
    print_missing_summary,
    validate_no_duplicate_grid_date,
    validate_required_columns,
)


def build_final_dataset(config: dict, month: str) -> pd.DataFrame:
    feature_frames = []
    base_columns = ["grid_id", "row", "col", "lon", "lat", "date"]

    for feature_name, feature in config["features"].items():
        path = check_file_exists(feature_clean_path(config, feature_name, month))
        df = pd.read_csv(path)
        output_columns = _feature_output_columns(feature)
        validate_required_columns(df, ["grid_id", "date", *output_columns])

        if not feature_frames:
            keep_columns = [col for col in base_columns if col in df.columns]
            keep_columns.extend(output_columns)
        else:
            keep_columns = ["grid_id", "date", *output_columns]

        feature_frames.append(df[keep_columns])

    label_path = check_file_exists(odiac_label_cls_path(config, month))
    label_df = pd.read_csv(label_path)

    label_column = config["label"]["label_column"]
    log_column = config["label"]["log_column"]
    class_column = config["label"]["class_column"]
    validate_required_columns(
        label_df,
        ["grid_id", "date", label_column, log_column, class_column],
    )

    final_df = feature_frames[0]
    for df in feature_frames[1:]:
        final_df = final_df.merge(df, on=["grid_id", "date"], how="inner")

    label_keep = ["grid_id", "date", label_column, log_column, class_column]
    final_df = final_df.merge(label_df[label_keep], on=["grid_id", "date"], how="inner")

    sort_columns = [col for col in ["col", "row"] if col in final_df.columns]
    if sort_columns:
        final_df = final_df.sort_values(sort_columns).reset_index(drop=True)

    required_columns = ["grid_id", "date"]
    for feature in config["features"].values():
        required_columns.extend(_feature_output_columns(feature))
    required_columns.extend([label_column, log_column, class_column])
    validate_required_columns(final_df, required_columns)
    validate_no_duplicate_grid_date(final_df)
    _validate_no_missing_major_columns(final_df, required_columns)

    output_csv = final_dataset_path(config, month)
    ensure_parent_dir(output_csv)
    final_df.to_csv(output_csv, index=False)

    print(f"Saved final monthly dataset: {output_csv}")
    print(f"Rows: {len(final_df)}")
    print_missing_summary(final_df)
    print_label_distribution(final_df, class_column)
    return final_df


def build_combined_dataset(config: dict, months: list[str]) -> pd.DataFrame:
    frames = []
    for month in months:
        path = check_file_exists(final_dataset_path(config, month))
        frames.append(pd.read_csv(path))

    combined_df = pd.concat(frames, ignore_index=True)
    validate_no_duplicate_grid_date(combined_df)

    output_csv = combined_dataset_path(config, months)
    ensure_parent_dir(output_csv)
    combined_df.to_csv(output_csv, index=False)

    print(f"Saved combined dataset: {output_csv}")
    print(f"Rows: {len(combined_df)}")
    return combined_df


def _validate_no_missing_major_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = df[columns].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            "Final dataset contains missing values in required columns:\n"
            f"{missing}"
        )


def _feature_output_columns(feature: dict) -> list[str]:
    if "output_columns" in feature:
        return list(feature["output_columns"])
    return [feature["output_column"]]
