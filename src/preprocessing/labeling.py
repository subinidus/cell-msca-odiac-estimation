from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import (
    ensure_parent_dir,
    label_thresholds_path,
    odiac_label_cls_path,
    odiac_label_path,
)
from src.utils.validation import (
    check_file_exists,
    print_label_distribution,
    validate_label_classes,
    validate_required_columns,
)


def create_label_cls_per_month(config: dict, month: str) -> pd.DataFrame:
    input_csv = check_file_exists(odiac_label_path(config, month))
    output_csv = odiac_label_cls_path(config, month)
    ensure_parent_dir(output_csv)

    df = pd.read_csv(input_csv)
    label_column = config["label"]["label_column"]
    log_column = config["label"]["log_column"]
    class_column = config["label"]["class_column"]
    n_classes = config["label"]["n_classes"]

    validate_required_columns(df, ["grid_id", "date", label_column])
    _validate_label_values(df, label_column, month)
    df[log_column] = np.log1p(df[label_column].clip(lower=0))
    thresholds = _quantile_thresholds(df[label_column], n_classes)
    df[class_column] = _apply_thresholds(df[label_column], thresholds, n_classes)

    df = _label_keep_columns(df, label_column, log_column, class_column)
    validate_label_classes(df, class_column, n_classes)
    df.to_csv(output_csv, index=False)

    print(f"Saved per-month classified labels: {output_csv}")
    print_label_distribution(df, class_column)
    return df


def create_label_cls_global_range(config: dict, months: list[str]) -> dict[str, pd.DataFrame]:
    label_column = config["label"]["label_column"]
    log_column = config["label"]["log_column"]
    class_column = config["label"]["class_column"]
    n_classes = config["label"]["n_classes"]

    monthly_frames = {}
    values = []

    for month in months:
        path = check_file_exists(odiac_label_path(config, month))
        df = pd.read_csv(path)
        validate_required_columns(df, ["grid_id", "date", label_column])
        _validate_label_values(df, label_column, month)
        monthly_frames[month] = df
        values.append(df[label_column].dropna())

    all_values = pd.concat(values, ignore_index=True)
    if all_values.empty:
        raise ValueError("Cannot compute global label thresholds: no label_reg values.")

    thresholds = _quantile_thresholds(all_values, n_classes)
    _save_global_thresholds(config, months, thresholds)

    outputs = {}
    for month, df in monthly_frames.items():
        output_csv = odiac_label_cls_path(config, month)
        ensure_parent_dir(output_csv)

        df[log_column] = np.log1p(df[label_column].clip(lower=0))
        df[class_column] = _apply_thresholds(df[label_column], thresholds, n_classes)
        df = _label_keep_columns(df, label_column, log_column, class_column)
        validate_label_classes(df, class_column, n_classes)
        df.to_csv(output_csv, index=False)
        outputs[month] = df

        print(f"Saved global-range classified labels: {output_csv}")
        print_label_distribution(df, class_column)

    return outputs


def _quantile_thresholds(values: pd.Series, n_classes: int) -> np.ndarray:
    quantiles = [i / n_classes for i in range(1, n_classes)]
    return values.quantile(quantiles).to_numpy(dtype=np.float64)


def _validate_label_values(df: pd.DataFrame, label_column: str, month: str) -> None:
    missing = df[label_column].isna().sum()
    if missing:
        raise ValueError(
            f"Cannot create label classes for {month}: "
            f"{missing} missing values in {label_column}."
        )


def _apply_thresholds(
    values: pd.Series,
    thresholds: np.ndarray,
    n_classes: int,
) -> pd.Series:
    classes = np.searchsorted(thresholds, values.to_numpy(), side="right")
    classes = np.clip(classes, 0, n_classes - 1)
    return pd.Series(classes, index=values.index, dtype="int64")


def _save_global_thresholds(config: dict, months: list[str], thresholds: np.ndarray) -> None:
    output_csv = label_thresholds_path(config, months)
    ensure_parent_dir(output_csv)
    threshold_df = pd.DataFrame(
        {
            "quantile": [0.2, 0.4, 0.6, 0.8],
            "threshold": thresholds,
        }
    )
    threshold_df.to_csv(output_csv, index=False)
    print(f"Saved global label thresholds: {output_csv}")


def _label_keep_columns(
    df: pd.DataFrame,
    label_column: str,
    log_column: str,
    class_column: str,
) -> pd.DataFrame:
    keep_columns = [
        "grid_id",
        "row",
        "col",
        "lon",
        "lat",
        "date",
        label_column,
        log_column,
        class_column,
    ]
    available_columns = [col for col in keep_columns if col in df.columns]
    return df[available_columns]
