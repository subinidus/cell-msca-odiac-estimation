from __future__ import annotations

import pandas as pd

from src.utils.paths import ensure_parent_dir, feature_clean_path, feature_raw_path
from src.utils.validation import check_file_exists, validate_required_columns


def clean_feature(config: dict, feature_name: str, month: str) -> pd.DataFrame:
    if feature_name not in config["features"]:
        raise ValueError(f"Unknown feature '{feature_name}'")

    feature = config["features"][feature_name]
    input_csv = check_file_exists(feature_raw_path(config, feature_name, month))
    output_csv = feature_clean_path(config, feature_name, month)
    ensure_parent_dir(output_csv)

    df = pd.read_csv(input_csv)
    output_column = feature["output_column"]
    validate_required_columns(df, ["grid_id", "date", output_column])

    missing_before = df[output_column].isna().sum()
    fill_value = _compute_fill_value(
    series=df[output_column],
    fill_method=feature.get("fill_method", "mean"),
    feature_name=feature_name,
    month=month,
    )
    df[output_column] = df[output_column].fillna(fill_value)
    missing_after = df[output_column].isna().sum()

    df.to_csv(output_csv, index=False)

    print(f"Cleaned {feature_name} for {month}: {output_csv}")
    print(f"Missing before: {missing_before}")
    print(f"Fill value: {fill_value}")
    print(f"Missing after: {missing_after}")
    return df


def _compute_fill_value(
    series: pd.Series,
    fill_method: str,
    feature_name: str,
    month: str,
) -> float:
    if fill_method == "mean":
        fill_value = series.mean()
    elif fill_method == "median":
        fill_value = series.median()
    elif fill_method == "zero":
        return 0.0
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")

    if pd.isna(fill_value):
        raise ValueError(
            f"All values are missing for feature '{feature_name}' in {month}. "
            f"Cannot fill missing values using '{fill_method}'. "
            "Please check the GEE extraction result or use fill_method: zero only if this is scientifically justified."
        )

    return float(fill_value)
