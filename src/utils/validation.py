from __future__ import annotations

from pathlib import Path

import pandas as pd


def check_file_exists(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required file is missing: {file_path}")
    return file_path


def validate_no_duplicate_grid_date(df: pd.DataFrame) -> None:
    validate_required_columns(df, ["grid_id", "date"])
    duplicated = df.duplicated(subset=["grid_id", "date"]).sum()
    if duplicated:
        raise ValueError(f"Found duplicated grid_id/date rows: {duplicated}")


def validate_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_label_classes(
    df: pd.DataFrame,
    class_column: str,
    n_classes: int,
) -> None:
    validate_required_columns(df, [class_column])
    invalid = sorted(set(df[class_column].dropna()) - set(range(n_classes)))
    if invalid:
        raise ValueError(f"Invalid label classes in {class_column}: {invalid}")


def print_missing_summary(df: pd.DataFrame) -> None:
    print("Missing values by column:")
    print(df.isna().sum())


def print_label_distribution(df: pd.DataFrame, class_column: str) -> None:
    print(f"{class_column} distribution:")
    print(df[class_column].value_counts().sort_index())
