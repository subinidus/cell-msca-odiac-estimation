from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "region",
    "date",
    "temporal",
    "paths",
    "features",
    "label",
    "run",
]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")

    _require_nested(config, ["project", "name"])
    _require_nested(config, ["project", "gee_project_id"])

    for key in [
        "name",
        "min_lon",
        "max_lon",
        "min_lat",
        "max_lat",
        "grid_crs",
        "output_crs",
        "cell_size_m",
    ]:
        _require_nested(config, ["region", key])

    for key in ["mode", "start_month", "end_month"]:
        _require_nested(config, ["date", key])

    if "mode" not in config["temporal"]:
        raise ValueError("Config missing required key: temporal.mode")

    for key in ["grid_geojson", "grid_csv", "raw_dir", "interim_dir", "processed_dir"]:
        _require_nested(config, ["paths", key])

    pattern = config["paths"].get("odiactif_pattern") or config["paths"].get(
        "odiac_tif_pattern"
    )
    if not pattern:
        raise ValueError(
            "Config paths must include 'odiactif_pattern' "
            "or 'odiac_tif_pattern'."
        )

    if config["date"]["mode"] not in {"single_month", "monthly_range"}:
        raise ValueError("date.mode must be 'single_month' or 'monthly_range'")

    _parse_month(config["date"]["start_month"])
    _parse_month(config["date"]["end_month"])

    if not config["features"]:
        raise ValueError("Config must define at least one feature.")

    for feature_name, feature in config["features"].items():
        for key in ["type", "collection", "stream", "fill_method"]:
            if key not in feature:
                raise ValueError(f"Feature '{feature_name}' missing key: {key}")

        if feature["type"] not in {
            "image",
            "landcover_fraction",
            "vector_power_plant",
        }:
            raise ValueError(
                f"Feature '{feature_name}' has unsupported type: {feature['type']}"
            )

        if feature["type"] in {"image", "landcover_fraction"}:
            for key in [
                "band",
                "output_column",
                "scale",
                "reducer",
                "temporal_resolution",
            ]:
                if key not in feature:
                    raise ValueError(f"Feature '{feature_name}' missing key: {key}")
            if feature["reducer"] not in {"mean", "sum"}:
                raise ValueError(
                    f"Feature '{feature_name}' has unsupported reducer: "
                    f"{feature['reducer']}"
                )
            if feature["temporal_resolution"] not in {"monthly", "yearly"}:
                raise ValueError(
                    f"Feature '{feature_name}' has unsupported temporal_resolution: "
                    f"{feature['temporal_resolution']}"
                )
            if (
                feature["type"] == "landcover_fraction"
                and "landcover_class_value" not in feature
            ):
                raise ValueError(
                    f"Feature '{feature_name}' must define landcover_class_value"
                )

        if feature["type"] == "vector_power_plant":
            for key in [
                "output_columns",
                "capacity_column",
                "fuel_column",
                "fossil_fuels",
            ]:
                if key not in feature:
                    raise ValueError(f"Feature '{feature_name}' missing key: {key}")
            if not feature["output_columns"]:
                raise ValueError(f"Feature '{feature_name}' output_columns is empty")

    if config["label"].get("n_classes") != 5:
        raise ValueError("This pipeline currently expects label.n_classes = 5")


def generate_monthly_periods(config: dict[str, Any]) -> list[str]:
    start = _parse_month(config["date"]["start_month"])
    end = _parse_month(config["date"]["end_month"])
    if start > end:
        raise ValueError("date.start_month must be earlier than date.end_month")

    if config["date"]["mode"] == "single_month":
        return [start.strftime("%Y-%m")]

    months = []
    year = start.year
    month = start.month

    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1

    return months


def month_date_range(month: str) -> tuple[str, str]:
    parsed = _parse_month(month)
    start_date = parsed.strftime("%Y-%m-01")

    next_year = parsed.year
    next_month = parsed.month + 1
    if next_month == 13:
        next_year += 1
        next_month = 1

    end_date = f"{next_year:04d}-{next_month:02d}-01"
    return start_date, end_date


def _require_nested(config: dict[str, Any], path: list[str]) -> None:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise ValueError(f"Config missing required key: {dotted}")
        current = current[key]


def _parse_month(month: str) -> datetime:
    try:
        return datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"Month must use YYYY-MM format, got: {month}") from exc
