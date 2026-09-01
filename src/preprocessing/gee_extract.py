from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.utils.config import month_date_range
from src.utils.paths import (
    ensure_dir,
    ensure_parent_dir,
    feature_batch_dir,
    feature_raw_path,
)
from src.utils.validation import check_file_exists, validate_required_columns


_EE_INITIALIZED = False
SUPPORTED_FEATURE_TYPES = {"image", "landcover_fraction"}
SUPPORTED_TEMPORAL_RESOLUTIONS = {"monthly", "yearly"}
OPTIONAL_GRID_COLUMNS = ["row", "col", "lon", "lat"]


def extract_gee_feature(config: dict, feature_name: str, month: str) -> pd.DataFrame:
    import ee
    import geemap
    import geopandas as gpd

    if feature_name not in config["features"]:
        raise ValueError(f"Unknown feature '{feature_name}'")

    _initialize_earth_engine(config["project"]["gee_project_id"])

    feature = config["features"][feature_name]
    _validate_feature_config(feature_name, feature)

    grid_path = check_file_exists(config["paths"]["grid_geojson"])
    output_csv = feature_raw_path(config, feature_name, month)
    batch_dir = feature_batch_dir(config, feature_name, month)

    ensure_parent_dir(output_csv)
    ensure_dir(batch_dir)

    grid = gpd.read_file(grid_path).reset_index(drop=True)
    validate_required_columns(grid, ["grid_id"])

    image, image_info = _build_feature_image(ee, feature, month)
    reducer_name = feature.get("reducer", "mean")
    reducer = _get_reducer(ee, reducer_name)

    print("GEE feature extraction")
    print(f"Feature: {feature_name}")
    print(f"Type: {feature['type']}")
    print(f"Collection: {feature['collection']}")
    print(f"Band: {feature['band']}")
    print(f"Reducer: {reducer_name}")
    print(f"Temporal resolution: {feature['temporal_resolution']}")
    print(f"{image_info['period_label']}: {image_info['period_value']}")
    print(f"Output path: {output_csv}")
    print(f"Image count: {image_info['image_count']}")

    batch_size = config.get("gee", {}).get("batch_size", 500)
    tile_scale = config.get("gee", {}).get("tile_scale", 4)
    num_batches = math.ceil(len(grid) / batch_size)
    all_batches = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(grid))
        batch_gdf = grid.iloc[start_idx:end_idx].copy()
        batch_df = _extract_batch(
            geemap=geemap,
            image=image,
            reducer=reducer,
            reducer_name=reducer_name,
            batch_gdf=batch_gdf,
            batch_dir=batch_dir,
            batch_idx=batch_idx,
            feature_name=feature_name,
            feature=feature,
            month=month,
            tile_scale=tile_scale,
        )
        all_batches.append(batch_df)

    final_df = pd.concat(all_batches, ignore_index=True)
    sort_columns = [col for col in ["col", "row"] if col in final_df.columns]
    if sort_columns:
        final_df = final_df.sort_values(sort_columns).reset_index(drop=True)
    final_df.to_csv(output_csv, index=False)

    output_column = feature["output_column"]
    print(f"Saved {feature_name} extraction: {output_csv}")
    print(f"Rows: {len(final_df)}")
    print(f"Missing {output_column}: {final_df[output_column].isna().sum()}")
    return final_df


def _build_feature_image(ee: Any, feature: dict, month: str):
    feature_type = feature["type"]
    temporal_resolution = feature["temporal_resolution"]
    year = int(month[:4])

    if feature_type == "landcover_fraction":
        if temporal_resolution != "yearly":
            raise ValueError("landcover_fraction features must use yearly resolution.")
        return _build_landcover_fraction_image(ee, feature, year)

    if feature_type != "image":
        raise ValueError(
            f"Unsupported feature type '{feature_type}'. "
            f"Supported types: {sorted(SUPPORTED_FEATURE_TYPES)}"
        )

    if temporal_resolution == "monthly":
        start_date, end_date = month_date_range(month)
        collection = (
            ee.ImageCollection(feature["collection"])
            .filterDate(start_date, end_date)
            .select(feature["band"])
        )
        image_count = collection.size().getInfo()
        if image_count == 0:
            raise ValueError(
                f"No GEE images found for collection '{feature['collection']}' "
                f"from {start_date} to {end_date}."
            )

        image = collection.mean().rename(feature["output_column"])
        return image, {
            "period_label": "Date range",
            "period_value": f"{start_date} to {end_date}",
            "image_count": image_count,
        }

    if temporal_resolution == "yearly":
        collection, image_count, used_year, fallback_used = _get_yearly_collection(
            ee=ee,
            feature=feature,
            requested_year=year,
            error_label="GEE image",
        )

        image = collection.mosaic().rename(feature["output_column"])
        period_value = str(used_year)
        if fallback_used:
            period_value = f"{used_year} (fallback for requested {year})"

        return image, {
            "period_label": "Year",
            "period_value": period_value,
            "image_count": image_count,
        }

    raise ValueError(
        f"Unsupported temporal_resolution '{temporal_resolution}'. "
        f"Supported values: {sorted(SUPPORTED_TEMPORAL_RESOLUTIONS)}"
    )


def _build_landcover_fraction_image(ee: Any, feature: dict, year: int):
    collection, image_count, used_year, fallback_used = _get_yearly_collection(
        ee=ee,
        feature=feature,
        requested_year=year,
        error_label="land cover image",
    )

    landcover_class_value = feature.get("landcover_class_value", 13)
    image = collection.first()
    urban = image.eq(landcover_class_value).rename(feature["output_column"])

    period_value = str(used_year)
    if fallback_used:
        period_value = f"{used_year} (fallback for requested {year})"

    return urban, {
        "period_label": "Year",
        "period_value": period_value,
        "image_count": image_count,
    }


def _get_yearly_collection(
    ee: Any,
    feature: dict,
    requested_year: int,
    error_label: str,
):
    """Return a yearly ImageCollection, using feature['fallback_year'] if needed."""
    years_to_try = [requested_year]
    fallback_year = feature.get("fallback_year")
    if fallback_year is not None and int(fallback_year) != requested_year:
        years_to_try.append(int(fallback_year))

    last_count = 0
    for candidate_year in years_to_try:
        start_date = f"{candidate_year}-01-01"
        end_date = f"{candidate_year + 1}-01-01"
        collection = (
            ee.ImageCollection(feature["collection"])
            .filterDate(start_date, end_date)
            .select(feature["band"])
        )
        image_count = collection.size().getInfo()
        last_count = image_count
        if image_count > 0:
            fallback_used = candidate_year != requested_year
            if fallback_used:
                print(
                    f"Using fallback_year={candidate_year} for collection "
                    f"'{feature['collection']}' because requested year "
                    f"{requested_year} was unavailable."
                )
            return collection, image_count, candidate_year, fallback_used

    raise ValueError(
        f"No yearly {error_label} found for collection "
        f"'{feature['collection']}' in {requested_year}. "
        f"fallback_year={fallback_year}, last_image_count={last_count}."
    )


def _get_reducer(ee: Any, reducer_name: str):
    if reducer_name == "mean":
        return ee.Reducer.mean()
    if reducer_name == "sum":
        return ee.Reducer.sum()
    raise ValueError(f"Unsupported reducer: {reducer_name}. Use 'mean' or 'sum'.")


def _validate_feature_config(feature_name: str, feature: dict) -> None:
    required_keys = [
        "type",
        "collection",
        "band",
        "output_column",
        "stream",
        "scale",
        "reducer",
        "temporal_resolution",
        "fill_method",
    ]
    missing = [key for key in required_keys if key not in feature]
    if missing:
        raise ValueError(f"Feature '{feature_name}' missing required keys: {missing}")

    if feature["type"] not in SUPPORTED_FEATURE_TYPES:
        raise ValueError(
            f"Feature '{feature_name}' has unsupported type '{feature['type']}'. "
            f"Supported types: {sorted(SUPPORTED_FEATURE_TYPES)}"
        )

    if feature["temporal_resolution"] not in SUPPORTED_TEMPORAL_RESOLUTIONS:
        raise ValueError(
            f"Feature '{feature_name}' has unsupported temporal_resolution "
            f"'{feature['temporal_resolution']}'. Supported values: "
            f"{sorted(SUPPORTED_TEMPORAL_RESOLUTIONS)}"
        )

    if feature["type"] == "landcover_fraction" and "landcover_class_value" not in feature:
        raise ValueError(
            f"Feature '{feature_name}' must define landcover_class_value."
        )


def _initialize_earth_engine(project_id: str) -> None:
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    import ee

    ee.Initialize(project=project_id)
    _EE_INITIALIZED = True
    print(f"Google Earth Engine initialized with project: {project_id}")


def _extract_batch(
    geemap,
    image,
    reducer,
    reducer_name: str,
    batch_gdf,
    batch_dir,
    batch_idx: int,
    feature_name: str,
    feature: dict,
    month: str,
    tile_scale: int,
) -> pd.DataFrame:
    batch_path = batch_dir / f"{feature_name}_batch_{batch_idx:03d}.csv"
    if batch_path.exists():
        print(f"[{feature_name} batch {batch_idx:03d}] exists, skipping.")
        return pd.read_csv(batch_path)

    print(f"[{feature_name} batch {batch_idx:03d}] extracting {len(batch_gdf)} cells")
    batch_ee = geemap.geopandas_to_ee(batch_gdf)

    reduced = image.reduceRegions(
        collection=batch_ee,
        reducer=reducer,
        scale=feature["scale"],
        tileScale=tile_scale,
    )
    result_gdf = geemap.ee_to_gdf(reduced)
    result_df = pd.DataFrame(result_gdf.drop(columns="geometry", errors="ignore"))

    result_df = _normalize_output_column(result_df, feature, reducer_name)

    result_df["date"] = month
    output_column = feature["output_column"]
    keep_columns = ["grid_id"]
    keep_columns.extend(col for col in OPTIONAL_GRID_COLUMNS if col in result_df.columns)
    keep_columns.extend(["date", output_column])
    validate_required_columns(result_df, keep_columns)
    result_df = result_df[keep_columns]
    result_df.to_csv(batch_path, index=False)
    return result_df


def _normalize_output_column(
    result_df: pd.DataFrame,
    feature: dict,
    reducer_name: str,
) -> pd.DataFrame:
    output_column = feature["output_column"]
    if output_column in result_df.columns:
        return result_df

    if reducer_name in result_df.columns:
        return result_df.rename(columns={reducer_name: output_column})

    raise ValueError(
        "Could not find extracted value column after GEE reduceRegions. "
        f"Expected '{output_column}' or reducer column '{reducer_name}'. "
        f"Available columns: {list(result_df.columns)}"
    )