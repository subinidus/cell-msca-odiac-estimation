from __future__ import annotations

import pandas as pd

from src.utils.paths import ensure_parent_dir, vector_feature_clean_path
from src.utils.validation import check_file_exists, validate_required_columns


_EE_INITIALIZED = False


def _feature_collection_to_dataframe(collection) -> pd.DataFrame:
    """
    Convert an Earth Engine FeatureCollection to a pandas DataFrame without
    geemap.ee_to_df().

    geemap.ee_to_df() can fail with some pandas versions because of an internal
    DataFrame.drop(columns=..., axis=...) compatibility issue.
    """
    info = collection.getInfo()
    features = info.get("features", [])

    if not features:
        return pd.DataFrame()

    rows = []
    for item in features:
        props = item.get("properties", {}).copy()

        # If longitude/latitude are not provided as properties, recover them
        # from point geometry.
        geom = item.get("geometry")
        if geom and geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                props.setdefault("longitude", coords[0])
                props.setdefault("latitude", coords[1])

        rows.append(props)

    return pd.DataFrame(rows)


def extract_vector_feature(config: dict, feature_name: str, month: str) -> pd.DataFrame:
    if feature_name != "power_plants":
        raise ValueError(
            "Unsupported vector feature: "
            f"{feature_name}. Currently only 'power_plants' is supported."
        )

    return extract_power_plant_features(config, month)


def extract_power_plant_features(config: dict, month: str) -> pd.DataFrame:
    import ee
    import geopandas as gpd

    feature_name = "power_plants"
    feature = config["features"][feature_name]
    output_csv = vector_feature_clean_path(config, feature_name, month)
    ensure_parent_dir(output_csv)

    if config.get("run", {}).get("skip_existing", True) and output_csv.exists():
        print(f"[SKIP] Existing vector feature file found: {output_csv}")
        return pd.read_csv(output_csv)

    _initialize_earth_engine(config["project"]["gee_project_id"])

    grid_path = check_file_exists(config["paths"]["grid_geojson"])
    grid = gpd.read_file(grid_path)
    validate_required_columns(grid, ["grid_id"])
    grid = grid[["grid_id", "geometry"]].copy().to_crs("EPSG:4326")

    print("Vector feature extraction")
    print(f"Feature: {feature_name}")
    print(f"Collection: {feature['collection']}")
    print(f"Month: {month}")

    # Filter the global power plant dataset to the grid bounding box before
    # downloading metadata to pandas. This is much lighter and also avoids the
    # geemap.ee_to_df() pandas compatibility issue.
    minx, miny, maxx, maxy = grid.total_bounds
    region_geom = ee.Geometry.Rectangle([float(minx), float(miny), float(maxx), float(maxy)])

    try:
        collection = ee.FeatureCollection(feature["collection"]).filterBounds(region_geom)
        power_df = _feature_collection_to_dataframe(collection)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load power plant collection: {feature['collection']}"
        ) from exc

    print(f"Power plant rows loaded inside bounding box: {len(power_df)}")
    print(f"Available columns: {list(power_df.columns)}")

    count_column, fossil_capacity_column = _power_plant_output_columns(feature)

    # If no plants are found in the bounding box, still save a valid clean CSV
    # with zero values for every grid cell.
    if power_df.empty:
        print("[WARNING] No power plants found inside the target bounding box.")
        result_df = pd.DataFrame({"grid_id": grid["grid_id"]})
        result_df["date"] = month
        result_df[count_column] = 0
        result_df[fossil_capacity_column] = 0.0
        result_df = result_df[["grid_id", "date", count_column, fossil_capacity_column]]
        result_df.to_csv(output_csv, index=False)
        print(f"Output path: {output_csv}")
        return result_df

    _validate_power_plant_columns(power_df, feature)

    power_df = power_df.copy()
    if "grid_id" in power_df.columns:
        power_df = power_df.drop(columns=["grid_id"])

    capacity_column = feature["capacity_column"]
    fuel_column = feature["fuel_column"]
    fossil_fuels = set(feature.get("fossil_fuels", []))

    power_df[capacity_column] = pd.to_numeric(
        power_df[capacity_column],
        errors="coerce",
    ).fillna(0.0)

    power_gdf = gpd.GeoDataFrame(
        power_df,
        geometry=gpd.points_from_xy(power_df["longitude"], power_df["latitude"]),
        crs="EPSG:4326",
    )

    try:
        joined = gpd.sjoin(
            power_gdf,
            grid,
            how="inner",
            predicate="within",
        )
    except Exception as exc:
        raise RuntimeError("Spatial join failed for power plant features.") from exc

    if joined.empty:
        print("[WARNING] No power plants were assigned to grid cells after spatial join.")
        count_df = pd.DataFrame(columns=["grid_id", count_column])
        fossil_capacity_df = pd.DataFrame(columns=["grid_id", fossil_capacity_column])
        fossil_joined = joined.copy()
    else:
        joined["is_fossil"] = joined[fuel_column].isin(fossil_fuels)
        fossil_joined = joined[joined["is_fossil"]].copy()

        count_df = joined.groupby("grid_id").size().rename(count_column).reset_index()
        fossil_capacity_df = (
            fossil_joined.groupby("grid_id")[capacity_column]
            .sum()
            .rename(fossil_capacity_column)
            .reset_index()
        )

    result_df = pd.DataFrame({"grid_id": grid["grid_id"]})
    result_df = result_df.merge(count_df, on="grid_id", how="left")
    result_df = result_df.merge(fossil_capacity_df, on="grid_id", how="left")
    result_df[count_column] = result_df[count_column].fillna(0).astype("int64")
    result_df[fossil_capacity_column] = result_df[fossil_capacity_column].fillna(0.0)
    result_df["date"] = month

    result_df = result_df[["grid_id", "date", count_column, fossil_capacity_column]]
    result_df.to_csv(output_csv, index=False)

    print(f"Power plants inside grid: {len(joined)}")
    print(f"Fossil power plants inside grid: {len(fossil_joined)}")
    print(f"Output path: {output_csv}")
    print(f"{count_column} summary:")
    print(result_df[count_column].describe())
    print(f"{fossil_capacity_column} summary:")
    print(result_df[fossil_capacity_column].describe())

    return result_df


def _initialize_earth_engine(project_id: str) -> None:
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    import ee

    ee.Initialize(project=project_id)
    _EE_INITIALIZED = True
    print(f"Google Earth Engine initialized with project: {project_id}")


def _validate_power_plant_columns(power_df: pd.DataFrame, feature: dict) -> None:
    required_columns = [
        "longitude",
        "latitude",
        feature["capacity_column"],
        feature["fuel_column"],
    ]
    missing = [col for col in required_columns if col not in power_df.columns]
    if missing:
        raise ValueError(
            "Power plant data is missing required columns: "
            f"{missing}. Available columns: {list(power_df.columns)}"
        )


def _power_plant_output_columns(feature: dict) -> tuple[str, str]:
    output_columns = feature["output_columns"]
    expected = ["power_plant_count", "fossil_capacity_mw"]
    if output_columns != expected:
        raise ValueError(
            "power_plants.output_columns must be "
            f"{expected}, got: {output_columns}"
        )
    return output_columns[0], output_columns[1]
