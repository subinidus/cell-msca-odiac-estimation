from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import ensure_parent_dir, get_odiac_path, odiac_label_path
from src.utils.validation import check_file_exists, validate_required_columns


def extract_odiac_label(config: dict, month: str) -> pd.DataFrame:
    import geopandas as gpd
    import rasterio
    from rasterstats import zonal_stats

    grid_path = check_file_exists(config["paths"]["grid_geojson"])
    odiac_tif = get_odiac_path(config, month)
    if not odiac_tif.exists():
        raise FileNotFoundError(
            "Missing ODIAC GeoTIFF for monthly label extraction.\n"
            f"Expected path: {odiac_tif}"
        )

    output_csv = odiac_label_path(config, month)
    ensure_parent_dir(output_csv)

    grid = gpd.read_file(grid_path)
    validate_required_columns(grid, ["grid_id", "row", "col", "lon", "lat"])

    with rasterio.open(odiac_tif) as src:
        raster_crs = src.crs
        nodata = src.nodata
        print(f"ODIAC raster: {odiac_tif}")
        print(f"ODIAC CRS: {raster_crs}")
        print(f"ODIAC resolution: {src.res}")

    if grid.crs != raster_crs:
        print("Reprojecting grid to ODIAC raster CRS.")
        grid_for_stats = grid.to_crs(raster_crs)
    else:
        grid_for_stats = grid.copy()

    print(f"Running ODIAC zonal statistics for {month}")
    stats = zonal_stats(
        vectors=grid_for_stats,
        raster=str(odiac_tif),
        stats=["mean", "max", "min"],
        nodata=nodata,
        all_touched=True,
    )
    stats_df = pd.DataFrame(stats)

    label_column = config["label"]["label_column"]
    log_column = config["label"]["log_column"]

    result_df = grid[["grid_id", "row", "col", "lon", "lat"]].copy()
    result_df["date"] = month
    result_df[label_column] = stats_df["mean"]
    result_df[log_column] = np.log1p(result_df[label_column].clip(lower=0))
    result_df["odiac_min"] = stats_df["min"]
    result_df["odiac_max"] = stats_df["max"]

    result_df.to_csv(output_csv, index=False)
    print(f"Saved ODIAC label CSV: {output_csv}")
    print(f"Missing {label_column}: {result_df[label_column].isna().sum()}")
    return result_df
