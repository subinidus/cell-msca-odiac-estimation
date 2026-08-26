from __future__ import annotations

import numpy as np

from src.utils.paths import ensure_parent_dir


def create_grid(config: dict):
    import geopandas as gpd
    from shapely.geometry import box

    region = config["region"]
    grid_geojson = config["paths"]["grid_geojson"]
    grid_csv = config["paths"]["grid_csv"]

    ensure_parent_dir(grid_geojson)
    ensure_parent_dir(grid_csv)

    bbox_wgs84 = gpd.GeoDataFrame(
        {"name": [f"{region['name']}_bbox"]},
        geometry=[
            box(
                region["min_lon"],
                region["min_lat"],
                region["max_lon"],
                region["max_lat"],
            )
        ],
        crs=region["output_crs"],
    )
    bbox_grid_crs = bbox_wgs84.to_crs(region["grid_crs"])

    minx, miny, maxx, maxy = bbox_grid_crs.total_bounds
    cell_size = region["cell_size_m"]

    grid_cells = []
    row_indices = []
    col_indices = []

    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(maxy - cell_size, miny - cell_size, -cell_size)

    for col, x in enumerate(x_coords):
        for row, y in enumerate(y_coords):
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))
            row_indices.append(row)
            col_indices.append(col)

    grid = gpd.GeoDataFrame(
        {"row": row_indices, "col": col_indices},
        geometry=grid_cells,
        crs=region["grid_crs"],
    )
    grid = gpd.overlay(grid, bbox_grid_crs, how="intersection").reset_index(drop=True)
    grid["grid_id"] = [f"{region['name'].upper()}_{i:05d}" for i in range(len(grid))]

    centroids = grid.geometry.centroid.to_crs(region["output_crs"])
    grid["lon"] = centroids.x
    grid["lat"] = centroids.y

    grid_output = grid.to_crs(region["output_crs"])
    grid_output.to_file(grid_geojson, driver="GeoJSON")
    grid_output.drop(columns="geometry").to_csv(grid_csv, index=False)

    print("Grid created successfully.")
    print(f"Grid cells: {len(grid_output)}")
    print(f"GeoJSON: {grid_geojson}")
    print(f"CSV: {grid_csv}")

    return grid_output

