from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.paths import get_odiac_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a monthly ODIAC GeoTIFF.")
    parser.add_argument(
        "--config",
        default="configs/delhi_2020_2024.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="Month to inspect, formatted as YYYY-MM. Defaults to config start_month.",
    )
    args = parser.parse_args()

    import rasterio
    from rasterio.windows import from_bounds

    config = load_config(args.config)
    month = args.month or config["date"]["start_month"]
    odiac_path = get_odiac_path(config, month)

    if not odiac_path.exists():
        raise FileNotFoundError(f"ODIAC GeoTIFF not found: {odiac_path}")

    region = config["region"]
    with rasterio.open(odiac_path) as src:
        print("ODIAC GeoTIFF loaded successfully.")
        print(f"File path: {odiac_path}")
        print(f"CRS: {src.crs}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Number of bands: {src.count}")
        print(f"Bounds: {src.bounds}")
        print(f"Resolution: {src.res}")
        print(f"Nodata value: {src.nodata}")
        print(f"Data type: {src.dtypes}")

        window = from_bounds(
            region["min_lon"],
            region["min_lat"],
            region["max_lon"],
            region["max_lat"],
            transform=src.transform,
        )
        arr = src.read(1, window=window)

        print(f"Region window shape: {arr.shape}")
        print(f"Raw min: {arr.min()}")
        print(f"Raw max: {arr.max()}")
        print(f"Raw mean: {arr.mean()}")


if __name__ == "__main__":
    main()

