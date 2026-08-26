from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cleaning import clean_feature
from src.preprocessing.gee_extract import extract_gee_feature
from src.preprocessing.grid import create_grid
from src.preprocessing.labeling import (
    create_label_cls_global_range,
    create_label_cls_per_month,
)
from src.preprocessing.merge import build_combined_dataset, build_final_dataset
from src.preprocessing.raster_label import extract_odiac_label
from src.preprocessing.tensor_builder import build_tensor, build_tensor_index
from src.preprocessing.vector_features import extract_vector_feature
from src.utils.config import generate_monthly_periods, load_config
from src.utils.paths import (
    ensure_dir,
    feature_clean_path,
    feature_raw_path,
    final_dataset_path,
    odiac_label_cls_path,
    odiac_label_path,
    tensor_index_path,
    tensor_path,
)
from src.utils.validation import check_file_exists


def main() -> None:
    parser = argparse.ArgumentParser(description="Run carbon hotspot preprocessing.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    months = generate_monthly_periods(config)

    _prepare_directories(config)
    _print_run_summary(config, months)

    run_flags = config["run"]

    if run_flags.get("create_grid", False):
        create_grid(config)
        check_file_exists(config["paths"]["grid_geojson"])
        check_file_exists(config["paths"]["grid_csv"])

    for month in months:
        print(f"\n========== Processing {month} ==========")

        if run_flags.get("extract_features", False):
            for feature_name, feature_cfg in config["features"].items():
                feature_type = feature_cfg.get("type", "image")
                if feature_type in {"image", "landcover_fraction"}:
                    extract_gee_feature(config, feature_name, month)
                    check_file_exists(feature_raw_path(config, feature_name, month))
                elif feature_type == "vector_power_plant":
                    extract_vector_feature(config, feature_name, month)
                    check_file_exists(feature_clean_path(config, feature_name, month))
                else:
                    raise ValueError(f"Unsupported feature type: {feature_type}")

        if run_flags.get("clean_features", False):
            for feature_name, feature_cfg in config["features"].items():
                feature_type = feature_cfg.get("type", "image")
                if feature_type in {"image", "landcover_fraction"}:
                    clean_feature(config, feature_name, month)
                    check_file_exists(feature_clean_path(config, feature_name, month))
                elif feature_type == "vector_power_plant":
                    check_file_exists(feature_clean_path(config, feature_name, month))
                else:
                    raise ValueError(f"Unsupported feature type: {feature_type}")

        if run_flags.get("extract_label", False):
            extract_odiac_label(config, month)
            check_file_exists(odiac_label_path(config, month))

        if (
            run_flags.get("create_label_cls", False)
            and config["label"].get("quantile_scope") == "per_month"
        ):
            create_label_cls_per_month(config, month)
            check_file_exists(odiac_label_cls_path(config, month))

    if run_flags.get("create_label_cls", False):
        quantile_scope = config["label"].get("quantile_scope")
        if quantile_scope == "global_range":
            print("\n========== Creating global-range label classes ==========")
            create_label_cls_global_range(config, months)
            for month in months:
                check_file_exists(odiac_label_cls_path(config, month))
        elif quantile_scope != "per_month":
            raise ValueError(
                "label.quantile_scope must be 'per_month' or 'global_range'"
            )

    if run_flags.get("build_final_dataset", False):
        print("\n========== Building monthly final datasets ==========")
        for month in months:
            build_final_dataset(config, month)
            check_file_exists(final_dataset_path(config, month))

    if run_flags.get("build_combined_dataset", False):
        print("\n========== Building combined final dataset ==========")
        build_combined_dataset(config, months)

    if run_flags.get("build_tensor", False):
        print("\n========== Building monthly tensors ==========")
        for month in months:
            build_tensor(config, month)
            check_file_exists(tensor_path(config, month))

    if run_flags.get("build_tensor_index", False):
        print("\n========== Building tensor index ==========")
        build_tensor_index(config, months)
        check_file_exists(tensor_index_path(config, months))

    print("\nPreprocessing pipeline completed successfully.")


def _prepare_directories(config: dict) -> None:
    for key in ["raw_dir", "interim_dir", "processed_dir"]:
        if key in config["paths"]:
            ensure_dir(config["paths"][key])

    ensure_dir(Path(config["paths"]["grid_geojson"]).parent)
    ensure_dir(Path(config["paths"]["grid_csv"]).parent)
    ensure_dir(Path(config["paths"]["raw_dir"]) / "odiac")


def _print_run_summary(config: dict, months: list[str]) -> None:
    print("Carbon hotspot preprocessing")
    print(f"Project: {config['project']['name']}")
    print(f"Region: {config['region']['name']}")
    print(f"Temporal mode: {config['temporal']['mode']}")
    print(f"Months: {months[0]} to {months[-1]} ({len(months)} months)")
    print(f"Features: {', '.join(config['features'].keys())}")


if __name__ == "__main__":
    main()
