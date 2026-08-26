from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(file_path: str | Path) -> Path:
    parent = Path(file_path).parent
    ensure_dir(parent)
    return parent


def make_date_tag(date_label: str) -> str:
    return date_label.replace("-", "_")


def month_to_odiac_yy_mm(month: str) -> tuple[str, str]:
    year, month_number = month.split("-")
    return year[-2:], month_number


def range_tag(months: list[str]) -> str:
    start_year = months[0].split("-")[0]
    end_year = months[-1].split("-")[0]
    return f"{start_year}_{end_year}"


def get_odiac_path(config: dict, month: str) -> Path:
    yy, mm = month_to_odiac_yy_mm(month)

    pattern = config["paths"].get("odiactif_pattern") or config["paths"].get(
        "odiac_tif_pattern"
    )

    if not pattern:
        raise ValueError(
            "ODIAC path pattern is missing. "
            "Please define paths.odiactif_pattern or paths.odiac_tif_pattern in config."
        )

    return Path(
        pattern.format(
            yy=yy,
            mm=mm,
            month=month,
            date_tag=make_date_tag(month),
        )
    )


def feature_raw_path(config: dict, feature_name: str, month: str) -> Path:
    feature = config["features"][feature_name]
    date_tag = make_date_tag(month)
    filename = f"{feature['stream']}_{feature_name}_{date_tag}.csv"
    return Path(config["paths"]["interim_dir"]) / filename


def feature_clean_path(config: dict, feature_name: str, month: str) -> Path:
    feature = config["features"][feature_name]
    date_tag = make_date_tag(month)
    filename = f"{feature['stream']}_{feature_name}_{date_tag}_clean.csv"
    return Path(config["paths"]["processed_dir"]) / filename


def vector_feature_clean_path(config: dict, feature_name: str, month: str) -> Path:
    return feature_clean_path(config, feature_name, month)


def feature_batch_dir(config: dict, feature_name: str, month: str) -> Path:
    date_tag = make_date_tag(month)
    return Path(config["paths"]["interim_dir"]) / f"{feature_name}_batches_{date_tag}"


def odiac_label_path(config: dict, month: str) -> Path:
    return Path(config["paths"]["processed_dir"]) / f"odiac_label_{make_date_tag(month)}.csv"


def odiac_label_cls_path(config: dict, month: str) -> Path:
    filename = f"odiac_label_{make_date_tag(month)}_cls.csv"
    return Path(config["paths"]["processed_dir"]) / filename


def final_dataset_path(config: dict, month: str) -> Path:
    filename = f"final_ms_mca_dataset_{make_date_tag(month)}.csv"
    return Path(config["paths"]["processed_dir"]) / filename


def combined_dataset_path(config: dict, months: list[str]) -> Path:
    filename = f"final_ms_mca_dataset_{range_tag(months)}.csv"
    return Path(config["paths"]["processed_dir"]) / filename


def tensor_path(config: dict, month: str) -> Path:
    region = config["region"]["name"]
    filename = f"ms_mca_tensor_{region}_{make_date_tag(month)}.npz"
    return Path(config["paths"]["processed_dir"]) / filename


def tensor_index_path(config: dict, months: list[str]) -> Path:
    filename = f"tensor_index_{range_tag(months)}.csv"
    return Path(config["paths"]["processed_dir"]) / filename


def label_thresholds_path(config: dict, months: list[str]) -> Path:
    filename = f"label_thresholds_{range_tag(months)}_global.csv"
    return Path(config["paths"]["processed_dir"]) / filename
