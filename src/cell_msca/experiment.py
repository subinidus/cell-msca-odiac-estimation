"""Executable validation-only runner for the Phase 3 baseline suite."""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import (
    LIGHTGBM_MODEL_NAMES,
    BaselineDataProtocol,
    EarlyStoppingFactory,
    EstimatorFactory,
    LightGBMConfig,
    TrainMeanConfig,
    ValidationSelection,
    fit_lightgbm_baseline,
    fit_train_mean,
    tune_on_validation,
)
from .data import CellDataset, file_sha256
from .evaluate import (
    verify_baseline_result_from_prediction_csv,
    write_metrics_json,
    write_prediction_csv,
)
from .neural_baselines import ConcatMLPConfig, fit_concat_mlp
from .splits import CellFixedSplitConfig

BASELINE_EXPERIMENT_SCHEMA_VERSION = "cell_msca.baseline_experiment.v1"
BASELINE_SUITE_MODELS = (
    "train_mean",
    *LIGHTGBM_MODEL_NAMES,
    "concat_mlp",
)


@dataclass(frozen=True)
class BaselineExperimentConfig:
    config_path: Path
    project_root: Path
    npz_glob: str
    split_csv: Path
    split_metadata_json: Path
    output_dir: Path
    split_config: CellFixedSplitConfig
    train_seed: int
    target_scale: float
    legacy_data_version: str | None
    legacy_target_unit: str | None
    models: Mapping[str, Any]


@dataclass(frozen=True)
class BaselineSuiteTuningResult:
    selections: Mapping[str, ValidationSelection]
    artifacts: ValidationArtifactSet | None = None

    def __post_init__(self) -> None:
        if set(self.selections) != set(BASELINE_SUITE_MODELS):
            raise ValueError("baseline suite result must contain all five models")
        selected_rows = [
            selection.validation_evaluation.result
            for selection in self.selections.values()
        ]
        for field in (
            "data_sha256",
            "split_sha256",
            "split_config_sha256",
            "preprocessing_sha256",
        ):
            if len({row[field] for row in selected_rows}) != 1:
                raise ValueError(f"baseline suite has inconsistent {field}")

    @property
    def selected_validation_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.selections[name].validation_evaluation.result
            for name in BASELINE_SUITE_MODELS
        )


@dataclass(frozen=True)
class ValidationArtifactSet:
    output_dir: Path
    candidate_results_json: Path
    selected_results_json: Path
    prediction_csvs: tuple[Path, ...]
    manifest_json: Path


def load_baseline_experiment_config(
    path: str | Path,
) -> BaselineExperimentConfig:
    """Load and validate the executable example's validation-only contract."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("baseline config root must be an object")
    if values.get("schema_version") != BASELINE_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported baseline experiment schema_version")
    if values.get("stage") != "validation_tuning":
        raise ValueError(
            "baseline config stage must be validation_tuning; test is gated in code"
        )
    project_root = (config_path.parent / values.get("project_root", "..")).resolve()
    split_values = values.get("split")
    if not isinstance(split_values, dict):
        raise ValueError("baseline config requires a split object")
    split_config = CellFixedSplitConfig(
        split_seed=int(split_values["split_seed"]),
        validation_ratio=float(split_values["validation_ratio"]),
        test_ratio=float(split_values["test_ratio"]),
    )
    if split_config.split_seed != 42:
        raise ValueError("baseline example requires split_seed=42")
    models = values.get("models")
    if not isinstance(models, dict) or set(models) != set(BASELINE_SUITE_MODELS):
        raise ValueError(
            "models must contain exactly the five Phase 3 baseline entries"
        )
    for model_name, candidates in models.items():
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{model_name} must have a non-empty candidate list")
        if not all(isinstance(candidate, dict) for candidate in candidates):
            raise ValueError(f"{model_name} candidates must be objects")
    data_values = values.get("data")
    if not isinstance(data_values, dict):
        raise ValueError("baseline config requires a data object")
    legacy_data_version = data_values.get("legacy_data_version")
    legacy_target_unit = data_values.get("legacy_target_unit")
    if (legacy_data_version is None) != (legacy_target_unit is None):
        raise ValueError("legacy data version and target unit must be supplied together")
    return BaselineExperimentConfig(
        config_path=config_path,
        project_root=project_root,
        npz_glob=str(data_values["npz_glob"]),
        split_csv=(project_root / str(values["split_csv"])).resolve(),
        split_metadata_json=(
            project_root / str(values["split_metadata_json"])
        ).resolve(),
        output_dir=(project_root / str(values["output_dir"])).resolve(),
        split_config=split_config,
        train_seed=int(values["train_seed"]),
        target_scale=float(data_values.get("target_scale", 1.0)),
        legacy_data_version=(
            None if legacy_data_version is None else str(legacy_data_version)
        ),
        legacy_target_unit=(
            None if legacy_target_unit is None else str(legacy_target_unit)
        ),
        models=models,
    )


def save_validation_tuning_artifacts(
    config: BaselineExperimentConfig,
    result: BaselineSuiteTuningResult,
    validation_split: Any,
) -> ValidationArtifactSet:
    """Save and round-trip-verify all validation artifacts without overwrite."""

    output_dir = config.output_dir
    candidate_results_path = output_dir / "candidate_results.json"
    selected_results_path = output_dir / "selected_results.json"
    manifest_path = output_dir / "validation_run_manifest.json"
    prediction_paths: list[Path] = []
    candidate_entries: list[dict[str, Any]] = []
    for model_name in BASELINE_SUITE_MODELS:
        selection = result.selections[model_name]
        for candidate_index, evaluation in enumerate(
            selection.candidate_evaluations,
            start=1,
        ):
            prediction_paths.append(
                output_dir
                / (
                    f"{model_name}.candidate_{candidate_index:03d}."
                    "validation_predictions.csv"
                )
            )
            candidate_entries.append(
                {
                    "candidate_id": f"{model_name}.candidate_{candidate_index:03d}",
                    "selected": candidate_index - 1 == selection.selected_index,
                    "prediction_csv": prediction_paths[-1].name,
                    "result": evaluation.result,
                }
            )

    all_paths = [
        candidate_results_path,
        selected_results_path,
        manifest_path,
        *prediction_paths,
    ]
    existing = [str(path) for path in all_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "validation artifacts already exist; refusing to overwrite: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_index = 0
    for model_name in BASELINE_SUITE_MODELS:
        selection = result.selections[model_name]
        for evaluation in selection.candidate_evaluations:
            prediction_path = prediction_paths[entry_index]
            write_prediction_csv(
                prediction_path,
                y_true_original=validation_split.target_original,
                y_pred_original=evaluation.predictions.pred_original,
                y_true_log=validation_split.target_log,
                y_pred_log=evaluation.predictions.pred_log,
                cell_ids=validation_split.cell_ids,
            )
            verify_baseline_result_from_prediction_csv(
                prediction_path,
                evaluation.result,
            )
            candidate_entries[entry_index]["prediction_metric_verification"] = (
                "passed_including_spearman"
            )
            entry_index += 1

    experiment_config_sha256 = file_sha256(config.config_path)
    selected_rows = list(result.selected_validation_rows)
    first_row = selected_rows[0]
    candidate_payload = {
        "schema_version": "cell_msca.validation_candidate_results.v1",
        "experiment_config_sha256": experiment_config_sha256,
        "candidates": candidate_entries,
    }
    selected_payload = {
        "schema_version": "cell_msca.validation_selected_results.v1",
        "experiment_config_sha256": experiment_config_sha256,
        "results": selected_rows,
    }
    manifest = {
        "schema_version": "cell_msca.validation_run_manifest.v1",
        "stage": "validation_tuning",
        "test_subset_materialized": False,
        "experiment_config_sha256": experiment_config_sha256,
        "data_sha256": first_row["data_sha256"],
        "split_sha256": first_row["split_sha256"],
        "split_config_sha256": first_row["split_config_sha256"],
        "preprocessing_sha256": first_row["preprocessing_sha256"],
        "split_seed": first_row["split_seed"],
        "train_seed": first_row["train_seed"],
        "candidate_results_json": candidate_results_path.name,
        "selected_results_json": selected_results_path.name,
        "prediction_csvs": [path.name for path in prediction_paths],
        "prediction_metric_verification": "passed_for_all_including_spearman",
    }
    write_metrics_json(candidate_results_path, candidate_payload)
    write_metrics_json(selected_results_path, selected_payload)
    write_metrics_json(manifest_path, manifest)
    return ValidationArtifactSet(
        output_dir=output_dir,
        candidate_results_json=candidate_results_path,
        selected_results_json=selected_results_path,
        prediction_csvs=tuple(prediction_paths),
        manifest_json=manifest_path,
    )


def run_validation_tuning_from_config(
    config_path: str | Path,
    *,
    estimator_factory: EstimatorFactory | None = None,
    early_stopping_factory: EarlyStoppingFactory | None = None,
    concat_fit_candidate: Callable[[Any, ConcatMLPConfig], Any] = fit_concat_mlp,
) -> BaselineSuiteTuningResult:
    """Run five baselines without materializing or evaluating the test subset."""

    config = load_baseline_experiment_config(config_path)
    pattern = str((config.project_root / config.npz_glob).resolve())
    npz_paths = [Path(value) for value in sorted(glob.glob(pattern))]
    if not npz_paths:
        raise FileNotFoundError(f"baseline data glob matched no NPZ files: {pattern}")
    dataset = CellDataset(
        npz_paths,
        target_scale=config.target_scale,
        legacy_data_version=config.legacy_data_version,
        legacy_target_unit=config.legacy_target_unit,
    )
    protocol = BaselineDataProtocol.from_dataset(
        dataset,
        config.split_csv,
        config.split_metadata_json,
        split_config=config.split_config,
        train_seed=config.train_seed,
    )
    data = protocol.tuning_data()
    selections: dict[str, ValidationSelection] = {}
    mean_candidates = [
        TrainMeanConfig(**candidate) for candidate in config.models["train_mean"]
    ]
    selections["train_mean"] = tune_on_validation(
        data,
        mean_candidates,
        fit_train_mean,
    )
    for model_name in LIGHTGBM_MODEL_NAMES:
        candidates = [
            LightGBMConfig(model_name=model_name, **candidate)
            for candidate in config.models[model_name]
        ]

        def fit_lightgbm(data_value: Any, candidate: LightGBMConfig) -> Any:
            return fit_lightgbm_baseline(
                data_value,
                candidate,
                estimator_factory=estimator_factory,
                early_stopping_factory=early_stopping_factory,
            )

        selections[model_name] = tune_on_validation(
            data,
            candidates,
            fit_lightgbm,
        )
    mlp_candidates = [
        ConcatMLPConfig(**candidate) for candidate in config.models["concat_mlp"]
    ]
    selections["concat_mlp"] = tune_on_validation(
        data,
        mlp_candidates,
        concat_fit_candidate,
    )
    unsaved_result = BaselineSuiteTuningResult(selections)
    artifacts = save_validation_tuning_artifacts(
        config,
        unsaved_result,
        data.validation,
    )
    return BaselineSuiteTuningResult(selections, artifacts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 validation-only baseline tuning",
    )
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = run_validation_tuning_from_config(arguments.config)
    if result.artifacts is None:
        raise RuntimeError("validation runner completed without saved artifacts")
    print(
        json.dumps(
            {
                "selected_results": result.selected_validation_rows,
                "validation_run_manifest": str(result.artifacts.manifest_json),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
