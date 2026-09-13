"""Phase 3 baseline models under one leakage-controlled data protocol."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .data import (
    STREAM_A_FEATURES,
    STREAM_B_FEATURES,
    CellDataset,
    PreprocessingStats,
    canonical_sha256,
    fit_train_preprocessing,
)
from .evaluate import (
    BaselineResultContext,
    evaluate_baseline_predictions,
    validate_baseline_result_row,
    write_metrics_json,
)
from .metrics import regression_metrics
from .splits import (
    CellFixedSplitConfig,
    SplitManifest,
    load_persistent_split,
)
from .target import (
    NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
    PredictionSupportDiagnostics,
    duan_smearing_factor,
    inverse_target,
    project_nonnegative_predictions,
    target_transform,
)

LIGHTGBM_MODEL_NAMES = (
    "lightgbm_raw",
    "lightgbm_log1p",
    "lightgbm_tweedie",
)
FROZEN_BASELINE_SELECTION_SCHEMA_VERSION = (
    "cell_msca.frozen_baseline_selection.v1"
)
_FINAL_TEST_AUTHORITY = object()


class TestEvaluationBlockedError(RuntimeError):
    """Raised when test data is requested before validation choices are frozen."""


@dataclass(frozen=True)
class BaselineArraySplit:
    """Materialized seven-feature arrays for one explicitly named split."""

    name: str
    features: NDArray[np.float64]
    target_original: NDArray[np.float64]
    target_log: NDArray[np.float64]
    cell_ids: NDArray[np.str_]
    month_ids: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        target_original = np.asarray(self.target_original, dtype=np.float64).ravel()
        target_log = np.asarray(self.target_log, dtype=np.float64).ravel()
        cell_ids = np.asarray(self.cell_ids, dtype=str).ravel()
        month_ids = (
            None
            if self.month_ids is None
            else np.asarray(self.month_ids, dtype=str).ravel()
        )
        expected_features = len(STREAM_A_FEATURES) + len(STREAM_B_FEATURES)
        if features.ndim != 2 or features.shape[1] != expected_features:
            raise ValueError(
                f"features must have shape [N, {expected_features}]; got {features.shape}"
            )
        lengths = {
            features.shape[0],
            target_original.size,
            target_log.size,
            cell_ids.size,
        }
        if len(lengths) != 1 or not lengths or features.shape[0] == 0:
            raise ValueError("baseline split arrays must have one non-zero sample count")
        if month_ids is not None and month_ids.size != features.shape[0]:
            raise ValueError("month_ids length does not match baseline samples")
        if not np.all(np.isfinite(features)):
            raise ValueError("baseline features must contain only finite values")
        if not np.all(np.isfinite(target_original)) or np.any(target_original < 0.0):
            raise ValueError("original targets must be finite and non-negative")
        if not np.all(np.isfinite(target_log)):
            raise ValueError("log targets must contain only finite values")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "target_original", target_original)
        object.__setattr__(self, "target_log", target_log)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "month_ids", month_ids)

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class BaselineProvenance:
    data_version: str
    data_sha256: str
    split_sha256: str
    split_config_sha256: str
    preprocessing_sha256: str
    split_seed: int
    train_seed: int
    target_scale: float


@dataclass(frozen=True)
class TuningData:
    """Train/validation-only materialization passed to tuning functions."""

    train: BaselineArraySplit
    validation: BaselineArraySplit
    provenance: BaselineProvenance

    def subset(self, name: str) -> BaselineArraySplit:
        if name == "train":
            return self.train
        if name == "validation":
            return self.validation
        if name == "test":
            raise TestEvaluationBlockedError(
                "test subset is unavailable during validation-only tuning"
            )
        raise ValueError(f"unknown split name: {name!r}")


@dataclass(frozen=True)
class TestEvaluationAuthorization:
    data_sha256: str
    split_sha256: str
    split_config_sha256: str
    preprocessing_sha256: str
    selected_config_sha256: str
    model_name: str
    _authority: object = field(repr=False, compare=False)


class BaselineDataProtocol:
    """One verified split and one train-fitted preprocessing contract for all models."""

    def __init__(
        self,
        dataset: CellDataset,
        manifest: SplitManifest,
        preprocessing: PreprocessingStats,
        split_config: CellFixedSplitConfig,
        *,
        train_seed: int,
    ) -> None:
        self._dataset = dataset
        self._manifest = manifest
        self.preprocessing = preprocessing
        self.split_config = split_config
        self.provenance = BaselineProvenance(
            data_version=dataset.feature_metadata.data_version,
            data_sha256=dataset.data_sha256,
            split_sha256=manifest.split_sha256,
            split_config_sha256=canonical_sha256(split_config.to_dict()),
            preprocessing_sha256=preprocessing.sha256,
            split_seed=split_config.split_seed,
            train_seed=int(train_seed),
            target_scale=dataset.target_scale,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: CellDataset,
        split_csv: str | Path,
        metadata_json: str | Path,
        *,
        split_config: CellFixedSplitConfig | None = None,
        train_seed: int = 42,
    ) -> BaselineDataProtocol:
        """Validate metadata/data hashes before fitting train-only preprocessing."""

        config = split_config or CellFixedSplitConfig()
        if config.split_seed != 42:
            raise ValueError("Phase 3 baseline protocol requires split_seed=42")
        if not isinstance(train_seed, int):
            raise ValueError("train_seed must be an integer")
        manifest = load_persistent_split(
            dataset,
            split_csv,
            metadata_json=metadata_json,
            config=config,
        )
        if manifest.split_mode != "cell_fixed":
            raise ValueError("Phase 3 baselines require the cell-fixed split")
        preprocessing = fit_train_preprocessing(
            dataset,
            manifest.cell_ids("train"),
            split_sha256=manifest.split_sha256,
            split_name="train",
        )
        dataset.set_preprocessing(
            preprocessing,
            split_sha256=manifest.split_sha256,
            split_name="train",
        )
        return cls(
            dataset,
            manifest,
            preprocessing,
            config,
            train_seed=train_seed,
        )

    def tuning_data(self) -> TuningData:
        """Materialize train and validation only; test is deliberately absent."""

        return TuningData(
            train=self._materialize("train"),
            validation=self._materialize("validation"),
            provenance=self.provenance,
        )

    def test_data(
        self,
        authorization: TestEvaluationAuthorization | None = None,
    ) -> BaselineArraySplit:
        """Materialize test only after validation selection is explicitly frozen."""

        if authorization is None or authorization._authority is not _FINAL_TEST_AUTHORITY:
            raise TestEvaluationBlockedError(
                "test evaluation requires an authorization from a frozen "
                "validation-only selection"
            )
        if authorization.data_sha256 != self.provenance.data_sha256:
            raise TestEvaluationBlockedError("test authorization data hash mismatch")
        if authorization.split_sha256 != self.provenance.split_sha256:
            raise TestEvaluationBlockedError("test authorization split hash mismatch")
        if authorization.split_config_sha256 != self.provenance.split_config_sha256:
            raise TestEvaluationBlockedError(
                "test authorization split config hash mismatch"
            )
        if authorization.preprocessing_sha256 != self.provenance.preprocessing_sha256:
            raise TestEvaluationBlockedError(
                "test authorization preprocessing hash mismatch"
            )
        return self._materialize("test")

    def _materialize(self, split: str) -> BaselineArraySplit:
        indices = self._manifest.sample_indices(self._dataset, split)
        n_samples = len(indices)
        n_features = len(STREAM_A_FEATURES) + len(STREAM_B_FEATURES)
        features = np.empty((n_samples, n_features), dtype=np.float64)
        target_original = np.empty(n_samples, dtype=np.float64)
        target_log = np.empty(n_samples, dtype=np.float64)
        cell_ids = np.empty(n_samples, dtype=object)
        month_ids = np.empty(n_samples, dtype=object)
        for output_index, dataset_index in enumerate(indices):
            sample = self._dataset[dataset_index]
            features[output_index] = np.concatenate(
                (sample["stream_a"], sample["stream_b"])
            )
            target_original[output_index] = sample["target_original"]
            target_log[output_index] = sample["target_log"]
            cell_ids[output_index] = sample["cell_id"]
            month_ids[output_index] = sample["month_id"]
        return BaselineArraySplit(
            name=split,
            features=features,
            target_original=target_original,
            target_log=target_log,
            cell_ids=cell_ids.astype(str),
            month_ids=month_ids.astype(str),
        )


@dataclass(frozen=True)
class BaselinePredictions:
    pred_original: NDArray[np.float64]
    pred_log: NDArray[np.float64]
    unprojected_original: NDArray[np.float64] | None = None
    support_diagnostics: PredictionSupportDiagnostics | None = None

    def __post_init__(self) -> None:
        pred_original = np.asarray(self.pred_original, dtype=np.float64).ravel()
        pred_log = np.asarray(self.pred_log, dtype=np.float64).ravel()
        if pred_original.shape != pred_log.shape or pred_original.size == 0:
            raise ValueError("original/log predictions must have one non-zero shape")
        if not np.all(np.isfinite(pred_original)) or not np.all(np.isfinite(pred_log)):
            raise ValueError("baseline predictions must contain only finite values")
        unprojected_original = (
            pred_original.copy()
            if self.unprojected_original is None
            else np.asarray(self.unprojected_original, dtype=np.float64).ravel()
        )
        if unprojected_original.shape != pred_original.shape:
            raise ValueError("unprojected/final original predictions must align")
        projected, expected_diagnostics = project_nonnegative_predictions(
            unprojected_original
        )
        if not np.array_equal(pred_original, projected):
            raise ValueError(
                "pred_original must equal the recorded nonnegative support projection"
            )
        diagnostics = self.support_diagnostics or expected_diagnostics
        if diagnostics != expected_diagnostics:
            raise ValueError("prediction support diagnostics do not match predictions")
        object.__setattr__(self, "pred_original", pred_original)
        object.__setattr__(self, "pred_log", pred_log)
        object.__setattr__(self, "unprojected_original", unprojected_original)
        object.__setattr__(self, "support_diagnostics", diagnostics)


@dataclass(frozen=True)
class LogInverseSelection:
    inverse_mode: Literal["median", "duan_smearing"]
    smearing_factor: float | None
    validation_original_mae: float


@dataclass(frozen=True)
class ModelContract:
    model_name: str
    config_sha256: str
    target_transform: Literal["identity", "log1p"]
    target_scale: float
    loss_objective: str
    inverse_mode: Literal["none", "median", "duan_smearing"]
    smearing_factor: float | None
    prediction_support_policy: str = NONNEGATIVE_PREDICTION_SUPPORT_POLICY
    best_iteration: int | None = None
    best_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.prediction_support_policy != NONNEGATIVE_PREDICTION_SUPPORT_POLICY:
            raise ValueError("model contract has an unsupported prediction policy")


class FittedBaseline(Protocol):
    contract: ModelContract

    def predict(self, features: NDArray[np.float64]) -> BaselinePredictions: ...


@dataclass(frozen=True)
class TrainMeanConfig:
    model_name: str = "train_mean"

    def __post_init__(self) -> None:
        if self.model_name != "train_mean":
            raise ValueError("train mean model_name must be 'train_mean'")

    def to_dict(self, *, train_seed: int, target_scale: float) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "train_seed": train_seed,
            "target_scale": target_scale,
            "objective": "mean_original_target",
            "prediction_support_policy": NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
        }


@dataclass(frozen=True)
class FittedTrainMean:
    mean_original: float
    contract: ModelContract

    def predict(self, features: NDArray[np.float64]) -> BaselinePredictions:
        n_samples = _validated_feature_count(features)
        pred_original = np.full(n_samples, self.mean_original, dtype=np.float64)
        return _original_prediction_pair(
            pred_original,
            target_scale=self.contract.target_scale,
            model_name=self.contract.model_name,
        )


def fit_train_mean(
    data: TuningData,
    config: TrainMeanConfig | None = None,
) -> FittedTrainMean:
    """Fit the original-target mean using train samples only."""

    config = config or TrainMeanConfig()
    mean_original = float(np.mean(data.train.target_original))
    config_sha256 = canonical_sha256(
        config.to_dict(
            train_seed=data.provenance.train_seed,
            target_scale=data.provenance.target_scale,
        )
    )
    return FittedTrainMean(
        mean_original=mean_original,
        contract=ModelContract(
            model_name=config.model_name,
            config_sha256=config_sha256,
            target_transform="identity",
            target_scale=data.provenance.target_scale,
            loss_objective="mean_original_target",
            inverse_mode="none",
            smearing_factor=None,
        ),
    )


@dataclass(frozen=True)
class LightGBMConfig:
    model_name: Literal[
        "lightgbm_raw",
        "lightgbm_log1p",
        "lightgbm_tweedie",
    ]
    n_estimators: int = 500
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 0.0
    early_stopping_rounds: int = 50
    tweedie_variance_power: float = 1.5

    def __post_init__(self) -> None:
        if self.model_name not in LIGHTGBM_MODEL_NAMES:
            raise ValueError(f"unsupported LightGBM baseline: {self.model_name!r}")
        if self.n_estimators <= 0 or self.early_stopping_rounds <= 0:
            raise ValueError("LightGBM iteration counts must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.num_leaves <= 1 or self.min_child_samples <= 0:
            raise ValueError("num_leaves and min_child_samples are invalid")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if not np.isfinite(self.reg_lambda) or self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be finite and non-negative")
        if not 1.0 < self.tweedie_variance_power < 2.0:
            raise ValueError("tweedie_variance_power must be in (1, 2)")

    @property
    def target_space(self) -> Literal["original", "log"]:
        return "log" if self.model_name == "lightgbm_log1p" else "original"

    @property
    def objective(self) -> str:
        if self.model_name == "lightgbm_tweedie":
            return "tweedie"
        return "regression_l1"

    def to_dict(self, *, train_seed: int, target_scale: float) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_space": self.target_space,
            "target_scale": target_scale,
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "tweedie_variance_power": self.tweedie_variance_power,
            "train_seed": train_seed,
            "prediction_support_policy": NONNEGATIVE_PREDICTION_SUPPORT_POLICY,
        }

    def estimator_params(self, *, train_seed: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "objective": self.objective,
            "metric": "None",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": train_seed,
            "data_random_seed": train_seed,
            "feature_fraction_seed": train_seed,
            "bagging_seed": train_seed,
            "deterministic": True,
            "n_jobs": 1,
            "verbosity": -1,
        }
        if self.model_name == "lightgbm_tweedie":
            params["tweedie_variance_power"] = self.tweedie_variance_power
        return params


@dataclass
class FittedLightGBM:
    estimator: Any
    target_space: Literal["original", "log"]
    contract: ModelContract

    def predict(self, features: NDArray[np.float64]) -> BaselinePredictions:
        _validated_feature_count(features)
        model_prediction = np.asarray(
            self.estimator.predict(np.asarray(features, dtype=np.float64)),
            dtype=np.float64,
        ).ravel()
        if self.target_space == "log":
            return _log_prediction_pair(
                model_prediction,
                scale=self.contract.target_scale,
                mode=self.contract.inverse_mode,
                smearing_factor=self.contract.smearing_factor,
            )
        return _original_prediction_pair(
            model_prediction,
            target_scale=self.contract.target_scale,
            model_name=self.contract.model_name,
        )


EstimatorFactory = Callable[..., Any]
EarlyStoppingFactory = Callable[[int], Any]


def fit_lightgbm_baseline(
    data: TuningData,
    config: LightGBMConfig,
    *,
    estimator_factory: EstimatorFactory | None = None,
    early_stopping_factory: EarlyStoppingFactory | None = None,
) -> FittedLightGBM:
    """Fit one LightGBM variant with validation as its sole eval_set."""

    if estimator_factory is None:
        try:
            import lightgbm as lgb
        except ImportError as error:
            raise ImportError(
                "LightGBM baselines require the optional 'lightgbm' package"
            ) from error
        estimator_factory = lgb.LGBMRegressor
        if early_stopping_factory is None:
            early_stopping_factory = lambda rounds: lgb.early_stopping(
                rounds,
                verbose=False,
            )
    estimator = estimator_factory(
        **config.estimator_params(train_seed=data.provenance.train_seed)
    )
    if config.target_space == "log":
        train_target = data.train.target_log
        validation_target = data.validation.target_log
    else:
        train_target = data.train.target_original
        validation_target = data.validation.target_original
    fit_kwargs: dict[str, Any] = {
        "eval_names": ["validation"],
        "eval_metric": _original_unit_mae_metric(
            target_space=config.target_space,
            target_scale=data.provenance.target_scale,
        ),
    }
    fit_parameters = inspect.signature(estimator.fit).parameters
    if "eval_X" in fit_parameters and "eval_y" in fit_parameters:
        fit_kwargs["eval_X"] = data.validation.features
        fit_kwargs["eval_y"] = validation_target
    else:
        fit_kwargs["eval_set"] = [(data.validation.features, validation_target)]
    if early_stopping_factory is not None:
        fit_kwargs["callbacks"] = [
            early_stopping_factory(config.early_stopping_rounds)
        ]
    estimator.fit(data.train.features, train_target, **fit_kwargs)
    best_iteration = int(
        getattr(estimator, "best_iteration_", 0) or config.n_estimators
    )
    config_sha256 = canonical_sha256(
        config.to_dict(
            train_seed=data.provenance.train_seed,
            target_scale=data.provenance.target_scale,
        )
    )
    if config.target_space == "log":
        selection = select_log_inverse_on_validation(
            train_true_log=data.train.target_log,
            train_pred_log=np.asarray(
                estimator.predict(data.train.features),
                dtype=np.float64,
            ),
            validation_true_original=data.validation.target_original,
            validation_pred_log=np.asarray(
                estimator.predict(data.validation.features),
                dtype=np.float64,
            ),
            target_scale=data.provenance.target_scale,
        )
        inverse_mode = selection.inverse_mode
        smearing_factor = selection.smearing_factor
        target_transform_name: Literal["identity", "log1p"] = "log1p"
        loss_objective = f"{config.objective}_on_log1p_target"
    else:
        inverse_mode = "none"
        smearing_factor = None
        target_transform_name = "identity"
        loss_objective = config.objective
    return FittedLightGBM(
        estimator=estimator,
        target_space=config.target_space,
        contract=ModelContract(
            model_name=config.model_name,
            config_sha256=config_sha256,
            target_transform=target_transform_name,
            target_scale=data.provenance.target_scale,
            loss_objective=loss_objective,
            inverse_mode=inverse_mode,
            smearing_factor=smearing_factor,
            best_iteration=best_iteration,
        ),
    )


def select_log_inverse_on_validation(
    *,
    train_true_log: NDArray[np.float64],
    train_pred_log: NDArray[np.float64],
    validation_true_original: NDArray[np.float64],
    validation_pred_log: NDArray[np.float64],
    target_scale: float,
) -> LogInverseSelection:
    """Select median or Duan using train residuals and validation raw MAE."""

    factor = duan_smearing_factor(train_pred_log, train_true_log)
    median_unprojected = inverse_target(
        validation_pred_log,
        scale=target_scale,
        mode="median",
    )
    duan_unprojected = inverse_target(
        validation_pred_log,
        scale=target_scale,
        mode="duan_smearing",
        smearing_factor=factor,
    )
    median_prediction, _ = project_nonnegative_predictions(median_unprojected)
    duan_prediction, _ = project_nonnegative_predictions(duan_unprojected)
    median_mae = float(
        regression_metrics(validation_true_original, median_prediction)["mae"]
    )
    duan_mae = float(
        regression_metrics(validation_true_original, duan_prediction)["mae"]
    )
    if duan_mae < median_mae:
        return LogInverseSelection("duan_smearing", factor, duan_mae)
    return LogInverseSelection("median", None, median_mae)


def _original_unit_mae_metric(
    *,
    target_space: Literal["original", "log"],
    target_scale: float,
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], tuple[str, float, bool]]:
    def metric(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> tuple[str, float, bool]:
        if target_space == "log":
            true_original = inverse_target(y_true, scale=target_scale, mode="median")
            unprojected_prediction = inverse_target(
                y_pred,
                scale=target_scale,
                mode="median",
            )
        else:
            true_original = np.asarray(y_true, dtype=np.float64)
            unprojected_prediction = np.asarray(y_pred, dtype=np.float64)
        pred_original, _ = project_nonnegative_predictions(
            unprojected_prediction
        )
        mae = regression_metrics(true_original, pred_original)["mae"]
        return "original_unit_mae", float(mae), False

    return metric


def _validated_feature_count(features: NDArray[np.float64]) -> int:
    array = np.asarray(features, dtype=np.float64)
    expected_features = len(STREAM_A_FEATURES) + len(STREAM_B_FEATURES)
    if array.ndim != 2 or array.shape[1] != expected_features:
        raise ValueError(
            f"features must have shape [N, {expected_features}]; got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("prediction features must contain only finite values")
    return int(array.shape[0])


def _original_prediction_pair(
    pred_original: NDArray[np.float64],
    *,
    target_scale: float,
    model_name: str,
) -> BaselinePredictions:
    if not model_name:
        raise ValueError("model_name must not be empty")
    unprojected_original = np.asarray(pred_original, dtype=np.float64).ravel()
    final_original, diagnostics = project_nonnegative_predictions(
        unprojected_original
    )
    pred_log = target_transform(final_original, scale=target_scale)
    return BaselinePredictions(
        final_original,
        pred_log,
        unprojected_original,
        diagnostics,
    )


def _log_prediction_pair(
    pred_log: NDArray[np.float64],
    *,
    scale: float,
    mode: Literal["median", "duan_smearing"],
    smearing_factor: float | None,
) -> BaselinePredictions:
    """Keep native log predictions while projecting their original-unit inverse."""

    native_pred_log = np.asarray(pred_log, dtype=np.float64).ravel()
    unprojected_original = inverse_target(
        native_pred_log,
        scale=scale,
        mode=mode,
        smearing_factor=smearing_factor,
    )
    final_original, diagnostics = project_nonnegative_predictions(
        unprojected_original
    )
    return BaselinePredictions(
        final_original,
        native_pred_log,
        unprojected_original,
        diagnostics,
    )


@dataclass(frozen=True)
class BaselineEvaluation:
    result: dict[str, Any]
    predictions: BaselinePredictions


def evaluate_fitted_baseline(
    model: FittedBaseline,
    split: BaselineArraySplit,
    provenance: BaselineProvenance,
) -> BaselineEvaluation:
    """Evaluate any fitted baseline through the single shared result schema."""

    if split.name not in {"validation", "test"}:
        raise ValueError("baseline evaluation is allowed on validation or test only")
    predictions = model.predict(split.features)
    if predictions.pred_original.size != split.n_samples:
        raise ValueError("prediction sample count does not match evaluation split")
    contract = model.contract
    context = BaselineResultContext(
        model_name=contract.model_name,
        data_version=provenance.data_version,
        data_sha256=provenance.data_sha256,
        split_sha256=provenance.split_sha256,
        split_config_sha256=provenance.split_config_sha256,
        config_sha256=contract.config_sha256,
        preprocessing_sha256=provenance.preprocessing_sha256,
        split_seed=provenance.split_seed,
        train_seed=provenance.train_seed,
        evaluation_split=split.name,
        target_transform=contract.target_transform,
        target_scale=contract.target_scale,
        loss_objective=contract.loss_objective,
        inverse_mode=contract.inverse_mode,
        smearing_factor=contract.smearing_factor,
        best_iteration=contract.best_iteration,
        best_epoch=contract.best_epoch,
    )
    result = evaluate_baseline_predictions(
        context,
        y_true_original=split.target_original,
        y_pred_original=predictions.pred_original,
        y_true_log=split.target_log,
        y_pred_log=predictions.pred_log,
        prediction_support=predictions.support_diagnostics,
    )
    return BaselineEvaluation(result, predictions)


@dataclass(frozen=True)
class ValidationSelection:
    selected_model: FittedBaseline
    validation_evaluation: BaselineEvaluation
    candidate_evaluations: tuple[BaselineEvaluation, ...]
    selected_index: int
    split_sha256: str
    preprocessing_sha256: str

    def __post_init__(self) -> None:
        if not self.candidate_evaluations:
            raise ValueError("validation selection requires candidate evaluations")
        if not 0 <= self.selected_index < len(self.candidate_evaluations):
            raise ValueError("selected_index is outside candidate evaluations")
        selected = self.candidate_evaluations[self.selected_index]
        if selected.result != self.validation_evaluation.result:
            raise ValueError("selected validation evaluation/index mismatch")

    @property
    def candidate_results(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            evaluation.result for evaluation in self.candidate_evaluations
        )


def tune_on_validation(
    data: TuningData,
    candidates: Sequence[Any],
    fit_candidate: Callable[[TuningData, Any], FittedBaseline],
) -> ValidationSelection:
    """Fit candidates and select only by validation original-unit MAE."""

    if not candidates:
        raise ValueError("validation tuning requires at least one candidate")
    fitted: list[FittedBaseline] = []
    evaluations: list[BaselineEvaluation] = []
    for candidate in candidates:
        model = fit_candidate(data, candidate)
        evaluation = evaluate_fitted_baseline(
            model,
            data.validation,
            data.provenance,
        )
        fitted.append(model)
        evaluations.append(evaluation)
    best_index = min(
        range(len(evaluations)),
        key=lambda index: evaluations[index].result["headline_metrics"][
            "original_unit"
        ]["mae"],
    )
    return ValidationSelection(
        selected_model=fitted[best_index],
        validation_evaluation=evaluations[best_index],
        candidate_evaluations=tuple(evaluations),
        selected_index=best_index,
        split_sha256=data.provenance.split_sha256,
        preprocessing_sha256=data.provenance.preprocessing_sha256,
    )


def create_frozen_baseline_selection(
    path: str | Path,
    selections: Mapping[str, ValidationSelection],
) -> Path:
    """Persist validation decisions without opening the test subset.

    The resulting file is the only supported source of test authorization. It
    deliberately captures the already-selected inverse mode and Duan factor so
    neither can be changed after test access is granted.
    """

    if not selections:
        raise ValueError("at least one validation selection is required")
    rows: list[Mapping[str, Any]] = []
    frozen_models: dict[str, dict[str, Any]] = {}
    for mapping_name, selection in selections.items():
        result = selection.validation_evaluation.result
        validate_baseline_result_row(result)
        model_name = str(result["model_name"])
        if mapping_name != model_name:
            raise ValueError(
                f"selection key/model mismatch: {mapping_name!r} != {model_name!r}"
            )
        if result["evaluation_split"] != "validation":
            raise ValueError("only validation results can be frozen")
        if model_name in frozen_models:
            raise ValueError(f"duplicate frozen model_name: {model_name!r}")
        if result["config_sha256"] != selection.selected_model.contract.config_sha256:
            raise ValueError("selected model config does not match validation result")
        rows.append(result)
        frozen_models[model_name] = {
            "config_sha256": result["config_sha256"],
            "target_transform": result["target_transform"],
            "target_scale": result["target_scale"],
            "loss_objective": result["loss_objective"],
            "inverse_mode": result["inverse_mode"],
            "smearing_factor": result["smearing_factor"],
            "best_iteration": result["best_iteration"],
            "best_epoch": result["best_epoch"],
            "prediction_support_policy": result[
                "prediction_support_policy"
            ],
            "pre_projection_negative_count": result[
                "pre_projection_negative_count"
            ],
            "pre_projection_negative_fraction": result[
                "pre_projection_negative_fraction"
            ],
            "pre_projection_minimum": result["pre_projection_minimum"],
            "projection_applied_count": result["projection_applied_count"],
            "validation_metrics": {
                "headline_metrics": result["headline_metrics"],
                "secondary_metrics": result["secondary_metrics"],
            },
        }

    common_fields = (
        "data_version",
        "data_sha256",
        "split_sha256",
        "split_config_sha256",
        "preprocessing_sha256",
        "split_seed",
        "train_seed",
    )
    common: dict[str, Any] = {}
    for field_name in common_fields:
        values = {json.dumps(row[field_name], sort_keys=True) for row in rows}
        if len(values) != 1:
            raise ValueError(f"frozen selections have inconsistent {field_name}")
        common[field_name] = rows[0][field_name]
    payload = {
        "schema_version": FROZEN_BASELINE_SELECTION_SCHEMA_VERSION,
        **common,
        "models": frozen_models,
    }
    return write_metrics_json(path, payload, overwrite=False)


def authorize_test_evaluation(
    protocol: BaselineDataProtocol,
    selection: ValidationSelection,
    frozen_selection_json: str | Path,
) -> TestEvaluationAuthorization:
    """Validate a frozen file before issuing an in-memory test capability."""

    result = selection.validation_evaluation.result
    if result["evaluation_split"] != "validation":
        raise TestEvaluationBlockedError(
            "test authorization requires a validation-only selection"
        )
    if result["config_sha256"] != selection.selected_model.contract.config_sha256:
        raise TestEvaluationBlockedError(
            "selected model config does not match its validation result"
        )
    frozen_path = Path(frozen_selection_json)
    try:
        with frozen_path.open("r", encoding="utf-8") as stream:
            frozen = json.load(stream)
    except FileNotFoundError as error:
        raise TestEvaluationBlockedError(
            f"frozen baseline selection file is missing: {frozen_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise TestEvaluationBlockedError(
            f"cannot read frozen baseline selection file: {frozen_path}"
        ) from error
    if not isinstance(frozen, Mapping):
        raise TestEvaluationBlockedError("frozen baseline selection must be an object")
    if frozen.get("schema_version") != FROZEN_BASELINE_SELECTION_SCHEMA_VERSION:
        raise TestEvaluationBlockedError("frozen selection schema_version mismatch")

    expected_common = {
        "data_version": protocol.provenance.data_version,
        "data_sha256": protocol.provenance.data_sha256,
        "split_sha256": protocol.provenance.split_sha256,
        "split_config_sha256": protocol.provenance.split_config_sha256,
        "preprocessing_sha256": protocol.provenance.preprocessing_sha256,
        "split_seed": protocol.provenance.split_seed,
        "train_seed": protocol.provenance.train_seed,
    }
    for field_name, expected in expected_common.items():
        if frozen.get(field_name) != expected:
            raise TestEvaluationBlockedError(
                f"frozen selection {field_name} mismatch"
            )
    models = frozen.get("models")
    model_name = str(result["model_name"])
    if not isinstance(models, Mapping) or not isinstance(
        models.get(model_name), Mapping
    ):
        raise TestEvaluationBlockedError(
            f"frozen selection has no entry for {model_name!r}"
        )
    frozen_model = models[model_name]
    selected_fields = (
        "config_sha256",
        "target_transform",
        "target_scale",
        "loss_objective",
        "inverse_mode",
        "smearing_factor",
        "best_iteration",
        "best_epoch",
        "prediction_support_policy",
        "pre_projection_negative_count",
        "pre_projection_negative_fraction",
        "pre_projection_minimum",
        "projection_applied_count",
    )
    for field_name in selected_fields:
        if (
            field_name not in frozen_model
            or frozen_model[field_name] != result[field_name]
        ):
            raise TestEvaluationBlockedError(
                f"frozen selection {model_name}.{field_name} mismatch"
            )
    frozen_metrics = frozen_model.get("validation_metrics")
    expected_metrics = {
        "headline_metrics": result["headline_metrics"],
        "secondary_metrics": result["secondary_metrics"],
    }
    if frozen_metrics != expected_metrics:
        raise TestEvaluationBlockedError(
            f"frozen selection {model_name}.validation_metrics mismatch"
        )
    return TestEvaluationAuthorization(
        data_sha256=protocol.provenance.data_sha256,
        split_sha256=selection.split_sha256,
        split_config_sha256=protocol.provenance.split_config_sha256,
        preprocessing_sha256=selection.preprocessing_sha256,
        selected_config_sha256=result["config_sha256"],
        model_name=model_name,
        _authority=_FINAL_TEST_AUTHORITY,
    )


def evaluate_selection_on_test(
    protocol: BaselineDataProtocol,
    selection: ValidationSelection,
    frozen_selection_json: str | Path,
) -> BaselineEvaluation:
    """Open and evaluate test only through a frozen validation selection."""

    authorization = authorize_test_evaluation(
        protocol,
        selection,
        frozen_selection_json,
    )
    test_split = protocol.test_data(authorization)
    if authorization.selected_config_sha256 != (
        selection.selected_model.contract.config_sha256
    ):
        raise TestEvaluationBlockedError("test authorization config hash mismatch")
    return evaluate_fitted_baseline(
        selection.selected_model,
        test_split,
        protocol.provenance,
    )
