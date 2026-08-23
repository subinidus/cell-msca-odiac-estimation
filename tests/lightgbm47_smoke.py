"""Manual LightGBM 4.7 smoke; prints JSON and never opens a test split."""

from __future__ import annotations

import inspect
import json

import lightgbm as lgb
import numpy as np

from cell_msca.baselines import (
    LIGHTGBM_MODEL_NAMES,
    BaselineArraySplit,
    BaselineProvenance,
    LightGBMConfig,
    TuningData,
    fit_lightgbm_baseline,
)


def main() -> int:
    if lgb.__version__ != "4.7.0":
        raise RuntimeError(f"expected LightGBM 4.7.0; got {lgb.__version__}")
    fit_parameters = inspect.signature(lgb.LGBMRegressor.fit).parameters
    if "eval_X" not in fit_parameters or "eval_y" not in fit_parameters:
        raise RuntimeError("LightGBM 4.7 fit lacks eval_X/eval_y")

    rng = np.random.default_rng(20260823)
    train_features = rng.normal(size=(96, 7))
    validation_features = rng.normal(size=(32, 7))

    def target(features: np.ndarray) -> np.ndarray:
        signal = 1.7 + 0.25 * features[:, 0] - 0.15 * features[:, 3]
        return np.exp(np.clip(signal, 0.2, 3.0)) - 1.0

    train_original = target(train_features)
    validation_original = target(validation_features)
    data = TuningData(
        train=BaselineArraySplit(
            name="train",
            features=train_features,
            target_original=train_original,
            target_log=np.log1p(train_original),
            cell_ids=np.asarray([f"train_{index}" for index in range(96)]),
        ),
        validation=BaselineArraySplit(
            name="validation",
            features=validation_features,
            target_original=validation_original,
            target_log=np.log1p(validation_original),
            cell_ids=np.asarray([f"validation_{index}" for index in range(32)]),
        ),
        provenance=BaselineProvenance(
            data_version="synthetic_lightgbm47_smoke",
            data_sha256="1" * 64,
            split_sha256="2" * 64,
            split_config_sha256="3" * 64,
            preprocessing_sha256="4" * 64,
            split_seed=42,
            train_seed=3407,
            target_scale=1.0,
        ),
    )
    records = []
    for model_name in LIGHTGBM_MODEL_NAMES:
        fitted = fit_lightgbm_baseline(
            data,
            LightGBMConfig(
                model_name=model_name,
                n_estimators=40,
                learning_rate=0.08,
                num_leaves=15,
                min_child_samples=5,
                early_stopping_rounds=6,
            ),
        )
        prediction = fitted.predict(data.validation.features)
        eval_names = sorted(getattr(fitted.estimator, "evals_result_", {}))
        if eval_names != ["validation"]:
            raise RuntimeError(f"unexpected evaluation data names: {eval_names}")
        if prediction.pred_original.shape != (32,):
            raise RuntimeError("unexpected original prediction shape")
        if prediction.pred_log.shape != (32,):
            raise RuntimeError("unexpected log prediction shape")
        if not fitted.contract.best_iteration or fitted.contract.best_iteration <= 0:
            raise RuntimeError("missing positive best_iteration")
        records.append(
            {
                "model_name": model_name,
                "prediction_shape": list(prediction.pred_original.shape),
                "best_iteration": fitted.contract.best_iteration,
                "validation_eval_names": eval_names,
                "fit_validation_api": "eval_X/eval_y single arrays",
            }
        )
    print(
        json.dumps(
            {
                "lightgbm_version": lgb.__version__,
                "test_split_created_or_evaluated": False,
                "models": records,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
