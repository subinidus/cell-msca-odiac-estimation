from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import cell_msca.baselines as baselines_module
import cell_msca.neural_baselines as neural_module
from cell_msca.baselines import (
    LIGHTGBM_MODEL_NAMES,
    BaselineDataProtocol,
    BaselinePredictions,
    LightGBMConfig,
    ModelContract,
    TestEvaluationBlockedError,
    TrainMeanConfig,
    _log_prediction_pair,
    _original_prediction_pair,
    _original_unit_mae_metric,
    authorize_test_evaluation,
    create_frozen_baseline_selection,
    evaluate_fitted_baseline,
    fit_lightgbm_baseline,
    fit_train_mean,
    select_log_inverse_on_validation,
    tune_on_validation,
)
from cell_msca.data import CellDataset, canonical_sha256
from cell_msca.evaluate import (
    BASELINE_RESULT_FIELDS,
    verify_baseline_result_from_prediction_csv,
    write_prediction_csv,
    write_prediction_support_diagnostic_csv,
)
from cell_msca.experiment import (
    BASELINE_SUITE_MODELS,
    load_baseline_experiment_config,
    run_validation_tuning_from_config,
)
from cell_msca.metrics import spearman_correlation
from cell_msca.neural_baselines import ConcatMLPConfig, fit_concat_mlp
from cell_msca.splits import (
    CellFixedSplitConfig,
    create_persistent_cell_fixed_split,
)
from tests.synthetic_npz import write_synthetic_months


class FakeConcatMLP:
    def __init__(self, mean_log: float, contract: ModelContract) -> None:
        self.mean_log = mean_log
        self.contract = contract

    def predict(self, features: np.ndarray) -> BaselinePredictions:
        pred_log = np.full(len(features), self.mean_log, dtype=np.float64)
        pred_original = np.expm1(pred_log)
        return BaselinePredictions(pred_original, pred_log)


def fit_fake_concat_mlp(data: object, config: ConcatMLPConfig) -> FakeConcatMLP:
    config_sha256 = canonical_sha256(
        config.to_dict(
            train_seed=data.provenance.train_seed,
            target_scale=data.provenance.target_scale,
        )
    )
    return FakeConcatMLP(
        float(np.mean(data.train.target_log)),
        ModelContract(
            model_name="concat_mlp",
            config_sha256=config_sha256,
            target_transform="log1p",
            target_scale=data.provenance.target_scale,
            loss_objective=f"{config.loss}_on_log1p_target",
            inverse_mode="median",
            smearing_factor=None,
            best_epoch=2,
        ),
    )


class FakeLightGBMRegressor:
    instances: list[FakeLightGBMRegressor] = []

    def __init__(self, **params: object) -> None:
        self.params = params
        self.fit_kwargs: dict[str, object] = {}
        self.mean_target = 0.0
        self.best_iteration_ = 3
        self.__class__.instances.append(self)

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        **kwargs: object,
    ) -> FakeLightGBMRegressor:
        self.fit_features = np.asarray(features, dtype=np.float64).copy()
        self.fit_target = np.asarray(target, dtype=np.float64).copy()
        self.fit_kwargs = kwargs
        self.mean_target = float(np.mean(self.fit_target))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.mean_target, dtype=np.float64)


class FakeModernLightGBMRegressor(FakeLightGBMRegressor):
    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        eval_X: np.ndarray,
        eval_y: np.ndarray,
        eval_names: list[str],
        eval_metric: object,
        callbacks: list[object] | None = None,
    ) -> FakeModernLightGBMRegressor:
        self.fit_features = np.asarray(features, dtype=np.float64).copy()
        self.fit_target = np.asarray(target, dtype=np.float64).copy()
        self.fit_kwargs = {
            "eval_X": eval_X,
            "eval_y": eval_y,
            "eval_names": eval_names,
            "eval_metric": eval_metric,
            "callbacks": callbacks,
        }
        self.mean_target = float(np.mean(self.fit_target))
        return self


class Phase3BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeLightGBMRegressor.instances.clear()

    def test_all_baselines_share_split_hash_and_prediction_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, data = self._build_protocol(Path(directory))
            models = [fit_train_mean(data)]
            models.extend(
                fit_lightgbm_baseline(
                    data,
                    LightGBMConfig(model_name=model_name, n_estimators=5),
                    estimator_factory=FakeLightGBMRegressor,
                )
                for model_name in LIGHTGBM_MODEL_NAMES
            )
            models.append(
                fit_fake_concat_mlp(
                    data,
                    ConcatMLPConfig(
                        hidden_sizes=(6,),
                        learning_rate=0.005,
                        batch_size=8,
                        max_epochs=4,
                        patience=2,
                    ),
                )
            )

            evaluations = [
                evaluate_fitted_baseline(model, data.validation, data.provenance)
                for model in models
            ]

            for evaluation in evaluations:
                self.assertEqual(
                    evaluation.predictions.pred_original.shape,
                    data.validation.target_original.shape,
                )
                self.assertEqual(
                    evaluation.predictions.pred_log.shape,
                    data.validation.target_log.shape,
                )
                self.assertEqual(set(evaluation.result), BASELINE_RESULT_FIELDS)
                self.assertEqual(
                    set(evaluation.result["headline_metrics"]),
                    {"original_unit"},
                )
                self.assertEqual(
                    set(evaluation.result["secondary_metrics"]),
                    {"log_space", "spearman"},
                )
            self.assertEqual(
                {evaluation.result["split_sha256"] for evaluation in evaluations},
                {data.provenance.split_sha256},
            )
            self.assertEqual(
                {evaluation.result["data_sha256"] for evaluation in evaluations},
                {data.provenance.data_sha256},
            )
            self.assertEqual(
                {
                    evaluation.result["split_config_sha256"]
                    for evaluation in evaluations
                },
                {data.provenance.split_config_sha256},
            )
            self.assertEqual(
                {
                    evaluation.result["preprocessing_sha256"]
                    for evaluation in evaluations
                },
                {data.provenance.preprocessing_sha256},
            )
            by_name = {
                evaluation.result["model_name"]: evaluation
                for evaluation in evaluations
            }
            for model_name in ("train_mean", "lightgbm_raw", "lightgbm_tweedie"):
                self.assertEqual(by_name[model_name].result["inverse_mode"], "none")
                self.assertEqual(
                    by_name[model_name].result["target_transform"],
                    "identity",
                )
            for model_name in ("lightgbm_log1p", "concat_mlp"):
                self.assertIn(
                    by_name[model_name].result["inverse_mode"],
                    {"median", "duan_smearing"},
                )
                self.assertEqual(
                    by_name[model_name].result["target_transform"],
                    "log1p",
                )
                self.assertEqual(by_name[model_name].result["target_scale"], 1.0)
            for model_name in LIGHTGBM_MODEL_NAMES:
                self.assertEqual(by_name[model_name].result["best_iteration"], 3)
                self.assertIsNone(by_name[model_name].result["best_epoch"])
            self.assertIsNotNone(by_name["concat_mlp"].result["best_epoch"])
            self.assertIsNone(by_name["train_mean"].result["best_iteration"])

    def test_tuning_cannot_access_test_until_selection_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, protocol, data = self._build_protocol(root)

            with self.assertRaisesRegex(TestEvaluationBlockedError, "unavailable"):
                data.subset("test")
            with self.assertRaisesRegex(TestEvaluationBlockedError, "authorization"):
                protocol.test_data()

            selection = tune_on_validation(
                data,
                [TrainMeanConfig()],
                fit_train_mean,
            )
            self.assertEqual(
                selection.validation_evaluation.result["evaluation_split"],
                "validation",
            )
            with self.assertRaisesRegex(TestEvaluationBlockedError, "missing"):
                authorize_test_evaluation(
                    protocol,
                    selection,
                    root / "missing.json",
                )

            frozen_path = create_frozen_baseline_selection(
                root / "FROZEN_BASELINE_SELECTION.json",
                {"train_mean": selection},
            )
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual(frozen["data_sha256"], data.provenance.data_sha256)
            self.assertEqual(frozen["split_sha256"], data.provenance.split_sha256)
            self.assertEqual(
                frozen["preprocessing_sha256"],
                data.provenance.preprocessing_sha256,
            )
            self.assertIn("validation_metrics", frozen["models"]["train_mean"])
            authorization = authorize_test_evaluation(
                protocol,
                selection,
                frozen_path,
            )
            self.assertEqual(authorization.model_name, "train_mean")
            self.assertEqual(
                authorization.selected_config_sha256,
                selection.selected_model.contract.config_sha256,
            )
            # Deliberately do not call protocol.test_data in Phase 3 validation work.

    def test_frozen_selection_rejects_hash_mismatch_before_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, protocol, data = self._build_protocol(root)
            selection = tune_on_validation(data, [TrainMeanConfig()], fit_train_mean)
            frozen_path = create_frozen_baseline_selection(
                root / "FROZEN_BASELINE_SELECTION.json",
                {"train_mean": selection},
            )
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            frozen["data_sha256"] = "0" * 64
            changed_path = root / "changed_hash.json"
            changed_path.write_text(
                json.dumps(frozen),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TestEvaluationBlockedError, "data_sha256"):
                authorize_test_evaluation(protocol, selection, changed_path)

            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            del frozen["models"]["train_mean"]["best_iteration"]
            missing_field_path = root / "missing_model_field.json"
            missing_field_path.write_text(json.dumps(frozen), encoding="utf-8")
            with self.assertRaisesRegex(TestEvaluationBlockedError, "best_iteration"):
                authorize_test_evaluation(protocol, selection, missing_field_path)

    def test_lightgbm_early_stopping_receives_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, data = self._build_protocol(Path(directory))
            callback_rounds: list[int] = []

            def callback_factory(rounds: int) -> object:
                callback_rounds.append(rounds)
                return object()

            fit_lightgbm_baseline(
                data,
                LightGBMConfig(
                    model_name="lightgbm_raw",
                    n_estimators=10,
                    early_stopping_rounds=4,
                ),
                estimator_factory=FakeLightGBMRegressor,
                early_stopping_factory=callback_factory,
            )
            estimator = FakeLightGBMRegressor.instances[-1]
            eval_set = estimator.fit_kwargs["eval_set"]

            self.assertEqual(len(eval_set), 1)
            np.testing.assert_array_equal(eval_set[0][0], data.validation.features)
            np.testing.assert_array_equal(
                eval_set[0][1],
                data.validation.target_original,
            )
            self.assertEqual(estimator.fit_kwargs["eval_names"], ["validation"])
            self.assertEqual(callback_rounds, [4])
            metric_name, metric_value, higher_is_better = estimator.fit_kwargs[
                "eval_metric"
            ](
                data.validation.target_original,
                data.validation.target_original,
            )
            self.assertEqual(metric_name, "original_unit_mae")
            self.assertEqual(metric_value, 0.0)
            self.assertFalse(higher_is_better)

    def test_modern_lightgbm_api_receives_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, data = self._build_protocol(Path(directory))
            model = fit_lightgbm_baseline(
                data,
                LightGBMConfig(model_name="lightgbm_raw", n_estimators=5),
                estimator_factory=FakeModernLightGBMRegressor,
                early_stopping_factory=lambda rounds: ("early_stopping", rounds),
            )
            estimator = model.estimator

            self.assertNotIn("eval_set", estimator.fit_kwargs)
            np.testing.assert_array_equal(
                estimator.fit_kwargs["eval_X"],
                data.validation.features,
            )
            np.testing.assert_array_equal(
                estimator.fit_kwargs["eval_y"],
                data.validation.target_original,
            )

    def test_log_lightgbm_checkpoint_metric_uses_median_inverse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, data = self._build_protocol(Path(directory))
            model = fit_lightgbm_baseline(
                data,
                LightGBMConfig(model_name="lightgbm_log1p", n_estimators=5),
                estimator_factory=FakeModernLightGBMRegressor,
            )
            metric = model.estimator.fit_kwargs["eval_metric"]
            true_log = np.log1p(np.array([1.0, 3.0]))
            pred_log = np.log1p(np.array([0.0, 2.0]))
            name, value, higher_is_better = metric(true_log, pred_log)

            self.assertEqual(name, "original_unit_mae")
            self.assertAlmostEqual(value, 1.0)
            self.assertFalse(higher_is_better)

    def test_duan_factor_uses_train_residual_arrays_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, data = self._build_protocol(Path(directory))
            received: list[tuple[np.ndarray, np.ndarray]] = []

            def capture_duan(pred_log: np.ndarray, true_log: np.ndarray) -> float:
                received.append((np.asarray(pred_log).copy(), np.asarray(true_log).copy()))
                return 1.0

            with mock.patch.object(
                baselines_module,
                "duan_smearing_factor",
                capture_duan,
            ):
                fit_lightgbm_baseline(
                    data,
                    LightGBMConfig(model_name="lightgbm_log1p", n_estimators=5),
                    estimator_factory=FakeLightGBMRegressor,
                )

            self.assertEqual(len(received), 1)
            expected_prediction = np.full(
                data.train.n_samples,
                np.mean(data.train.target_log),
            )
            np.testing.assert_array_equal(received[0][0], expected_prediction)
            np.testing.assert_array_equal(received[0][1], data.train.target_log)
            self.assertNotEqual(received[0][0].shape, data.validation.target_log.shape)

    def test_inverse_mode_selection_uses_validation_original_mae(self) -> None:
        selection = select_log_inverse_on_validation(
            train_true_log=np.array([np.log(2.0), np.log(2.0)]),
            train_pred_log=np.array([0.0, 0.0]),
            validation_true_original=np.array([1.0]),
            validation_pred_log=np.array([0.0]),
            target_scale=1.0,
        )

        self.assertEqual(selection.inverse_mode, "duan_smearing")
        self.assertAlmostEqual(float(selection.smearing_factor), 2.0)
        self.assertAlmostEqual(selection.validation_original_mae, 0.0)

    def test_identity_and_log_prediction_paths_apply_the_same_support_policy(self) -> None:
        identity = _original_prediction_pair(
            np.asarray([-2.0, 3.0]),
            target_scale=1.0,
            model_name="lightgbm_raw",
        )
        np.testing.assert_array_equal(identity.pred_original, [0.0, 3.0])
        np.testing.assert_array_equal(identity.unprojected_original, [-2.0, 3.0])
        np.testing.assert_allclose(identity.pred_log, np.log1p([0.0, 3.0]))

        native_log = np.asarray([-2.0, np.log(4.0)])
        log_path = _log_prediction_pair(
            native_log,
            scale=1.0,
            mode="median",
            smearing_factor=None,
        )
        np.testing.assert_array_equal(log_path.pred_original, [0.0, 3.0])
        np.testing.assert_array_equal(log_path.pred_log, native_log)

    def test_early_stopping_and_inverse_selection_use_projected_mae(self) -> None:
        metric = _original_unit_mae_metric(target_space="original", target_scale=1.0)
        name, value, higher_is_better = metric(
            np.asarray([0.0, 2.0]),
            np.asarray([-4.0, 2.0]),
        )
        self.assertEqual(name, "original_unit_mae")
        self.assertEqual(value, 0.0)
        self.assertFalse(higher_is_better)

        selection = select_log_inverse_on_validation(
            train_true_log=np.asarray([0.0, 0.0]),
            train_pred_log=np.asarray([0.0, 0.0]),
            validation_true_original=np.asarray([0.0]),
            validation_pred_log=np.asarray([-2.0]),
            target_scale=1.0,
        )
        self.assertEqual(selection.inverse_mode, "median")
        self.assertEqual(selection.validation_original_mae, 0.0)

    def test_support_diagnostic_csv_records_changed_rows_and_true_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.csv"
            true = np.asarray([4.0, 5.0, 6.0])
            true_before = true.copy()
            write_prediction_support_diagnostic_csv(
                path,
                y_true_original=true,
                unprojected_prediction=[-1.0, 2.0, -3.0],
                final_prediction=[0.0, 2.0, 0.0],
                cell_ids=["a", "b", "c"],
                month_ids=["2022-01", "2022-01", "2022-02"],
            )
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertIn("a,2022-01,4.0,-1.0,0.0", rows[1])
            self.assertIn("c,2022-02,6.0,-3.0,0.0", rows[2])
            np.testing.assert_array_equal(true, true_before)

    def test_prediction_csv_reproduces_shared_baseline_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, data = self._build_protocol(root)
            evaluation = evaluate_fitted_baseline(
                fit_train_mean(data),
                data.validation,
                data.provenance,
            )
            prediction_path = root / "predictions.csv"
            write_prediction_csv(
                prediction_path,
                y_true_original=data.validation.target_original,
                y_pred_original=evaluation.predictions.pred_original,
                y_true_log=data.validation.target_log,
                y_pred_log=evaluation.predictions.pred_log,
                cell_ids=data.validation.cell_ids,
            )

            recalculated = verify_baseline_result_from_prediction_csv(
                prediction_path,
                evaluation.result,
            )

            self.assertEqual(
                recalculated["headline_metrics"],
                evaluation.result["headline_metrics"],
            )
            self.assertEqual(
                recalculated["secondary_metrics"],
                evaluation.result["secondary_metrics"],
            )

    def test_spearman_is_secondary_and_uses_finite_constant_convention(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
            -1.0,
        )
        self.assertEqual(
            spearman_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]),
            0.0,
        )

    def test_concat_mlp_config_and_missing_torch_dependency_contract(self) -> None:
        config = ConcatMLPConfig(
            loss="log_huber",
            huber_delta=0.75,
            dropout=0.2,
            weight_decay=0.003,
        )
        values = config.to_dict(train_seed=9, target_scale=1.0)
        self.assertEqual(values["optimizer"], "AdamW")
        self.assertEqual(
            values["weight_decay_scope"],
            "weights_only_no_bias_or_normalization",
        )
        self.assertEqual(values["loss"], "log_huber")
        self.assertEqual(values["huber_delta"], 0.75)
        self.assertEqual(values["dropout"], 0.2)
        with self.assertRaisesRegex(ValueError, "log_l1.*log_huber"):
            ConcatMLPConfig(loss="mse")  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as directory:
            _, _, _, data = self._build_protocol(Path(directory))
            with mock.patch.object(
                neural_module.importlib,
                "import_module",
                side_effect=ImportError("torch unavailable"),
            ):
                with self.assertRaisesRegex(ImportError, "optional 'torch'"):
                    fit_concat_mlp(data, config)

    def test_adamw_weight_decay_excludes_bias_parameters(self) -> None:
        class Parameter:
            requires_grad = True

        weight = Parameter()
        bias = Parameter()

        class Model:
            @staticmethod
            def named_parameters() -> list[tuple[str, Parameter]]:
                return [("0.weight", weight), ("0.bias", bias)]

        groups = neural_module._adamw_parameter_groups(  # noqa: SLF001
            Model(),
            weight_decay=0.25,
        )
        self.assertEqual(groups[0]["params"], [weight])
        self.assertEqual(groups[0]["weight_decay"], 0.25)
        self.assertEqual(groups[1]["params"], [bias])
        self.assertEqual(groups[1]["weight_decay"], 0.0)

    def test_validation_runner_saves_and_verifies_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_protocol(root)
            config_path = root / "baseline.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": "cell_msca.baseline_experiment.v1",
                        "stage": "validation_tuning",
                        "prediction_support_policy": "nonnegative_max_zero_v1",
                        "project_root": ".",
                        "data": {
                            "npz_glob": "data/*.npz",
                            "target_scale": 1.0,
                        },
                        "split_csv": "cell_fixed_seed42.csv",
                        "split_metadata_json": (
                            "cell_fixed_seed42.metadata.v2.json"
                        ),
                        "output_dir": "validation_outputs",
                        "split": {
                            "split_seed": 42,
                            "validation_ratio": 0.15,
                            "test_ratio": 0.15,
                        },
                        "train_seed": 3407,
                        "models": {
                            "train_mean": [{}],
                            "lightgbm_raw": [{"n_estimators": 5}],
                            "lightgbm_log1p": [{"n_estimators": 5}],
                            "lightgbm_tweedie": [{"n_estimators": 5}],
                            "concat_mlp": [
                                {
                                    "hidden_sizes": [6],
                                    "dropout": 0.1,
                                    "loss": "log_l1",
                                    "huber_delta": 1.0,
                                    "max_epochs": 2,
                                    "patience": 1,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = run_validation_tuning_from_config(
                config_path,
                estimator_factory=FakeLightGBMRegressor,
                concat_fit_candidate=fit_fake_concat_mlp,
            )
            self.assertIsNotNone(result.artifacts)
            artifacts = result.artifacts
            assert artifacts is not None
            self.assertTrue(artifacts.candidate_results_json.is_file())
            self.assertTrue(artifacts.selected_results_json.is_file())
            self.assertTrue(artifacts.manifest_json.is_file())
            self.assertEqual(len(artifacts.prediction_csvs), 5)
            self.assertEqual(len(artifacts.negative_prediction_csvs), 5)
            candidate_payload = json.loads(
                artifacts.candidate_results_json.read_text(encoding="utf-8")
            )
            self.assertEqual(len(candidate_payload["candidates"]), 5)
            for entry in candidate_payload["candidates"]:
                self.assertEqual(
                    entry["prediction_metric_verification"],
                    "passed_including_spearman",
                )
                verify_baseline_result_from_prediction_csv(
                    artifacts.output_dir / entry["prediction_csv"],
                    entry["result"],
                )
            manifest = json.loads(
                artifacts.manifest_json.read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["test_subset_materialized"])
            frozen_path = create_frozen_baseline_selection(
                root / "FROZEN_BASELINE_SELECTION.json",
                result.selections,
            )
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertEqual(set(frozen["models"]), set(BASELINE_SUITE_MODELS))

            before = artifacts.candidate_results_json.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                run_validation_tuning_from_config(
                    config_path,
                    estimator_factory=FakeLightGBMRegressor,
                    concat_fit_candidate=fit_fake_concat_mlp,
                )
            self.assertEqual(before, artifacts.candidate_results_json.read_bytes())

    def test_executable_example_config_has_all_five_models(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_baseline_experiment_config(
            project_root / "configs" / "baseline_example.json"
        )

        self.assertEqual(set(config.models), set(BASELINE_SUITE_MODELS))
        self.assertEqual(config.split_config.split_seed, 42)
        self.assertEqual(
            config.split_metadata_json.name,
            "cell_fixed_seed42.metadata.v2.json",
        )
        self.assertEqual(config.npz_glob, "data/v1_legacy/*.npz")
        self.assertEqual(
            config.output_dir.name,
            "example_validation",
        )

    @staticmethod
    def _build_protocol(
        root: Path,
    ) -> tuple[CellDataset, object, BaselineDataProtocol, object]:
        dataset = CellDataset(
            write_synthetic_months(
                root / "data",
                n_months=3,
                height=4,
                width=4,
            )
        )
        config = CellFixedSplitConfig()
        split_path = root / "cell_fixed_seed42.csv"
        metadata_path = root / "cell_fixed_seed42.metadata.v2.json"
        manifest = create_persistent_cell_fixed_split(
            dataset,
            split_path,
            metadata_path,
            config=config,
        )
        protocol = BaselineDataProtocol.from_dataset(
            dataset,
            split_path,
            metadata_path,
            split_config=config,
            train_seed=3407,
        )
        return dataset, manifest, protocol, protocol.tuning_data()


if __name__ == "__main__":
    unittest.main()
