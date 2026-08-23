"""Synthetic architecture and runtime regression tests for Phase 4.1."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - exercised only without dependency
    torch = None  # type: ignore[assignment]

if torch is not None:
    from cell_msca.baselines import (
        BaselineArraySplit,
        BaselineProvenance,
        TuningData,
    )
    from cell_msca.data import STREAM_A_FEATURES, STREAM_B_FEATURES
    from cell_msca.model import (
        CELL_MSCA_VARIANTS,
        POLLUTION_ENVIRONMENT_FEATURES,
        SOCIO_INFRASTRUCTURE_FEATURES,
        CellMSCA,
        CellMSCAConfig,
        count_trainable_parameters,
        variant_parameter_counts,
    )
    from cell_msca.neural_baselines import (
        _adamw_parameter_groups,
        _resolve_torch_device,
        _set_deterministic_seed,
    )
    from cell_msca.train import (
        CELL_MSCA_CHECKPOINT_SCHEMA_VERSION,
        CellMSCACheckpointProvenance,
        CellMSCATrainingConfig,
        fit_cell_msca,
        load_selected_checkpoint,
    )


@unittest.skipIf(torch is None, "Phase 4.1 tests require PyTorch")
class CellMSCAModelTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        _set_deterministic_seed(torch, 3407)
        self.base_config = CellMSCAConfig(
            variant="forward",
            d_model=8,
            num_heads=2,
            encoder_layers=1,
            ffn_multiplier=2,
            head_hidden=8,
            dropout=0.0,
            attention_dropout=0.0,
        )
        self.pollution = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            dtype=torch.float32,
        )
        self.infrastructure = torch.tensor(
            [
                [0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2, 1.3],
                [1.4, 1.5, 1.6, 1.7],
            ],
            dtype=torch.float32,
        )

    def test_feature_order_stream_membership_and_token_specificity(self) -> None:
        self.assertEqual(POLLUTION_ENVIRONMENT_FEATURES, STREAM_A_FEATURES)
        self.assertEqual(SOCIO_INFRASTRUCTURE_FEATURES, STREAM_B_FEATURES)
        self.assertEqual(
            POLLUTION_ENVIRONMENT_FEATURES,
            ("no2_mean", "so2_mean", "co_mean"),
        )
        self.assertEqual(
            SOCIO_INFRASTRUCTURE_FEATURES,
            (
                "nightlight_mean",
                "urban_fraction",
                "power_plant_count",
                "fossil_capacity_mw",
            ),
        )
        model = CellMSCA(self.base_config)
        base = torch.zeros(1, 3)
        changed = base.clone()
        changed[0, 1] = 2.0
        base_tokens = model.pollution_tokenizer(base)
        changed_tokens = model.pollution_tokenizer(changed)
        torch.testing.assert_close(base_tokens[:, 0], changed_tokens[:, 0])
        torch.testing.assert_close(base_tokens[:, 2], changed_tokens[:, 2])
        self.assertFalse(torch.equal(base_tokens[:, 1], changed_tokens[:, 1]))

    def test_exact_output_token_and_attention_shapes_for_all_variants(self) -> None:
        for variant in CELL_MSCA_VARIANTS:
            with self.subTest(variant=variant):
                model = CellMSCA(replace(self.base_config, variant=variant)).eval()
                result = model.forward_with_details(
                    self.pollution,
                    self.infrastructure,
                )
                self.assertEqual(result.prediction_log.shape, (4,))
                self.assertEqual(
                    result.pollution_environment_tokens.shape,
                    (4, 3, 8),
                )
                self.assertEqual(
                    result.socio_infrastructure_tokens.shape,
                    (4, 4, 8),
                )
                if variant in {"forward", "bidirectional"}:
                    assert result.forward_attention is not None
                    self.assertEqual(result.forward_attention.shape, (4, 2, 4, 3))
                    torch.testing.assert_close(
                        result.forward_attention.sum(dim=-1),
                        torch.ones(4, 2, 4),
                    )
                else:
                    self.assertIsNone(result.forward_attention)
                if variant in {"reverse", "bidirectional"}:
                    assert result.reverse_attention is not None
                    self.assertEqual(result.reverse_attention.shape, (4, 2, 3, 4))
                    torch.testing.assert_close(
                        result.reverse_attention.sum(dim=-1),
                        torch.ones(4, 2, 3),
                    )
                else:
                    self.assertIsNone(result.reverse_attention)

    def test_finite_forward_backward_and_gradients_reach_both_streams(self) -> None:
        pollution = self.pollution.clone().requires_grad_(True)
        infrastructure = self.infrastructure.clone().requires_grad_(True)
        model = CellMSCA(self.base_config)
        prediction = model(pollution, infrastructure)
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        assert pollution.grad is not None
        assert infrastructure.grad is not None
        self.assertGreater(float(pollution.grad.abs().sum()), 0.0)
        self.assertGreater(float(infrastructure.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.pollution_tokenizer.weight.grad)
        self.assertIsNotNone(model.infrastructure_tokenizer.weight.grad)
        assert model.forward_cross is not None
        self.assertIsNotNone(model.forward_cross.attention.in_proj_weight.grad)

    def test_batch_permutation_equivariance(self) -> None:
        model = CellMSCA(self.base_config).eval()
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            original = model(self.pollution, self.infrastructure)
            permuted = model(
                self.pollution[permutation],
                self.infrastructure[permutation],
            )
        torch.testing.assert_close(permuted, original[permutation])

    def test_attention_never_mixes_samples(self) -> None:
        model = CellMSCA(
            replace(self.base_config, variant="bidirectional")
        ).eval()
        changed_pollution = self.pollution[:2].clone()
        changed_infrastructure = self.infrastructure[:2].clone()
        changed_pollution[1] = 1_000_000.0
        changed_infrastructure[1] = -1_000_000.0
        with torch.no_grad():
            alone = model(self.pollution[:1], self.infrastructure[:1])
            batched = model(changed_pollution, changed_infrastructure)
        torch.testing.assert_close(alone[0], batched[0])

    def test_forward_and_reverse_are_structurally_distinct(self) -> None:
        forward = CellMSCA(self.base_config)
        reverse = CellMSCA(replace(self.base_config, variant="reverse"))
        self.assertIsNotNone(forward.forward_cross)
        self.assertIsNone(forward.reverse_cross)
        self.assertIsNone(reverse.forward_cross)
        self.assertIsNotNone(reverse.reverse_cross)
        forward_details = forward.forward_with_details(
            self.pollution,
            self.infrastructure,
        )
        reverse_details = reverse.forward_with_details(
            self.pollution,
            self.infrastructure,
        )
        self.assertEqual(forward_details.forward_attention.shape[-2:], (4, 3))
        self.assertEqual(reverse_details.reverse_attention.shape[-2:], (3, 4))

    def test_bidirectional_and_no_attention_contracts(self) -> None:
        bidirectional = CellMSCA(
            replace(self.base_config, variant="bidirectional")
        )
        no_attention = CellMSCA(
            replace(self.base_config, variant="token_no_attention")
        )
        bidirectional_result = bidirectional.forward_with_details(
            self.pollution,
            self.infrastructure,
        )
        no_attention_result = no_attention.forward_with_details(
            self.pollution,
            self.infrastructure,
        )
        self.assertEqual(bidirectional_result.prediction_log.shape, (4,))
        self.assertIsNotNone(bidirectional_result.forward_attention)
        self.assertIsNotNone(bidirectional_result.reverse_attention)
        self.assertEqual(no_attention_result.prediction_log.shape, (4,))
        self.assertIsNone(no_attention_result.forward_attention)
        self.assertIsNone(no_attention_result.reverse_attention)

    def test_invalid_feature_shapes_fail_clearly(self) -> None:
        model = CellMSCA(self.base_config)
        with self.assertRaisesRegex(ValueError, r"pollution/environment.*\[batch, 3\]"):
            model(torch.zeros(4, 4), self.infrastructure)
        with self.assertRaisesRegex(ValueError, r"socio-infrastructure.*\[batch, 4\]"):
            model(self.pollution, torch.zeros(4, 3))
        with self.assertRaisesRegex(ValueError, "same batch size"):
            model(self.pollution[:2], self.infrastructure)
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = self.pollution.clone()
            bad[0, 0] = float("nan")
            model(bad, self.infrastructure)

    def test_deterministic_cpu_initialization_and_prediction(self) -> None:
        _set_deterministic_seed(torch, 2026)
        first = CellMSCA(self.base_config).eval()
        first_prediction = first(self.pollution, self.infrastructure).detach()
        first_state = {name: value.detach().clone() for name, value in first.state_dict().items()}
        _set_deterministic_seed(torch, 2026)
        second = CellMSCA(self.base_config).eval()
        second_prediction = second(self.pollution, self.infrastructure).detach()
        torch.testing.assert_close(first_prediction, second_prediction, rtol=0, atol=0)
        for name, value in second.state_dict().items():
            torch.testing.assert_close(first_state[name], value, rtol=0, atol=0)

    def test_bias_and_normalization_are_excluded_from_weight_decay(self) -> None:
        model = CellMSCA(
            replace(self.base_config, variant="bidirectional")
        )
        groups = _adamw_parameter_groups(model, weight_decay=0.125)
        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertEqual(groups[0]["weight_decay"], 0.125)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        for name, parameter in model.named_parameters():
            if name == "bias" or name.endswith(".bias") or parameter.ndim == 1:
                self.assertIn(id(parameter), no_decay_ids, name)
                self.assertNotIn(id(parameter), decay_ids, name)
            else:
                self.assertIn(id(parameter), decay_ids, name)

    def test_variant_parameter_counts_are_reported_and_structurally_expected(self) -> None:
        counts = variant_parameter_counts(self.base_config)
        self.assertEqual(set(counts), set(CELL_MSCA_VARIANTS))
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertEqual(counts["forward"], counts["reverse"])
        self.assertGreater(counts["bidirectional"], counts["forward"])
        self.assertEqual(
            counts["forward"],
            count_trainable_parameters(CellMSCA(self.base_config)),
        )

    def test_device_resolution_is_explicit(self) -> None:
        self.assertEqual(str(_resolve_torch_device(torch, "cpu")), "cpu")
        expected_auto = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(str(_resolve_torch_device(torch, "auto")), expected_auto)
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                _resolve_torch_device(torch, "cuda")

    def test_selected_checkpoint_round_trip_and_provenance(self) -> None:
        data = self._tiny_tuning_data()
        architecture = replace(self.base_config, variant="bidirectional")
        config = CellMSCATrainingConfig(
            architecture=architecture,
            loss="log_huber",
            huber_delta=0.75,
            learning_rate=1e-3,
            batch_size=4,
            max_epochs=2,
            patience=2,
            weight_decay=0.01,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.pt"
            fitted = fit_cell_msca(
                data,
                config,
                checkpoint_path=path,
                repository_root=Path(__file__).resolve().parents[1],
            )
            self.assertTrue(path.is_file())
            self.assertGreaterEqual(fitted.contract.best_epoch or 0, 1)
            self.assertEqual(fitted.parameter_count, count_trainable_parameters(fitted.model))
            self.assertIsNotNone(fitted.checkpoint_provenance)
            assert fitted.checkpoint_provenance is not None
            self.assertEqual(fitted.checkpoint_provenance.split_seed, 42)
            self.assertEqual(fitted.checkpoint_provenance.train_seed, 3407)
            self.assertEqual(fitted.checkpoint_provenance.pytorch_version, torch.__version__)
            expected_hashes = {
                "data_sha256": data.provenance.data_sha256,
                "split_sha256": data.provenance.split_sha256,
                "preprocessing_sha256": data.provenance.preprocessing_sha256,
                "configuration_sha256": fitted.contract.config_sha256,
            }
            loaded = load_selected_checkpoint(
                path,
                device="cpu",
                expected_hashes=expected_hashes,
            )
            features = data.validation.features
            pollution = torch.as_tensor(features[:, :3], dtype=torch.float32)
            infrastructure = torch.as_tensor(features[:, 3:], dtype=torch.float32)
            fitted.model.eval()
            with torch.no_grad():
                before = fitted.model(pollution, infrastructure).cpu()
                after = loaded.model(pollution, infrastructure).cpu()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertEqual(loaded.parameter_count, fitted.parameter_count)
            self.assertEqual(
                CELL_MSCA_CHECKPOINT_SCHEMA_VERSION,
                "cell_msca.selected_checkpoint.v1",
            )
            with self.assertRaisesRegex(ValueError, "data_sha256 mismatch"):
                load_selected_checkpoint(
                    path,
                    expected_hashes={"data_sha256": "f" * 64},
                )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                from cell_msca.train import save_selected_checkpoint

                save_selected_checkpoint(
                    path,
                    model=fitted.model,
                    config=config,
                    contract=fitted.contract,
                    provenance=fitted.checkpoint_provenance,
                    best_epoch=fitted.contract.best_epoch or 1,
                    validation_original_mae=0.0,
                )

    @staticmethod
    def _tiny_tuning_data() -> TuningData:
        rng = np.random.default_rng(17)
        train_features = rng.normal(size=(12, 7))
        validation_features = rng.normal(size=(6, 7))
        train_target = np.square(train_features[:, 0]) + np.abs(train_features[:, 4])
        validation_target = np.square(validation_features[:, 0]) + np.abs(
            validation_features[:, 4]
        )
        return TuningData(
            train=BaselineArraySplit(
                "train",
                train_features,
                train_target,
                np.log1p(train_target),
                np.asarray([f"train-{index}" for index in range(12)]),
            ),
            validation=BaselineArraySplit(
                "validation",
                validation_features,
                validation_target,
                np.log1p(validation_target),
                np.asarray([f"validation-{index}" for index in range(6)]),
            ),
            provenance=BaselineProvenance(
                data_version="synthetic_phase4",
                data_sha256="1" * 64,
                split_sha256="2" * 64,
                split_config_sha256="3" * 64,
                preprocessing_sha256="4" * 64,
                split_seed=42,
                train_seed=3407,
                target_scale=1.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
