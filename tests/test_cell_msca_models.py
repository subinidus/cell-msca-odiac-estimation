"""Synthetic architecture and runtime regression tests for Phase 4."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

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
        DirectionalCrossAttention,
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
        GIT_DIRTY_STATE_POLICY,
        CellMSCATrainingConfig,
        checkpoint_manifest_json,
        discover_git_state,
        fit_cell_msca,
        load_selected_checkpoint,
        save_selected_checkpoint,
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

    def test_numerical_tokenizer_is_exact_feature_specific_affine_map(self) -> None:
        model = CellMSCA(self.base_config)
        tokenizer = model.pollution_tokenizer
        values = self.pollution[:2]
        expected = (
            values.unsqueeze(-1) * tokenizer.weight.unsqueeze(0)
            + tokenizer.bias.unsqueeze(0)
        )
        actual = tokenizer(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        parameter_names = dict(tokenizer.named_parameters())
        self.assertEqual(set(parameter_names), {"weight", "bias"})
        self.assertNotIn("feature_embedding", parameter_names)
        self.assertNotIn("group_embedding", parameter_names)
        self.assertEqual(
            model.config.to_dict()["tokenizer"],
            "feature_specific_affine_z_f_equals_x_f_w_f_plus_b_f",
        )

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
                self.assertEqual(result.pooled_representation.shape, (4, 16))
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

    def test_all_variants_pass_two_d_model_to_identical_regression_heads(self) -> None:
        expected_state_shapes: dict[str, tuple[int, ...]] | None = None
        for variant in CELL_MSCA_VARIANTS:
            with self.subTest(variant=variant):
                model = CellMSCA(replace(self.base_config, variant=variant)).eval()
                observed_inputs: list[tuple[int, ...]] = []

                def record_head_input(_module: object, inputs: tuple[torch.Tensor, ...]) -> None:
                    observed_inputs.append(tuple(inputs[0].shape))

                handle = model.regression_head.register_forward_pre_hook(
                    record_head_input
                )
                try:
                    result = model.forward_with_details(
                        self.pollution,
                        self.infrastructure,
                    )
                finally:
                    handle.remove()
                self.assertEqual(observed_inputs, [(4, 2 * self.base_config.d_model)])
                self.assertEqual(
                    tuple(model.regression_head[0].normalized_shape),
                    (2 * self.base_config.d_model,),
                )
                self.assertEqual(
                    model.regression_head[1].in_features,
                    2 * self.base_config.d_model,
                )
                pooled_pollution = result.pollution_environment_tokens.mean(dim=1)
                pooled_infrastructure = (
                    result.socio_infrastructure_tokens.mean(dim=1)
                )
                if variant == "token_no_attention":
                    expected_pooled = torch.cat(
                        [pooled_pollution, pooled_infrastructure], dim=-1
                    )
                elif variant == "forward":
                    assert model.forward_cross is not None
                    updated_infrastructure, _ = model.forward_cross(
                        result.socio_infrastructure_tokens,
                        result.pollution_environment_tokens,
                    )
                    expected_pooled = torch.cat(
                        [pooled_pollution, updated_infrastructure.mean(dim=1)],
                        dim=-1,
                    )
                elif variant == "reverse":
                    assert model.reverse_cross is not None
                    updated_pollution, _ = model.reverse_cross(
                        result.pollution_environment_tokens,
                        result.socio_infrastructure_tokens,
                    )
                    expected_pooled = torch.cat(
                        [updated_pollution.mean(dim=1), pooled_infrastructure],
                        dim=-1,
                    )
                else:
                    assert model.forward_cross is not None
                    assert model.reverse_cross is not None
                    updated_infrastructure, _ = model.forward_cross(
                        result.socio_infrastructure_tokens,
                        result.pollution_environment_tokens,
                    )
                    updated_pollution, _ = model.reverse_cross(
                        result.pollution_environment_tokens,
                        result.socio_infrastructure_tokens,
                    )
                    expected_pooled = torch.cat(
                        [
                            updated_pollution.mean(dim=1),
                            updated_infrastructure.mean(dim=1),
                        ],
                        dim=-1,
                    )
                torch.testing.assert_close(
                    result.pooled_representation,
                    expected_pooled,
                    rtol=0,
                    atol=0,
                )
                state_shapes = {
                    name: tuple(value.shape)
                    for name, value in model.regression_head.state_dict().items()
                }
                if expected_state_shapes is None:
                    expected_state_shapes = state_shapes
                else:
                    self.assertEqual(state_shapes, expected_state_shapes)

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
        normalization_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, torch.nn.LayerNorm)
            for parameter in module.parameters(recurse=False)
        }
        for name, parameter in model.named_parameters():
            if name == "bias" or name.endswith(".bias") or id(parameter) in normalization_ids:
                self.assertIn(id(parameter), no_decay_ids, name)
                self.assertNotIn(id(parameter), decay_ids, name)
            else:
                self.assertIn(id(parameter), decay_ids, name)

        class ArbitraryVectorModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_scale = torch.nn.Parameter(torch.ones(5))
                self.normalization = torch.nn.LayerNorm(5)
                self.projection = torch.nn.Linear(5, 1)

        fixture = ArbitraryVectorModel()
        fixture_groups = _adamw_parameter_groups(fixture, weight_decay=0.125)
        fixture_decay_ids = {id(parameter) for parameter in fixture_groups[0]["params"]}
        fixture_no_decay_ids = {
            id(parameter) for parameter in fixture_groups[1]["params"]
        }
        self.assertIn(id(fixture.feature_scale), fixture_decay_ids)
        self.assertNotIn(id(fixture.feature_scale), fixture_no_decay_ids)
        self.assertIn(id(fixture.normalization.weight), fixture_no_decay_ids)
        self.assertIn(id(fixture.normalization.bias), fixture_no_decay_ids)

    def test_variant_parameter_counts_are_reported_and_structurally_expected(self) -> None:
        counts = variant_parameter_counts(self.base_config)
        self.assertEqual(set(counts), set(CELL_MSCA_VARIANTS))
        self.assertTrue(all(count > 0 for count in counts.values()))
        self.assertEqual(counts["forward"], counts["reverse"])
        cross_attention_parameters = count_trainable_parameters(
            DirectionalCrossAttention(self.base_config)
        )
        self.assertEqual(
            counts["forward"] - counts["token_no_attention"],
            cross_attention_parameters,
        )
        self.assertEqual(
            counts["bidirectional"] - counts["token_no_attention"],
            2 * cross_attention_parameters,
        )
        self.assertEqual(
            counts["forward"],
            count_trainable_parameters(CellMSCA(self.base_config)),
        )
        self.assertEqual(
            variant_parameter_counts(),
            {
                "token_no_attention": 19_905,
                "forward": 28_577,
                "reverse": 28_577,
                "bidirectional": 37_249,
            },
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
            with mock.patch.object(torch, "load", wraps=torch.load) as safe_load:
                loaded = load_selected_checkpoint(
                    path,
                    device="cpu",
                    expected_hashes=expected_hashes,
                )
            self.assertTrue(safe_load.call_args.kwargs["weights_only"])
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
            self.assertEqual(
                loaded.provenance.git_dirty_state_policy,
                GIT_DIRTY_STATE_POLICY,
            )
            manifest = json.loads(checkpoint_manifest_json(loaded))
            self.assertEqual(
                manifest["git_dirty_state_policy"],
                GIT_DIRTY_STATE_POLICY,
            )
            with self.assertRaisesRegex(ValueError, "data_sha256 mismatch"):
                load_selected_checkpoint(
                    path,
                    expected_hashes={"data_sha256": "f" * 64},
                )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                save_selected_checkpoint(
                    path,
                    model=fitted.model,
                    config=config,
                    contract=fitted.contract,
                    provenance=fitted.checkpoint_provenance,
                    best_epoch=fitted.contract.best_epoch or 1,
                    validation_original_mae=0.0,
                )

            atomic_path = Path(directory) / "atomic.pt"
            with mock.patch("cell_msca.train.os.replace", wraps=os.replace) as replace_call:
                save_selected_checkpoint(
                    atomic_path,
                    model=fitted.model,
                    config=config,
                    contract=fitted.contract,
                    provenance=fitted.checkpoint_provenance,
                    best_epoch=fitted.contract.best_epoch or 1,
                    validation_original_mae=0.0,
                )
            self.assertTrue(atomic_path.is_file())
            replace_call.assert_called_once()
            temporary_source, final_destination = replace_call.call_args.args
            self.assertEqual(Path(temporary_source).parent, atomic_path.parent)
            self.assertEqual(Path(final_destination), atomic_path)
            self.assertFalse(Path(temporary_source).exists())

            failed_path = Path(directory) / "failed.pt"

            def fail_after_partial_write(_payload: object, temporary: Path) -> None:
                Path(temporary).write_bytes(b"partial")
                raise RuntimeError("synthetic checkpoint write failure")

            with mock.patch(
                "cell_msca.train.torch.save",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic checkpoint"):
                    save_selected_checkpoint(
                        failed_path,
                        model=fitted.model,
                        config=config,
                        contract=fitted.contract,
                        provenance=fitted.checkpoint_provenance,
                        best_epoch=fitted.contract.best_epoch or 1,
                        validation_original_mae=0.0,
                    )
            self.assertFalse(failed_path.exists())
            self.assertEqual(
                list(Path(directory).glob(f".{failed_path.name}.*.tmp")),
                [],
            )

    def test_git_dirty_state_includes_untracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for command in (
                ["git", "init", "-b", "main"],
                ["git", "config", "user.name", "Phase 4 Test"],
                ["git", "config", "user.email", "phase4@example.invalid"],
            ):
                subprocess.run(command, cwd=root, check=True, capture_output=True)
            tracked = root / "tracked.txt"
            tracked.write_text("tracked\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "tracked.txt"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "test fixture"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            _commit, dirty = discover_git_state(root)
            self.assertFalse(dirty)
            (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
            _commit, dirty = discover_git_state(root)
            self.assertTrue(dirty)

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
