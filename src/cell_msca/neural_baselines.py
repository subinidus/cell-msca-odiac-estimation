"""Lazy-PyTorch neural baseline and reusable log-target trainer for Phase 3."""

from __future__ import annotations

import copy
import importlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .baselines import (
    BaselinePredictions,
    LogInverseSelection,
    ModelContract,
    TuningData,
    _validated_feature_count,
    select_log_inverse_on_validation,
)
from .data import STREAM_A_FEATURES, STREAM_B_FEATURES, canonical_sha256
from .metrics import regression_metrics
from .target import inverse_target

NeuralLoss = Literal["log_l1", "log_huber"]
NeuralModelFactory = Callable[[Any, int, Any], Any]
NeuralForward = Callable[[Any, Any], Any]


class LogNeuralTrainingConfig(Protocol):
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    weight_decay: float
    loss: NeuralLoss
    huber_delta: float
    device: str


@dataclass(frozen=True)
class ConcatMLPConfig:
    model_name: str = "concat_mlp"
    hidden_sizes: tuple[int, ...] = (32, 16)
    dropout: float = 0.1
    loss: NeuralLoss = "log_huber"
    huber_delta: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 0.0
    weight_decay: float = 1e-5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.model_name != "concat_mlp":
            raise ValueError("Concat-MLP model_name must be 'concat_mlp'")
        hidden_sizes = tuple(int(size) for size in self.hidden_sizes)
        object.__setattr__(self, "hidden_sizes", hidden_sizes)
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if self.loss not in {"log_l1", "log_huber"}:
            raise ValueError("Concat-MLP loss must be 'log_l1' or 'log_huber'")
        if not np.isfinite(self.huber_delta) or self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be finite and positive")
        if not np.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if not np.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

    def to_dict(self, *, train_seed: int, target_scale: float) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "loss": self.loss,
            "huber_delta": self.huber_delta,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "weight_decay": self.weight_decay,
            "optimizer": "AdamW",
            "weight_decay_scope": "weights_only_no_bias_or_normalization",
            "device": self.device,
            "deterministic_algorithms": True,
            "checkpoint_metric": "median_inverse_validation_original_unit_mae",
            "train_seed": train_seed,
            "target_transform": "log1p",
            "target_scale": target_scale,
        }


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: Any
    best_epoch: int
    inverse_selection: LogInverseSelection
    device: str


@dataclass(frozen=True)
class FittedConcatMLP:
    model: Any
    contract: ModelContract

    def predict(self, features: NDArray[np.float64]) -> BaselinePredictions:
        _validated_feature_count(features)
        torch = _require_torch()
        pred_log = _predict_log(
            torch,
            self.model,
            np.asarray(features, dtype=np.float64),
        )
        pred_original = inverse_target(
            pred_log,
            scale=self.contract.target_scale,
            mode=self.contract.inverse_mode,
            smearing_factor=self.contract.smearing_factor,
        )
        return BaselinePredictions(pred_original, pred_log)


def fit_concat_mlp(
    data: TuningData,
    config: ConcatMLPConfig | None = None,
) -> FittedConcatMLP:
    """Build a seven-feature MLP and train through the reusable neural trainer."""

    config = config or ConcatMLPConfig()
    training = fit_log_neural_model(
        data,
        config,
        model_factory=_build_concat_mlp,
    )
    config_sha256 = canonical_sha256(
        config.to_dict(
            train_seed=data.provenance.train_seed,
            target_scale=data.provenance.target_scale,
        )
    )
    return FittedConcatMLP(
        model=training.model,
        contract=ModelContract(
            model_name=config.model_name,
            config_sha256=config_sha256,
            target_transform="log1p",
            target_scale=data.provenance.target_scale,
            loss_objective=f"{config.loss}_on_log1p_target",
            inverse_mode=training.inverse_selection.inverse_mode,
            smearing_factor=training.inverse_selection.smearing_factor,
            best_epoch=training.best_epoch,
        ),
    )


def fit_log_neural_model(
    data: TuningData,
    config: LogNeuralTrainingConfig,
    *,
    model_factory: NeuralModelFactory,
    forward_batch: NeuralForward | None = None,
) -> NeuralTrainingResult:
    """Train a log-target neural model with a validation-only checkpoint.

    Cell-MSCA can reuse this trainer by supplying its own model factory. During
    training, checkpoint selection always uses median inverse validation MAE.
    Duan is calculated from train residuals and compared only after the selected
    checkpoint has been restored.
    """

    torch = _require_torch()
    train_seed = data.provenance.train_seed
    _set_deterministic_seed(torch, train_seed)
    device = _resolve_torch_device(torch, config.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(train_seed)
    input_size = len(STREAM_A_FEATURES) + len(STREAM_B_FEATURES)
    model = model_factory(torch, input_size, config).to(device)
    run_forward = forward_batch or _default_forward
    optimizer = torch.optim.AdamW(
        _adamw_parameter_groups(model, weight_decay=config.weight_decay),
        lr=config.learning_rate,
    )
    train_features = torch.as_tensor(
        data.train.features,
        dtype=torch.float32,
        device=device,
    )
    train_target = torch.as_tensor(
        data.train.target_log,
        dtype=torch.float32,
        device=device,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_mae = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(data.train.n_samples, generator=generator)
        for start in range(0, data.train.n_samples, config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            device_indices = batch_indices.to(device)
            prediction = run_forward(
                model,
                train_features[device_indices],
            ).reshape(-1)
            target = train_target[device_indices]
            loss = _log_loss(torch, prediction, target, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_pred_log = _predict_log(
            torch,
            model,
            data.validation.features,
            device=device,
            forward_batch=run_forward,
        )
        validation_mae = _median_inverse_validation_mae(
            validation_true_original=data.validation.target_original,
            validation_pred_log=validation_pred_log,
            target_scale=data.provenance.target_scale,
        )
        if validation_mae < best_validation_mae - config.min_delta:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_epoch == 0:
        raise RuntimeError("neural training produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    train_pred_log = _predict_log(
        torch,
        model,
        data.train.features,
        device=device,
        forward_batch=run_forward,
    )
    validation_pred_log = _predict_log(
        torch,
        model,
        data.validation.features,
        device=device,
        forward_batch=run_forward,
    )
    inverse_selection = select_log_inverse_on_validation(
        train_true_log=data.train.target_log,
        train_pred_log=train_pred_log,
        validation_true_original=data.validation.target_original,
        validation_pred_log=validation_pred_log,
        target_scale=data.provenance.target_scale,
    )
    return NeuralTrainingResult(model, best_epoch, inverse_selection, str(device))


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as error:
        raise ImportError(
            "neural baselines and Cell-MSCA require the optional 'torch' package; "
            "install PyTorch in an isolated execution environment before training"
        ) from error


def _build_concat_mlp(torch: Any, input_size: int, config: Any) -> Any:
    layers: list[Any] = []
    previous_size = input_size
    for hidden_size in config.hidden_sizes:
        layers.append(torch.nn.Linear(previous_size, hidden_size))
        layers.append(torch.nn.ReLU())
        if config.dropout > 0.0:
            layers.append(torch.nn.Dropout(p=config.dropout))
        previous_size = hidden_size
    layers.append(torch.nn.Linear(previous_size, 1))
    return torch.nn.Sequential(*layers)


def _adamw_parameter_groups(
    model: Any,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    weights: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not getattr(parameter, "requires_grad", True):
            continue
        parameter_ndim = getattr(parameter, "ndim", None)
        is_bias = name == "bias" or name.endswith(".bias")
        is_normalization = (
            parameter_ndim == 1
            or ".norm." in name
            or name.startswith("norm.")
            or "layernorm" in name.lower()
        )
        if is_bias or is_normalization:
            no_decay.append(parameter)
        else:
            weights.append(parameter)
    if not weights:
        raise ValueError("neural model has no trainable weight parameters")
    groups = [{"params": weights, "weight_decay": float(weight_decay)}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _log_loss(
    torch: Any,
    prediction: Any,
    target: Any,
    config: LogNeuralTrainingConfig,
) -> Any:
    if config.loss == "log_l1":
        return torch.nn.functional.l1_loss(prediction, target)
    if config.loss == "log_huber":
        return torch.nn.functional.huber_loss(
            prediction,
            target,
            delta=config.huber_delta,
        )
    raise ValueError("neural loss must be 'log_l1' or 'log_huber'")


def _predict_log(
    torch: Any,
    model: Any,
    features: NDArray[np.float64],
    *,
    device: Any | None = None,
    forward_batch: NeuralForward | None = None,
) -> NDArray[np.float64]:
    model.eval()
    resolved_device = device or _model_device(torch, model)
    run_forward = forward_batch or _default_forward
    with torch.no_grad():
        tensor = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=resolved_device,
        )
        prediction = run_forward(model, tensor).reshape(-1).detach().cpu().numpy()
    values = np.asarray(prediction, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("neural model produced non-finite log predictions")
    return values


def _default_forward(model: Any, features: Any) -> Any:
    return model(features)


def _model_device(torch: Any, model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_torch_device(torch: Any, requested: str) -> Any:
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no compatible CUDA device is available")
    return torch.device(requested)


def _set_deterministic_seed(torch: Any, seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("train_seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _median_inverse_validation_mae(
    *,
    validation_true_original: NDArray[np.float64],
    validation_pred_log: NDArray[np.float64],
    target_scale: float,
) -> float:
    pred_original = inverse_target(
        validation_pred_log,
        scale=target_scale,
        mode="median",
    )
    return float(
        regression_metrics(validation_true_original, pred_original)["mae"]
    )
