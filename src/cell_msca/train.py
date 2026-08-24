"""Reusable Phase 4 neural training, provenance, and checkpoint contracts."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from numpy.typing import NDArray

from .baselines import (
    BaselinePredictions,
    ModelContract,
    TuningData,
    _validated_feature_count,
)
from .data import STREAM_A_FEATURES, STREAM_B_FEATURES, canonical_sha256
from .model import CellMSCA, CellMSCAConfig, count_trainable_parameters
from .neural_baselines import (
    NeuralLoss,
    _predict_log,
    _resolve_torch_device,
    fit_log_neural_model,
)
from .target import inverse_target

CELL_MSCA_CHECKPOINT_SCHEMA_VERSION = "cell_msca.selected_checkpoint.v1"
GIT_DIRTY_STATE_POLICY = "tracked_and_untracked_files"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True)
class CellMSCATrainingConfig:
    """Architecture plus the Phase 3 log-target neural selection contract."""

    architecture: CellMSCAConfig = field(default_factory=CellMSCAConfig)
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
        if self.loss not in {"log_l1", "log_huber"}:
            raise ValueError("Cell-MSCA loss must be 'log_l1' or 'log_huber'")
        if not np.isfinite(self.huber_delta) or self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be finite and positive")
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

    @property
    def model_name(self) -> str:
        return f"cell_msca_{self.architecture.variant}"

    def to_dict(self, *, train_seed: int, target_scale: float) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture.to_dict(),
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
            "inverse_selection": "post_checkpoint_validation_original_unit_mae",
            "duan_residual_source": "train_only",
            "train_seed": train_seed,
            "target_transform": "log1p",
            "target_scale": target_scale,
        }


@dataclass(frozen=True)
class CellMSCACheckpointProvenance:
    data_version: str
    data_sha256: str
    split_sha256: str
    split_config_sha256: str
    preprocessing_sha256: str
    configuration_sha256: str
    split_seed: int
    train_seed: int
    target_scale: float
    git_commit_sha: str
    git_worktree_dirty: bool
    python_version: str
    pytorch_version: str
    numpy_version: str
    package_versions: dict[str, str]
    device: str
    git_dirty_state_policy: str = GIT_DIRTY_STATE_POLICY

    def __post_init__(self) -> None:
        for name in (
            "data_sha256",
            "split_sha256",
            "split_config_sha256",
            "preprocessing_sha256",
            "configuration_sha256",
        ):
            value = str(getattr(self, name)).lower()
            if not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
            object.__setattr__(self, name, value)
        git_sha = self.git_commit_sha.lower()
        if not _GIT_SHA_PATTERN.fullmatch(git_sha):
            raise ValueError("git_commit_sha must be a 40-64 character hex digest")
        object.__setattr__(self, "git_commit_sha", git_sha)
        if self.git_dirty_state_policy != GIT_DIRTY_STATE_POLICY:
            raise ValueError(
                "git_dirty_state_policy must include tracked and untracked files"
            )
        if self.split_seed == self.train_seed:
            # Equal values are allowed, but the separately named fields remain mandatory.
            pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FittedCellMSCA:
    model: CellMSCA
    contract: ModelContract
    parameter_count: int
    device: str
    checkpoint_path: Path | None = None
    checkpoint_provenance: CellMSCACheckpointProvenance | None = None

    def predict(self, features: NDArray[np.float64]) -> BaselinePredictions:
        _validated_feature_count(features)
        pred_log = _predict_log(
            torch,
            self.model,
            np.asarray(features, dtype=np.float64),
            device=torch.device(self.device),
            forward_batch=_cell_msca_forward,
        )
        pred_original = inverse_target(
            pred_log,
            scale=self.contract.target_scale,
            mode=self.contract.inverse_mode,
            smearing_factor=self.contract.smearing_factor,
        )
        return BaselinePredictions(pred_original, pred_log)


@dataclass(frozen=True)
class LoadedCellMSCACheckpoint:
    model: CellMSCA
    contract: ModelContract
    provenance: CellMSCACheckpointProvenance
    parameter_count: int
    best_epoch: int


def fit_cell_msca(
    data: TuningData,
    config: CellMSCATrainingConfig | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> FittedCellMSCA:
    """Fit using train/validation only and optionally persist the selected checkpoint."""

    config = config or CellMSCATrainingConfig()
    training = fit_log_neural_model(
        data,
        config,
        model_factory=lambda _torch, _input_size, candidate: CellMSCA(
            candidate.architecture
        ),
        forward_batch=_cell_msca_forward,
    )
    model = training.model
    config_sha256 = canonical_sha256(
        config.to_dict(
            train_seed=data.provenance.train_seed,
            target_scale=data.provenance.target_scale,
        )
    )
    contract = ModelContract(
        model_name=config.model_name,
        config_sha256=config_sha256,
        target_transform="log1p",
        target_scale=data.provenance.target_scale,
        loss_objective=f"{config.loss}_on_log1p_target",
        inverse_mode=training.inverse_selection.inverse_mode,
        smearing_factor=training.inverse_selection.smearing_factor,
        best_epoch=training.best_epoch,
    )
    parameter_count = count_trainable_parameters(model)
    saved_path: Path | None = None
    checkpoint_provenance: CellMSCACheckpointProvenance | None = None
    if checkpoint_path is not None:
        git_sha, dirty = discover_git_state(repository_root)
        checkpoint_provenance = build_checkpoint_provenance(
            data,
            config,
            configuration_sha256=config_sha256,
            device=training.device,
            git_commit_sha=git_sha,
            git_worktree_dirty=dirty,
        )
        saved_path = save_selected_checkpoint(
            checkpoint_path,
            model=model,
            config=config,
            contract=contract,
            provenance=checkpoint_provenance,
            best_epoch=training.best_epoch,
            validation_original_mae=(
                training.inverse_selection.validation_original_mae
            ),
            parameter_count=parameter_count,
        )
    return FittedCellMSCA(
        model=model,
        contract=contract,
        parameter_count=parameter_count,
        device=training.device,
        checkpoint_path=saved_path,
        checkpoint_provenance=checkpoint_provenance,
    )


def build_checkpoint_provenance(
    data: TuningData,
    config: CellMSCATrainingConfig,
    *,
    configuration_sha256: str,
    device: str,
    git_commit_sha: str,
    git_worktree_dirty: bool,
) -> CellMSCACheckpointProvenance:
    packages: dict[str, str] = {}
    for distribution in (
        "torch",
        "numpy",
        "filelock",
        "fsspec",
        "networkx",
        "sympy",
        "typing_extensions",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return CellMSCACheckpointProvenance(
        data_version=data.provenance.data_version,
        data_sha256=data.provenance.data_sha256,
        split_sha256=data.provenance.split_sha256,
        split_config_sha256=data.provenance.split_config_sha256,
        preprocessing_sha256=data.provenance.preprocessing_sha256,
        configuration_sha256=configuration_sha256,
        split_seed=data.provenance.split_seed,
        train_seed=data.provenance.train_seed,
        target_scale=data.provenance.target_scale,
        git_commit_sha=git_commit_sha,
        git_worktree_dirty=bool(git_worktree_dirty),
        python_version=platform.python_version(),
        pytorch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
        package_versions=packages,
        device=device,
    )


def save_selected_checkpoint(
    path: str | Path,
    *,
    model: CellMSCA,
    config: CellMSCATrainingConfig,
    contract: ModelContract,
    provenance: CellMSCACheckpointProvenance,
    best_epoch: int,
    validation_original_mae: float,
    parameter_count: int | None = None,
) -> Path:
    """Persist one validation-selected model without overwriting any artifact."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    if best_epoch <= 0:
        raise ValueError("best_epoch must be positive for a selected checkpoint")
    if not np.isfinite(validation_original_mae) or validation_original_mae < 0.0:
        raise ValueError("validation_original_mae must be finite and non-negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = (
        count_trainable_parameters(model)
        if parameter_count is None
        else parameter_count
    )
    payload = {
        "schema_version": CELL_MSCA_CHECKPOINT_SCHEMA_VERSION,
        "architecture_config": asdict(config.architecture),
        "training_config": config.to_dict(
            train_seed=provenance.train_seed,
            target_scale=provenance.target_scale,
        ),
        "model_contract": asdict(contract),
        "selection": {
            "best_epoch": int(best_epoch),
            "checkpoint_metric": (
                "median_inverse_validation_original_unit_mae"
            ),
            "validation_original_mae": float(validation_original_mae),
            "inverse_mode": contract.inverse_mode,
            "smearing_factor": contract.smearing_factor,
            "test_evaluated": False,
        },
        "provenance": provenance.to_dict(),
        "parameter_count": int(count),
        "feature_order": {
            "pollution_environment": list(STREAM_A_FEATURES),
            "socio_infrastructure": list(STREAM_B_FEATURES),
        },
        "model_state_dict": model.state_dict(),
    }
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        torch.save(payload, temporary_path)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite checkpoint: {destination}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def load_selected_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    expected_hashes: Mapping[str, str] | None = None,
) -> LoadedCellMSCACheckpoint:
    """Reload a selected model and reject provenance mismatches."""

    resolved_device = _resolve_torch_device(torch, device)
    payload = torch.load(
        Path(path),
        map_location=resolved_device,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Cell-MSCA checkpoint payload must be a mapping")
    if payload.get("schema_version") != CELL_MSCA_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported Cell-MSCA checkpoint schema_version")
    required = {
        "architecture_config",
        "training_config",
        "model_contract",
        "selection",
        "provenance",
        "parameter_count",
        "feature_order",
        "model_state_dict",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Cell-MSCA checkpoint is missing fields: {missing}")
    if payload["feature_order"] != {
        "pollution_environment": list(STREAM_A_FEATURES),
        "socio_infrastructure": list(STREAM_B_FEATURES),
    }:
        raise ValueError("Cell-MSCA checkpoint feature order mismatch")
    provenance = CellMSCACheckpointProvenance(**payload["provenance"])
    stored_configuration_sha256 = canonical_sha256(payload["training_config"])
    if stored_configuration_sha256 != provenance.configuration_sha256:
        raise ValueError("Cell-MSCA checkpoint configuration_sha256 mismatch")
    for name, expected in dict(expected_hashes or {}).items():
        if not hasattr(provenance, name):
            raise ValueError(f"unknown expected provenance field: {name}")
        if getattr(provenance, name) != expected:
            raise ValueError(f"Cell-MSCA checkpoint {name} mismatch")
    architecture = CellMSCAConfig(**payload["architecture_config"])
    model = CellMSCA(architecture).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    parameter_count = count_trainable_parameters(model)
    if parameter_count != int(payload["parameter_count"]):
        raise ValueError("Cell-MSCA checkpoint parameter_count mismatch")
    contract = ModelContract(**payload["model_contract"])
    if contract.config_sha256 != provenance.configuration_sha256:
        raise ValueError("Cell-MSCA checkpoint model contract hash mismatch")
    selection = payload["selection"]
    if selection.get("test_evaluated") is not False:
        raise ValueError("Phase 4.1 checkpoint must not record test evaluation")
    return LoadedCellMSCACheckpoint(
        model=model,
        contract=contract,
        provenance=provenance,
        parameter_count=parameter_count,
        best_epoch=int(selection["best_epoch"]),
    )


def discover_git_state(repository_root: str | Path | None = None) -> tuple[str, bool]:
    """Return the commit and tracked-plus-untracked worktree dirty state."""

    root = Path(repository_root) if repository_root is not None else Path.cwd()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not _GIT_SHA_PATTERN.fullmatch(commit):
        raise RuntimeError("git rev-parse returned an invalid commit SHA")
    return commit, bool(status.strip())


def _cell_msca_forward(model: CellMSCA, features: torch.Tensor) -> torch.Tensor:
    expected = len(STREAM_A_FEATURES) + len(STREAM_B_FEATURES)
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(f"Cell-MSCA trainer expects [batch, {expected}] features")
    split = len(STREAM_A_FEATURES)
    return model(features[:, :split], features[:, split:])


def checkpoint_manifest_json(checkpoint: LoadedCellMSCACheckpoint) -> str:
    """Return a JSON-safe summary without model weights."""

    return json.dumps(
        {
            "schema_version": CELL_MSCA_CHECKPOINT_SCHEMA_VERSION,
            "model_contract": asdict(checkpoint.contract),
            "provenance": checkpoint.provenance.to_dict(),
            "git_dirty_state_policy": checkpoint.provenance.git_dirty_state_policy,
            "parameter_count": checkpoint.parameter_count,
            "best_epoch": checkpoint.best_epoch,
            "test_evaluated": False,
        },
        sort_keys=True,
        indent=2,
    )
