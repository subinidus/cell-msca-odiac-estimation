"""Single-cell feature-token Cell-MSCA architectures for Phase 4.

PyTorch is intentionally imported only when this module is imported. The base
``cell_msca`` package remains usable for non-neural evaluation without PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .data import STREAM_A_FEATURES, STREAM_B_FEATURES

CellMSCAVariant = Literal[
    "token_no_attention",
    "forward",
    "reverse",
    "bidirectional",
]

POLLUTION_ENVIRONMENT_FEATURES = STREAM_A_FEATURES
SOCIO_INFRASTRUCTURE_FEATURES = STREAM_B_FEATURES
CELL_MSCA_VARIANTS: tuple[CellMSCAVariant, ...] = (
    "token_no_attention",
    "forward",
    "reverse",
    "bidirectional",
)


@dataclass(frozen=True)
class CellMSCAConfig:
    """Frozen architecture-only configuration shared by all variants."""

    variant: CellMSCAVariant = "forward"
    d_model: int = 32
    num_heads: int = 4
    encoder_layers: int = 1
    ffn_multiplier: int = 2
    head_hidden: int = 32
    dropout: float = 0.1
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.variant not in CELL_MSCA_VARIANTS:
            raise ValueError(
                f"variant must be one of {CELL_MSCA_VARIANTS}; got {self.variant!r}"
            )
        integer_fields = {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "encoder_layers": self.encoder_layers,
            "ffn_multiplier": self.ffn_multiplier,
            "head_hidden": self.head_hidden,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        for name, value in {
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
        }.items():
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update(
            {
                "pollution_environment_features": list(
                    POLLUTION_ENVIRONMENT_FEATURES
                ),
                "socio_infrastructure_features": list(
                    SOCIO_INFRASTRUCTURE_FEATURES
                ),
                "tokenizer": "feature_specific_affine_plus_feature_and_group_embedding",
                "encoder_contract": "pre_layernorm_residual_self_attention_ffn",
                "cross_attention_contract": (
                    "pre_layernorm_q_kv_residual_ffn_same_sample_only"
                ),
                "capacity_note": (
                    "tokenizers and stream encoders are matched; directional "
                    "cross-attention blocks and one-versus-two-stream pooled heads "
                    "create unavoidable variant-specific parameter counts"
                ),
            }
        )
        return values


@dataclass(frozen=True)
class CellMSCAForwardResult:
    """Detailed forward result used by structural and isolation tests."""

    prediction_log: Tensor
    pollution_environment_tokens: Tensor
    socio_infrastructure_tokens: Tensor
    forward_attention: Tensor | None
    reverse_attention: Tensor | None


class FeatureSpecificNumericalTokenizer(nn.Module):
    """Map each continuous scalar to its own learnable feature token."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        if n_features <= 0 or d_model <= 0:
            raise ValueError("n_features and d_model must be positive")
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.feature_embedding = nn.Parameter(torch.empty(n_features, d_model))
        self.group_embedding = nn.Parameter(torch.empty(d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(
                f"numerical tokenizer expects [batch, {self.n_features}]; "
                f"got {tuple(values.shape)}"
            )
        return (
            values.unsqueeze(-1) * self.weight.unsqueeze(0)
            + self.bias.unsqueeze(0)
            + self.feature_embedding.unsqueeze(0)
            + self.group_embedding.view(1, 1, -1)
        )


def _feed_forward(d_model: int, ffn_multiplier: int, dropout: float) -> nn.Module:
    hidden = d_model * ffn_multiplier
    return nn.Sequential(
        nn.Linear(d_model, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_model),
        nn.Dropout(dropout),
    )


class TokenEncoderBlock(nn.Module):
    """Independent pre-LayerNorm self-attention block for one feature stream."""

    def __init__(self, config: CellMSCAConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _feed_forward(
            config.d_model,
            config.ffn_multiplier,
            config.dropout,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class StreamTokenEncoder(nn.Module):
    """A capacity-matched stack used independently for both streams."""

    def __init__(self, config: CellMSCAConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TokenEncoderBlock(config) for _ in range(config.encoder_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)


class DirectionalCrossAttention(nn.Module):
    """Update query tokens from context tokens without mixing batch samples."""

    def __init__(self, config: CellMSCAConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.d_model)
        self.context_norm = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = _feed_forward(
            config.d_model,
            config.ffn_multiplier,
            config.dropout,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, query_tokens: Tensor, context_tokens: Tensor) -> tuple[Tensor, Tensor]:
        normalized_query = self.query_norm(query_tokens)
        normalized_context = self.context_norm(context_tokens)
        attended, weights = self.attention(
            normalized_query,
            normalized_context,
            normalized_context,
            need_weights=True,
            average_attn_weights=False,
        )
        updated = query_tokens + self.attention_dropout(attended)
        updated = updated + self.ffn(self.ffn_norm(updated))
        return self.output_norm(updated), weights


class CellMSCA(nn.Module):
    """Feature-token model for one cell and one month.

    ``forward`` returns one log-target prediction per sample. Use
    ``forward_with_details`` only for structural validation and diagnostics;
    attention values are not causal explanations.
    """

    def __init__(self, config: CellMSCAConfig | None = None) -> None:
        super().__init__()
        self.config = config or CellMSCAConfig()
        self.pollution_tokenizer = FeatureSpecificNumericalTokenizer(
            len(POLLUTION_ENVIRONMENT_FEATURES),
            self.config.d_model,
        )
        self.infrastructure_tokenizer = FeatureSpecificNumericalTokenizer(
            len(SOCIO_INFRASTRUCTURE_FEATURES),
            self.config.d_model,
        )
        self.pollution_encoder = StreamTokenEncoder(self.config)
        self.infrastructure_encoder = StreamTokenEncoder(self.config)

        self.forward_cross: DirectionalCrossAttention | None = None
        self.reverse_cross: DirectionalCrossAttention | None = None
        if self.config.variant in {"forward", "bidirectional"}:
            self.forward_cross = DirectionalCrossAttention(self.config)
        if self.config.variant in {"reverse", "bidirectional"}:
            self.reverse_cross = DirectionalCrossAttention(self.config)

        pooled_width = (
            self.config.d_model
            if self.config.variant in {"forward", "reverse"}
            else self.config.d_model * 2
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(pooled_width),
            nn.Linear(pooled_width, self.config.head_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.head_hidden, 1),
        )

    def forward(self, pollution_environment: Tensor, socio_infrastructure: Tensor) -> Tensor:
        return self.forward_with_details(
            pollution_environment,
            socio_infrastructure,
        ).prediction_log

    def forward_with_details(
        self,
        pollution_environment: Tensor,
        socio_infrastructure: Tensor,
    ) -> CellMSCAForwardResult:
        pollution_environment, socio_infrastructure = self._validated_inputs(
            pollution_environment,
            socio_infrastructure,
        )
        pollution_tokens = self.pollution_encoder(
            self.pollution_tokenizer(pollution_environment)
        )
        infrastructure_tokens = self.infrastructure_encoder(
            self.infrastructure_tokenizer(socio_infrastructure)
        )

        forward_attention: Tensor | None = None
        reverse_attention: Tensor | None = None
        if self.config.variant == "token_no_attention":
            pooled = torch.cat(
                [pollution_tokens.mean(dim=1), infrastructure_tokens.mean(dim=1)],
                dim=-1,
            )
        elif self.config.variant == "forward":
            if self.forward_cross is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("forward cross-attention module is missing")
            updated, forward_attention = self.forward_cross(
                infrastructure_tokens,
                pollution_tokens,
            )
            pooled = updated.mean(dim=1)
        elif self.config.variant == "reverse":
            if self.reverse_cross is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("reverse cross-attention module is missing")
            updated, reverse_attention = self.reverse_cross(
                pollution_tokens,
                infrastructure_tokens,
            )
            pooled = updated.mean(dim=1)
        else:
            if self.forward_cross is None or self.reverse_cross is None:
                raise RuntimeError("bidirectional cross-attention modules are missing")
            updated_infrastructure, forward_attention = self.forward_cross(
                infrastructure_tokens,
                pollution_tokens,
            )
            updated_pollution, reverse_attention = self.reverse_cross(
                pollution_tokens,
                infrastructure_tokens,
            )
            pooled = torch.cat(
                [
                    updated_pollution.mean(dim=1),
                    updated_infrastructure.mean(dim=1),
                ],
                dim=-1,
            )

        prediction = self.regression_head(pooled).squeeze(-1)
        return CellMSCAForwardResult(
            prediction_log=prediction,
            pollution_environment_tokens=pollution_tokens,
            socio_infrastructure_tokens=infrastructure_tokens,
            forward_attention=forward_attention,
            reverse_attention=reverse_attention,
        )

    def _validated_inputs(
        self,
        pollution_environment: Tensor,
        socio_infrastructure: Tensor,
    ) -> tuple[Tensor, Tensor]:
        expected_pollution = len(POLLUTION_ENVIRONMENT_FEATURES)
        expected_infrastructure = len(SOCIO_INFRASTRUCTURE_FEATURES)
        if not isinstance(pollution_environment, Tensor) or not isinstance(
            socio_infrastructure,
            Tensor,
        ):
            raise TypeError("Cell-MSCA inputs must be PyTorch tensors")
        if pollution_environment.ndim != 2 or pollution_environment.shape[1] != expected_pollution:
            raise ValueError(
                "pollution/environment input must have shape "
                f"[batch, {expected_pollution}]; got {tuple(pollution_environment.shape)}"
            )
        if (
            socio_infrastructure.ndim != 2
            or socio_infrastructure.shape[1] != expected_infrastructure
        ):
            raise ValueError(
                "socio-infrastructure input must have shape "
                f"[batch, {expected_infrastructure}]; got {tuple(socio_infrastructure.shape)}"
            )
        if pollution_environment.shape[0] == 0:
            raise ValueError("Cell-MSCA inputs must contain at least one sample")
        if pollution_environment.shape[0] != socio_infrastructure.shape[0]:
            raise ValueError("both Cell-MSCA streams must have the same batch size")
        if pollution_environment.device != socio_infrastructure.device:
            raise ValueError("both Cell-MSCA streams must be on the same device")
        if not torch.is_floating_point(pollution_environment) or not torch.is_floating_point(
            socio_infrastructure
        ):
            raise TypeError("Cell-MSCA inputs must use a floating-point dtype")
        if not torch.isfinite(pollution_environment).all() or not torch.isfinite(
            socio_infrastructure
        ).all():
            raise ValueError("Cell-MSCA inputs must contain only finite values")
        dtype = self.pollution_tokenizer.weight.dtype
        return (
            pollution_environment.to(dtype=dtype),
            socio_infrastructure.to(dtype=dtype),
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def variant_parameter_counts(config: CellMSCAConfig | None = None) -> dict[str, int]:
    """Instantiate capacity-matched variants and report unavoidable differences."""

    base = config or CellMSCAConfig()
    return {
        variant: count_trainable_parameters(
            CellMSCA(replace(base, variant=variant))
        )
        for variant in CELL_MSCA_VARIANTS
    }
