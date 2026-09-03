"""Coordinate- and sparse-set-conditioned flow for HRTF scalar fields."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.fm.cond import t_embed


@dataclass(frozen=True)
class HRTFFlowConfig:
    width: int = 128
    depth: int = 4
    context_dim: int = 128
    observation_width: int = 96
    coordinate_bands: int = 4
    time_dim: int = 32
    frequency_scale_hz: float = 15000.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def coordinate_features(unit_coordinates: torch.Tensor, bands: int) -> torch.Tensor:
    """Smooth Cartesian Fourier features suitable for second derivatives."""

    if bands < 0:
        raise ValueError("bands cannot be negative")
    features = [unit_coordinates]
    for band in range(bands):
        phase = unit_coordinates * (math.pi * (2**band))
        features.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(features, dim=-1)


class TanhMLP(nn.Module):
    """A smooth MLP; tanh is required because the loss uses second derivatives."""

    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int) -> None:
        super().__init__()
        if depth < 1 or width < 1:
            raise ValueError("width and depth must be positive")
        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SparseObservationEncoder(nn.Module):
    """Permutation-invariant DeepSets encoder for measured HRTF samples."""

    def __init__(self, coordinate_dim: int, width: int, context_dim: int) -> None:
        super().__init__()
        self.token_mlp = TanhMLP(coordinate_dim + 1, width, width, depth=2)
        self.output_mlp = TanhMLP(2 * width, context_dim, width, depth=2)

    def forward(
        self, coordinate_encoding: torch.Tensor, observed_values: torch.Tensor
    ) -> torch.Tensor:
        if observed_values.ndim == 2:
            observed_values = observed_values.unsqueeze(-1)
        tokens = self.token_mlp(torch.cat((coordinate_encoding, observed_values), dim=-1))
        pooled = torch.cat((tokens.mean(dim=1), tokens.amax(dim=1)), dim=-1)
        return self.output_mlp(pooled)


class ConditionalHRTFFieldFlow(nn.Module):
    """Shared conditional velocity field for Ma's 28 subject-40 field tasks."""

    def __init__(self, config: HRTFFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or HRTFFlowConfig()
        if self.config.time_dim < 2 or self.config.time_dim % 2:
            raise ValueError("time_dim must be a positive even number")
        if self.config.frequency_scale_hz <= 0:
            raise ValueError("frequency_scale_hz must be positive")
        coordinate_dim = 3 * (1 + 2 * self.config.coordinate_bands)
        self.observation_encoder = SparseObservationEncoder(
            coordinate_dim,
            self.config.observation_width,
            self.config.context_dim,
        )
        # frequency (linear + log), real/imag one-hot, and hemisphere one-hot
        field_descriptor_dim = 2 + 2 + 2
        self.context_projection = TanhMLP(
            self.config.context_dim + field_descriptor_dim,
            self.config.context_dim,
            self.config.width,
            depth=2,
        )
        query_input_dim = (
            1 + coordinate_dim + self.config.time_dim + self.config.context_dim
        )
        self.velocity_network = TanhMLP(
            query_input_dim,
            1,
            self.config.width,
            self.config.depth,
        )

    def encode_condition(
        self,
        observed_unit_xyz: torch.Tensor,
        observed_values: torch.Tensor,
        frequency_hz: torch.Tensor,
        component: torch.Tensor,
        hemisphere: torch.Tensor,
    ) -> torch.Tensor:
        encoded_coordinates = coordinate_features(
            observed_unit_xyz, self.config.coordinate_bands
        )
        sparse_context = self.observation_encoder(encoded_coordinates, observed_values)
        frequency = frequency_hz.to(dtype=observed_unit_xyz.dtype).reshape(-1)
        scaled_frequency = frequency / self.config.frequency_scale_hz
        frequency_descriptor = torch.stack(
            (
                scaled_frequency,
                torch.log1p(frequency) / math.log1p(self.config.frequency_scale_hz),
            ),
            dim=-1,
        )
        component_descriptor = F.one_hot(component.long(), num_classes=2).to(
            observed_unit_xyz.dtype
        )
        hemisphere_descriptor = F.one_hot(hemisphere.long(), num_classes=2).to(
            observed_unit_xyz.dtype
        )
        descriptor = torch.cat(
            (frequency_descriptor, component_descriptor, hemisphere_descriptor), dim=-1
        )
        return self.context_projection(torch.cat((sparse_context, descriptor), dim=-1))

    def velocity_with_context(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        query_unit_xyz: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim == 3 and state.shape[-1] == 1:
            state = state.squeeze(-1)
        batch_size, points = state.shape
        if query_unit_xyz.shape != (batch_size, points, 3):
            raise ValueError("query coordinates and state shapes are inconsistent")
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        if context.shape != (batch_size, self.config.context_dim):
            raise ValueError("context has an unexpected shape")
        query_features = coordinate_features(
            query_unit_xyz, self.config.coordinate_bands
        )
        time_features = t_embed(time, dim=self.config.time_dim)
        inputs = torch.cat(
            (
                state.unsqueeze(-1),
                query_features,
                time_features[:, None, :].expand(-1, points, -1),
                context[:, None, :].expand(-1, points, -1),
            ),
            dim=-1,
        )
        return self.velocity_network(inputs).squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        query_unit_xyz: torch.Tensor,
        observed_unit_xyz: torch.Tensor,
        observed_values: torch.Tensor,
        frequency_hz: torch.Tensor,
        component: torch.Tensor,
        hemisphere: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_condition(
            observed_unit_xyz,
            observed_values,
            frequency_hz,
            component,
            hemisphere,
        )
        return self.velocity_with_context(state, time, query_unit_xyz, context)
