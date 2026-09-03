"""Finite-rank Helmholtz Gaussian reference fields for functional flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn


class HelmholtzGaussianField(nn.Module):
    """A reproducible Gaussian random plane-wave field.

    White noise is not a well-defined function-space Gaussian measure and has
    no spatial derivatives.  This finite Fourier expansion instead produces a
    smooth field with unit marginal variance.  Every plane-wave basis uses the
    HRTF frequency's wave number, so the source field itself satisfies the
    homogeneous Helmholtz equation.  The same latent coefficients can be
    evaluated consistently at sparse, collocation, and dense coordinates.
    """

    def __init__(
        self,
        modes: int = 16,
        radius_m: float = 0.09,
        speed_of_sound_m_s: float = 343.0,
        seed: int = 1729,
    ) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be positive")
        if radius_m <= 0 or speed_of_sound_m_s <= 0:
            raise ValueError("radius and speed of sound must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        directions = torch.randn((modes, 3), generator=generator)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("directions", directions)
        self.modes = int(modes)
        self.radius_m = float(radius_m)
        self.speed_of_sound_m_s = float(speed_of_sound_m_s)
        self.seed = int(seed)

    @property
    def latent_dim(self) -> int:
        return 2 * self.modes

    def sample_latent(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            (batch_size, self.latent_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def forward(
        self,
        unit_coordinates: torch.Tensor,
        latent: torch.Tensor,
        frequency_hz: torch.Tensor,
    ) -> torch.Tensor:
        if unit_coordinates.ndim != 3 or unit_coordinates.shape[-1] != 3:
            raise ValueError("unit_coordinates must have shape [batch, point, 3]")
        if latent.shape != (unit_coordinates.shape[0], self.latent_dim):
            raise ValueError(
                f"latent must have shape [{unit_coordinates.shape[0]}, {self.latent_dim}]"
            )
        frequencies = torch.as_tensor(
            frequency_hz, device=unit_coordinates.device, dtype=unit_coordinates.dtype
        ).reshape(-1)
        if len(frequencies) != unit_coordinates.shape[0]:
            raise ValueError("frequency_hz must contain one value per batch item")
        wave_number_radius = (
            2.0 * math.pi * frequencies * self.radius_m / self.speed_of_sound_m_s
        )
        direction_projection = torch.einsum(
            "bnc,kc->bnk", unit_coordinates, self.directions.to(unit_coordinates)
        )
        phase = wave_number_radius[:, None, None] * direction_projection
        cosine_coefficients, sine_coefficients = latent.chunk(2, dim=-1)
        values = (
            torch.cos(phase) * cosine_coefficients[:, None, :]
            + torch.sin(phase) * sine_coefficients[:, None, :]
        ).sum(dim=-1)
        return values / math.sqrt(self.modes)

    def config_dict(self) -> dict[str, int | float]:
        return {
            "modes": self.modes,
            "radius_m": self.radius_m,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "seed": self.seed,
        }
