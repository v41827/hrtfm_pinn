"""Flow-matching objectives for scalar HRTF neural fields."""

from __future__ import annotations

import torch


def linear_field_path(
    source: torch.Tensor, target: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    """Interpolate source and target fields at one time per batch item."""

    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have matching [batch, point] shapes")
    if time.shape != (source.shape[0],):
        raise ValueError("time must have shape [batch]")
    return (1.0 - time[:, None]) * source + time[:, None] * target


def constant_field_velocity(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("source and target shapes must match")
    return target - source


def clean_endpoint_from_velocity(
    state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    r"""One-step clean endpoint estimate ``x_t + (1-t) v_theta``."""

    if state.shape != velocity.shape or state.ndim != 2:
        raise ValueError("state and velocity must have matching [batch, point] shapes")
    if time.shape != (state.shape[0],):
        raise ValueError("time must have shape [batch]")
    return state + (1.0 - time[:, None]) * velocity
