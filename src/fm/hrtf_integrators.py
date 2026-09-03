"""Training-time unrolling and inference-time integration for HRTF fields."""

from __future__ import annotations

import torch

from src.models.hrtf_flow import ConditionalHRTFFieldFlow


def euler_unroll_to_endpoint(
    model: ConditionalHRTFFieldFlow,
    initial_state: torch.Tensor,
    query_unit_xyz: torch.Tensor,
    context: torch.Tensor,
    *,
    steps: int,
    start_time: float = 0.0,
) -> torch.Tensor:
    """Differentiably unroll a field to t=1 for clean-endpoint losses."""

    if steps < 1 or not 0.0 <= start_time < 1.0:
        raise ValueError("steps must be positive and start_time must be in [0, 1)")
    state = initial_state
    batch_size = state.shape[0]
    dt = (1.0 - start_time) / steps
    for index in range(steps):
        time_value = start_time + index * dt
        time = torch.full(
            (batch_size,), time_value, device=state.device, dtype=state.dtype
        )
        velocity = model.velocity_with_context(state, time, query_unit_xyz, context)
        state = state + dt * velocity
    return state

def _set_observed_path(
    state: torch.Tensor,
    observed_indices: torch.Tensor,
    observed_source: torch.Tensor,
    observed_target: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    if observed_indices.ndim == 1:
        observed_indices = observed_indices.unsqueeze(0).expand(state.shape[0], -1)
    values = (1.0 - time[:, None]) * observed_source + time[:, None] * observed_target
    return state.scatter(1, observed_indices.long(), values)


@torch.no_grad()
def heun_integrate_hrtf(
    model: ConditionalHRTFFieldFlow,
    initial_state: torch.Tensor,
    query_unit_xyz: torch.Tensor,
    context: torch.Tensor,
    *,
    steps: int,
    observed_indices: torch.Tensor | None = None,
    observed_source: torch.Tensor | None = None,
    observed_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heun integration with an optional exact measured-data interpolation path."""

    if steps < 1:
        raise ValueError("steps must be positive")
    clamp_arguments = (observed_indices, observed_source, observed_target)
    clamp_enabled = all(value is not None for value in clamp_arguments)
    if not clamp_enabled and any(value is not None for value in clamp_arguments):
        raise ValueError("all observed-path arguments must be supplied together")

    state = initial_state
    batch_size = state.shape[0]
    grid = torch.linspace(0.0, 1.0, steps + 1, device=state.device, dtype=state.dtype)
    if clamp_enabled:
        zero = torch.zeros((batch_size,), device=state.device, dtype=state.dtype)
        state = _set_observed_path(
            state, observed_indices, observed_source, observed_target, zero
        )
    for index in range(steps):
        t0 = torch.full(
            (batch_size,), grid[index].item(), device=state.device, dtype=state.dtype
        )
        t1 = torch.full(
            (batch_size,), grid[index + 1].item(), device=state.device, dtype=state.dtype
        )
        dt = grid[index + 1] - grid[index]
        velocity_0 = model.velocity_with_context(state, t0, query_unit_xyz, context)
        predicted = state + dt * velocity_0
        if clamp_enabled:
            predicted = _set_observed_path(
                predicted, observed_indices, observed_source, observed_target, t1
            )
        velocity_1 = model.velocity_with_context(predicted, t1, query_unit_xyz, context)
        state = state + 0.5 * dt * (velocity_0 + velocity_1)
        if clamp_enabled:
            state = _set_observed_path(
                state, observed_indices, observed_source, observed_target, t1
            )
    return state
