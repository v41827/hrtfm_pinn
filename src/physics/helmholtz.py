"""Automatic-differentiation Helmholtz residuals for HRTF neural fields."""

from __future__ import annotations

import math

import torch


def laplacian(
    field: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Return the pointwise Cartesian Laplacian of a scalar neural field.

    The model used with this function is pointwise in ``coordinates`` once its
    global conditioning context is fixed.  Under that condition, differentiating
    ``field.sum()`` produces the correct per-point derivatives without forming a
    full Jacobian.
    """

    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [batch, point, 3]")
    if field.shape not in (coordinates.shape[:-1], coordinates.shape[:-1] + (1,)):
        raise ValueError("field must have shape [batch, point] or [batch, point, 1]")
    scalar_field = field.squeeze(-1) if field.ndim == 3 else field
    if not coordinates.requires_grad:
        raise ValueError("coordinates.requires_grad must be True")

    first = torch.autograd.grad(
        scalar_field.sum(),
        coordinates,
        create_graph=True,
        retain_graph=True,
    )[0]
    result = torch.zeros_like(scalar_field)
    for axis in range(3):
        second = torch.autograd.grad(
            first[..., axis].sum(),
            coordinates,
            create_graph=create_graph,
            retain_graph=True,
        )[0][..., axis]
        result = result + second
    return result


def normalized_helmholtz_residual(
    field: torch.Tensor,
    unit_coordinates: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    radius_m: float,
    speed_of_sound_m_s: float = 343.0,
    create_graph: bool = True,
) -> torch.Tensor:
    r"""Compute Ma et al.'s normalized homogeneous Helmholtz residual.

    The network consumes unit-sphere coordinates ``u = x / radius_m`` for
    numerical conditioning.  Since ``nabla_x^2 = nabla_u^2 / radius_m^2``,

    .. math::
        R(H) = \frac{\nabla_x^2 H}{k^2} + H
             = \frac{\nabla_u^2 H}{(k r)^2} + H.
    """

    if radius_m <= 0 or speed_of_sound_m_s <= 0:
        raise ValueError("radius and speed of sound must be positive")
    scalar_field = field.squeeze(-1) if field.ndim == 3 else field
    frequencies = torch.as_tensor(
        frequency_hz, device=scalar_field.device, dtype=scalar_field.dtype
    ).reshape(-1)
    if len(frequencies) != scalar_field.shape[0]:
        raise ValueError("frequency_hz must contain one value per batch item")
    wave_number_radius = (
        2.0 * math.pi * frequencies * float(radius_m) / float(speed_of_sound_m_s)
    )
    lap = laplacian(scalar_field, unit_coordinates, create_graph=create_graph)
    return lap / wave_number_radius[:, None].square() + scalar_field


def helmholtz_loss(
    field: torch.Tensor,
    unit_coordinates: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    radius_m: float,
    speed_of_sound_m_s: float = 343.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-squared Helmholtz loss and the pointwise residual."""

    residual = normalized_helmholtz_residual(
        field,
        unit_coordinates,
        frequency_hz,
        radius_m=radius_m,
        speed_of_sound_m_s=speed_of_sound_m_s,
        create_graph=True,
    )
    return residual.square().mean(), residual


def physical_helmholtz_loss(
    field: torch.Tensor,
    xyz_m: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    speed_of_sound_m_s: float = 343.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Helmholtz loss when the model directly consumes coordinates in metres.

    This is algebraically the same normalized residual used by Ma et al.,
    ``laplacian_x(H) / k^2 + H``, without the unit-sphere change of variables.
    """

    if speed_of_sound_m_s <= 0:
        raise ValueError("speed of sound must be positive")
    scalar_field = field.squeeze(-1) if field.ndim == 3 else field
    frequencies = torch.as_tensor(
        frequency_hz, device=scalar_field.device, dtype=scalar_field.dtype
    ).reshape(-1)
    if len(frequencies) != scalar_field.shape[0]:
        raise ValueError("frequency_hz must contain one value per batch item")
    wave_number = 2.0 * math.pi * frequencies / float(speed_of_sound_m_s)
    residual = (
        laplacian(scalar_field, xyz_m, create_graph=True)
        / wave_number[:, None].square()
        + scalar_field
    )
    return residual.square().mean(), residual
