from __future__ import annotations

import math
import unittest

import torch

from src.fm.functional_prior import HelmholtzGaussianField
from src.physics.helmholtz import normalized_helmholtz_residual, physical_helmholtz_loss


class HelmholtzResidualTest(unittest.TestCase):
    def test_plane_wave_is_zero_in_unit_coordinates(self) -> None:
        torch.manual_seed(3)
        coordinates = torch.randn(2, 24, 3, dtype=torch.float64, requires_grad=True)
        frequencies = torch.tensor([2067.1875, 14470.3125], dtype=torch.float64)
        radius_m = 0.09
        speed = 343.0
        kr = 2.0 * math.pi * frequencies * radius_m / speed
        plane_wave = torch.sin(kr[:, None] * coordinates[..., 0])

        residual = normalized_helmholtz_residual(
            plane_wave,
            coordinates,
            frequencies,
            radius_m=radius_m,
            speed_of_sound_m_s=speed,
            create_graph=False,
        )

        self.assertLess(float(residual.abs().max()), 1e-12)

    def test_gaussian_source_field_is_helmholtz_valid(self) -> None:
        prior = HelmholtzGaussianField(modes=8, radius_m=0.09, seed=11)
        coordinates = torch.randn(2, 16, 3, requires_grad=True)
        frequencies = torch.tensor([2067.1875, 14470.3125])
        latent = prior.sample_latent(2, device="cpu")
        field = prior(coordinates, latent, frequencies)

        residual = normalized_helmholtz_residual(
            field,
            coordinates,
            frequencies,
            radius_m=0.09,
            create_graph=False,
        )

        self.assertLess(float(residual.abs().max()), 2e-6)

    def test_plane_wave_is_zero_in_physical_coordinates(self) -> None:
        coordinates = torch.randn(2, 12, 3, dtype=torch.float64, requires_grad=True)
        frequencies = torch.tensor([2067.1875, 14470.3125], dtype=torch.float64)
        speed = 343.0
        wave_number = 2.0 * math.pi * frequencies / speed
        plane_wave = torch.cos(wave_number[:, None] * coordinates[..., 1])

        loss, residual = physical_helmholtz_loss(
            plane_wave,
            coordinates,
            frequencies,
            speed_of_sound_m_s=speed,
        )

        self.assertLess(float(loss), 1e-24)
        self.assertLess(float(residual.abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
