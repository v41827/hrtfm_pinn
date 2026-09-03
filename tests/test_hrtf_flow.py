from __future__ import annotations

import unittest

import torch

from src.fm.functional_prior import HelmholtzGaussianField
from src.fm.hrtf_integrators import (
    euler_unroll_independent_to_endpoint,
    euler_unroll_to_endpoint,
    heun_integrate_hrtf,
    heun_integrate_independent_hrtf,
)
from src.models.hrtf_flow import (
    ConditionalHRTFFieldFlow,
    HRTFFlowConfig,
    IndependentHRTFFieldFlow,
    IndependentHRTFFlowConfig,
)
from src.physics.helmholtz import helmholtz_loss, physical_helmholtz_loss


class HRTFFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        config = HRTFFlowConfig(
            width=16,
            depth=2,
            context_dim=16,
            observation_width=12,
            coordinate_bands=1,
            time_dim=8,
        )
        self.model = ConditionalHRTFFieldFlow(config)
        self.prior = HelmholtzGaussianField(modes=4)

    def test_physics_endpoint_supports_parameter_gradients(self) -> None:
        observed_coordinates = torch.randn(2, 6, 3)
        observed_values = torch.randn(2, 6)
        frequencies = torch.tensor([2067.1875, 4134.375])
        component = torch.tensor([0, 1])
        hemisphere = torch.tensor([0, 1])
        context = self.model.encode_condition(
            observed_coordinates,
            observed_values,
            frequencies,
            component,
            hemisphere,
        )
        query_coordinates = torch.randn(2, 5, 3, requires_grad=True)
        latent = self.prior.sample_latent(2, device="cpu")
        source = self.prior(query_coordinates, latent, frequencies)
        endpoint = euler_unroll_to_endpoint(
            self.model, source, query_coordinates, context, steps=2
        )
        loss, _ = helmholtz_loss(
            endpoint, query_coordinates, frequencies, radius_m=0.09
        )
        loss.backward()

        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_heun_path_ends_exactly_at_observations(self) -> None:
        query_coordinates = torch.randn(1, 8, 3)
        observed_indices = torch.tensor([1, 4, 7])
        observed_coordinates = query_coordinates[:, observed_indices]
        observed_values = torch.tensor([[0.2, -0.4, 0.8]])
        frequency = torch.tensor([6201.5625])
        context = self.model.encode_condition(
            observed_coordinates,
            observed_values,
            frequency,
            torch.tensor([0]),
            torch.tensor([0]),
        )
        latent = self.prior.sample_latent(1, device="cpu")
        source = self.prior(query_coordinates, latent, frequency)
        result = heun_integrate_hrtf(
            self.model,
            source,
            query_coordinates,
            context,
            steps=3,
            observed_indices=observed_indices,
            observed_source=source[:, observed_indices],
            observed_target=observed_values,
        )

        torch.testing.assert_close(result[:, observed_indices], observed_values)

    def test_independent_flow_physics_gradients_and_hard_path(self) -> None:
        model = IndependentHRTFFieldFlow(
            IndependentHRTFFlowConfig(width=6, depth=3)
        )
        coordinates = (0.09 * torch.randn(1, 8, 3)).requires_grad_(True)
        frequency = torch.tensor([4134.375])
        latent = self.prior.sample_latent(1, device="cpu")
        source = self.prior(coordinates / 0.09, latent, frequency)
        endpoint = euler_unroll_independent_to_endpoint(
            model, source, coordinates, steps=2
        )
        loss, _ = physical_helmholtz_loss(endpoint, coordinates, frequency)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

        observed_indices = torch.tensor([0, 3, 6])
        observed_values = torch.tensor([[0.3, -0.2, 0.7]])
        result = heun_integrate_independent_hrtf(
            model,
            source.detach(),
            coordinates.detach(),
            steps=3,
            observed_indices=observed_indices,
            observed_source=source.detach()[:, observed_indices],
            observed_target=observed_values,
        )
        torch.testing.assert_close(result[:, observed_indices], observed_values)


if __name__ == "__main__":
    unittest.main()
