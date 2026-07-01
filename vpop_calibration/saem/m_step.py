import torch
from typing import NamedTuple


from vpop_calibration.saem.utils import stochastic_approximation, clamp_eigen_values
from vpop_calibration.config import device


class MStepProposal(NamedTuple):
    beta: torch.Tensor
    omega: torch.Tensor


class MStepState:
    def __init__(
        self,
        design_matrix: torch.Tensor,
        nb_chains: int,
        nb_patients: int,
        nb_pdu: int,
        init_gaussian_params: torch.Tensor,
    ):
        # Gather the required properties from the statistical model
        self.X = design_matrix
        self.X_expanded = self.X.expand(nb_chains, -1, -1, -1)
        self.gram_matrix = (
            torch.matmul(self.X.transpose(-1, -2), self.X).sum(dim=0).to(device)
        )
        self.nb_chains = nb_chains
        self.nb_patients = nb_patients
        self.nb_pdu = nb_pdu
        # Initiate the sufficient statistics
        assert init_gaussian_params.shape == (
            self.nb_chains,
            self.nb_patients,
            self.nb_pdu,
        )

        self.cross_product = self.compute_cross_product(init_gaussian_params)
        self.outer_product, _ = self.compute_outer_product(init_gaussian_params)

    def compute_cross_product(self, gaussian_params: torch.Tensor) -> torch.Tensor:
        prod = (
            (self.X_expanded.transpose(-1, -2) @ gaussian_params.unsqueeze(-1))
            .sum(dim=1)
            .mean(dim=0)
        )
        return prod

    def compute_outer_product(
        self, gaussian_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the new outer product, given the gaussian parameters and the new linear estimator (beta)."""
        # The outer product is computed on the centered residuals resulting from the new beta estimation. This can be expressed as
        #     \frac{1}{nb_chains} \sum_{k=0}^{nb_chains} \sum_{i=0}^{nb_patients} (\Psi - \mu)^T (\Psi - \mu),
        #     with \mu = X * \beta
        #     and \beta is the solution to G * beta = cross_product
        # Todo: make this tooltip part of the docstring

        # Ensure the cross_product was updated first
        new_beta = torch.linalg.solve(self.gram_matrix, self.cross_product).to(device)
        new_mu = torch.matmul(self.X, new_beta.unsqueeze(0)).squeeze(-1).to(device)
        residuals = gaussian_params - new_mu.unsqueeze(0)
        residuals.unsqueeze_(-1)
        outer_prod_centered = (
            torch.matmul(residuals, residuals.transpose(-1, -2))
            .sum(dim=1)
            .mean(dim=0)
            .to(device)
        )
        return outer_prod_centered, new_beta

    def update(
        self, new_gaussian_params: torch.Tensor, learning_rate: float
    ) -> MStepProposal:
        """Given new values for the gaussian parameters, update the sufficient statistics and propose new estimates for beta and Omega, with stochastic approximation."""
        assert new_gaussian_params.shape == (
            self.nb_chains,
            self.nb_patients,
            self.nb_pdu,
        )

        target_cross_product = self.compute_cross_product(new_gaussian_params)
        self.cross_product = stochastic_approximation(
            previous=self.cross_product,
            new=target_cross_product,
            learning_rate=learning_rate,
        )
        target_outer_product, target_beta = self.compute_outer_product(
            new_gaussian_params
        )
        self.outer_product = stochastic_approximation(
            previous=self.outer_product,
            new=target_outer_product,
            learning_rate=learning_rate,
        )
        # Propose a new value for Omega based on the outerproduct, scaling by the number of patients
        target_omega = clamp_eigen_values(self.outer_product / self.nb_patients)

        return MStepProposal(beta=target_beta, omega=target_omega)
