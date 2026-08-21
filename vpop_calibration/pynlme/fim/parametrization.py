from typing import NamedTuple

import torch

from vpop_calibration.config import default_dtype, device
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import ResidualErrorEstimates

"""Data handling: 
 
The FIM is expressed in a single flat vector concatenating, in this order:
 
1. the population fixed effects ``beta``
2. the lower triangle of the inter-individual covariance matrix ``omega``
3. the log model-intrinsic parameters ``log_mi``
4. the active residual error parameters ``sigma_add`` then ``sigma_prop``
 
Defines how to name, flatten and unflatten the parameters, and where each block lives in the vector.
"""


class PopulationParameters(NamedTuple):
    """Population parameters rebuilt from a flat vector, keeping the autograd graph."""

    beta: torch.Tensor
    omega: torch.Tensor
    log_mi: torch.Tensor
    residual_var: ResidualErrorEstimates


class FimParametrization:
    """Layout of the population parameters inside the flat vector used by the FIM."""

    def __init__(self, model: StatisticalModel) -> None:
        self.model = model
        self.omega_indices = torch.tril_indices(
            model.nb_pdu, model.nb_pdu, device=device
        )
        self.sigma_mask = torch.cat(
            (
                model.residual_var.additive_output,
                model.residual_var.proportional_output,
            )
        )
        self.names = self._build_names()

    # --- Layout
    @property
    def nb_params(self) -> int:
        return len(self.names)

    @property
    def nb_omega(self) -> int:
        return self.omega_indices.shape[1]

    @property
    def mi_location(self) -> slice:
        """Location of the model-intrinsic parameters in the flat vector."""
        start = self.model.nb_betas + self.nb_omega
        return slice(start, start + self.model.nb_mi)

    def _build_names(self) -> list[str]:
        model = self.model
        names = list(model.beta_names)
        names += [
            f"omega_{model.pdu_names[i]}_{model.pdu_names[j]}"
            for i, j in zip(*self.omega_indices.tolist())
        ]
        names += list(model.mi_names)
        sigma_names = [
            f"{component}_{output}"
            for component in ("sigma_add", "sigma_prop")
            for output in model.output_names
        ]
        names += [
            name
            for name, active in zip(sigma_names, self.sigma_mask.tolist())
            if active
        ]
        return names

    def flatten(self) -> torch.Tensor:
        """Current population parameters, as a flat vector."""
        model = self.model
        blocks = [
            model.population_betas,
            model.omega_pop[self.omega_indices[0], self.omega_indices[1]],
            model.log_mi,
            torch.cat((model.residual_var.sigma_add, model.residual_var.sigma_prop))[
                self.sigma_mask
            ],
        ]
        return torch.cat(
            [block.detach().flatten().to(default_dtype) for block in blocks]
        )

    def unflatten(self, flat: torch.Tensor) -> PopulationParameters:
        """Rebuild the model parameters from the flat vector, keeping the autograd graph."""
        model = self.model
        cursor = 0

        beta = flat[cursor : cursor + model.nb_betas]
        cursor += model.nb_betas

        lower = torch.zeros(
            (model.nb_pdu, model.nb_pdu), device=device, dtype=flat.dtype
        ).index_put(
            (self.omega_indices[0], self.omega_indices[1]),
            flat[cursor : cursor + self.nb_omega],
        )
        omega = lower + lower.tril(-1).transpose(-1, -2)
        cursor += self.nb_omega

        if model.nb_mi == 0:
            log_mi = torch.empty(0, device=device, dtype=flat.dtype)
        else:
            log_mi = flat[cursor : cursor + model.nb_mi]
        cursor += model.nb_mi

        idx = self.sigma_mask.nonzero(as_tuple=True)[0]
        full_sigma = torch.zeros(
            2 * model.nb_outputs, device=device, dtype=flat.dtype
        ).index_put((idx,), flat[cursor:])
        sigma_add, sigma_prop = full_sigma.chunk(2)

        return PopulationParameters(
            beta=beta,
            omega=omega,
            log_mi=log_mi,
            residual_var=model.residual_var._replace(
                sigma_add=sigma_add, sigma_prop=sigma_prop
            ),
        )
