from typing import NamedTuple
import torch

from vpop_calibration.config import default_dtype, device
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import ResidualErrorEstimates


class PopulationParameters(NamedTuple):
    """Population parameters rebuilt from a flat vector, keeping the autograd graph."""

    beta: torch.Tensor
    omega: torch.Tensor
    log_mi: torch.Tensor
    residual_var: ResidualErrorEstimates


def get_sigma_mask(model: StatisticalModel) -> torch.Tensor:
    """Mask of active residual error parameters."""
    return torch.cat(
        (model.residual_var.additive_output, model.residual_var.proportional_output)
    )


def get_parameter_names(model: StatisticalModel) -> list[str]:
    """Generate the names of the parameters present in the FIM flat vector."""
    names = list(model.beta_names)
    names += [
        f"omega_{model.pdu_names[i]}_{model.pdu_names[j]}"
        for i, j in zip(*model.omega_indices.tolist())
    ]
    names += list(model.mi_names)
    sigma_names = [
        f"{component}_{output}"
        for component in ("sigma_add", "sigma_prop")
        for output in model.output_names
    ]
    sigma_mask = get_sigma_mask(model)
    names += [name for name, active in zip(sigma_names, sigma_mask.tolist()) if active]
    return names


def flatten(model: StatisticalModel) -> torch.Tensor:
    """Current population parameters, as a flat vector."""
    sigma_mask = get_sigma_mask(model)
    blocks = [
        model.population_betas,
        model.omega_pop[model.omega_indices[0], model.omega_indices[1]],
        model.log_mi,
        torch.cat((model.residual_var.sigma_add, model.residual_var.sigma_prop))[
            sigma_mask
        ],
    ]
    return torch.cat([block.detach().flatten().to(default_dtype) for block in blocks])


def unflatten(flat: torch.Tensor, model: StatisticalModel) -> PopulationParameters:
    """Rebuild the model parameters from the flat vector, keeping the autograd graph."""
    cursor = 0

    beta = flat[cursor : cursor + model.nb_betas]
    cursor += model.nb_betas

    nb_omega = model.omega_indices.shape[1]
    lower = torch.zeros(
        (model.nb_pdu, model.nb_pdu), device=device, dtype=flat.dtype
    ).index_put(
        (model.omega_indices[0], model.omega_indices[1]),
        flat[cursor : cursor + nb_omega],
    )
    omega = lower + lower.tril(-1).transpose(-1, -2)
    cursor += nb_omega

    if model.nb_mi == 0:
        log_mi = torch.empty(0, device=device, dtype=flat.dtype)
    else:
        log_mi = flat[cursor : cursor + model.nb_mi]
    cursor += model.nb_mi

    sigma_mask = get_sigma_mask(model)
    idx = sigma_mask.nonzero(as_tuple=True)[0]
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
