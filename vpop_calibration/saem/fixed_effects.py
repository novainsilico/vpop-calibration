import torch
from typing import Callable, NamedTuple
import numpy as np

from vpop_calibration.config import device, default_dtype


class FixedEffectsEvaluation(NamedTuple):
    """Fixed-effects objective evaluated at one or several candidate values.

    ``patient_loss`` is the negative observation log-likelihood of each patient.
    ``predictions`` are model predictions for each patient
    """

    patient_loss: torch.Tensor  # (..., nb_patients)
    predictions: torch.Tensor  # (..., nb_obs)


def fixed_effects_gradient(
    loss_fn: Callable, psi: torch.Tensor, eps_base
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward finite-difference gradient of ``loss_fn`` at ``psi``.

    ``loss_fn`` returns the gradient and the loss function at baseline,
    both with a leading candidate dimension. Trailing dimensions (e.g. one loss per patient) are kept: each
    derivative has shape ``(nb_params, *output_shape)``.
    """
    nb_params = psi.shape[0]
    loss_baseline = loss_fn(psi).patient_loss
    eps_scaled = eps_base * torch.clamp(psi.abs(), min=1.0)
    perturbation_matrix = torch.diag(eps_scaled)
    perturbed_psi = psi.unsqueeze(0) + perturbation_matrix
    loss_eps = loss_fn(perturbed_psi).patient_loss
    assert loss_eps.shape[0] == (
        nb_params
    ), f"Unexpected perturbed loss shape in gradient calculation: {loss_eps.shape} ."
    # Expand the epsilons to have the same shape as the loss tensors
    eps_expanded = eps_scaled.view(nb_params, *[1] * (loss_eps.dim() - 1))
    grad = (loss_eps - loss_baseline) / eps_expanded
    return grad, loss_baseline


def minimize_fixed_effects_loss(
    loss_fn: Callable,
    psi0: torch.Tensor,
    lr: float,
    nb_iter: int,
    eps_grad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert psi0.dim() == 1
    fixed_effects = psi0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([fixed_effects], lr=lr)
    if nb_iter <= 0:
        return (
            fixed_effects,
            torch.tensor([np.nan], device=device, dtype=default_dtype),
        )
    for _ in range(nb_iter):
        optimizer.zero_grad()
        grad, baseline_loss = fixed_effects_gradient(
            loss_fn=loss_fn, psi=fixed_effects.detach(), eps_base=eps_grad
        )
        per_patient_gradient_mean = grad.mean(dim=-1)
        fixed_effects.grad = per_patient_gradient_mean
        optimizer.step()
        per_patient_loss_mean = baseline_loss.mean(dim=-1)
    return fixed_effects.detach(), per_patient_loss_mean.detach()
