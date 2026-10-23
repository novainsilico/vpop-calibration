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

    Returns:

        * the gradient of the loss function with shape (F, P)
        * the loss function at baseline with shape (P, )
    """
    # torch.diag(eps_scaled) assumes psi is 1D to build the perturbation matrix
    assert psi.dim() == 1, f"{psi.dim()=}"
    num_fixed_effects = psi.shape[0]  # F
    eps_scaled = eps_base * torch.clamp(psi.abs(), min=1.0)
    candidates = torch.cat(
        [psi.unsqueeze(0), psi.unsqueeze(0) + torch.diag(eps_scaled)], dim=0
    )  # (1 + F, F)
    loss_all = loss_fn(candidates).patient_loss  # (1 + F, P)
    assert loss_all.shape[0] == 1 + num_fixed_effects
    loss_baseline, loss_eps = loss_all[0], loss_all[1:]
    grad = (loss_eps - loss_baseline) / eps_scaled.unsqueeze(-1)
    return grad, loss_baseline


def minimize_fixed_effects_loss(
    loss_fn: Callable,
    psi0: torch.Tensor,  # (F, )
    lr: float,
    nb_iter: int,
    eps_grad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert psi0.dim() == 1
    with torch.no_grad():
        fixed_effects = psi0.clone()
        optimizer = torch.optim.Adam([fixed_effects], lr=lr)
        if nb_iter <= 0:
            return (
                fixed_effects,
                torch.tensor([np.nan], device=device, dtype=default_dtype),
            )
        for _ in range(nb_iter):
            grad, baseline_loss = fixed_effects_gradient(
                loss_fn=loss_fn, psi=fixed_effects, eps_base=eps_grad
            )
            per_patient_gradient_mean = grad.mean(dim=-1)
            fixed_effects.grad = per_patient_gradient_mean
            optimizer.step()
            per_patient_loss_mean = baseline_loss.mean(dim=-1)
        return fixed_effects, per_patient_loss_mean
