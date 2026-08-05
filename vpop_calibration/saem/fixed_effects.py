import torch
from typing import Callable
import numpy as np


def compute_fixed_effects_gradient(
    loss_fn: Callable, psi: torch.Tensor, eps_base
) -> tuple[torch.Tensor, torch.Tensor]:
    loss = loss_fn(psi)
    eps_scaled = eps_base * torch.clamp(psi.abs(), min=1.0)
    perturbation_matrix = torch.diag(eps_scaled)
    perturbed_psi = psi.unsqueeze(0) + perturbation_matrix
    loss_eps = loss_fn(perturbed_psi)
    grad = (loss_eps - loss) / eps_scaled
    return grad, loss


def optimize_fixed_effects(
    loss_fn: Callable,
    psi0: torch.Tensor,
    lr: float,
    nb_iter: int,
    eps_grad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_effects = psi0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([fixed_effects], lr=lr)

    loss_output = torch.Tensor([np.nan])
    for _ in range(nb_iter):
        optimizer.zero_grad()
        grad, loss = compute_fixed_effects_gradient(
            loss_fn=loss_fn, psi=fixed_effects.detach(), eps_base=eps_grad
        )
        fixed_effects.grad = grad
        optimizer.step()
        loss_output = loss
    return fixed_effects.detach(), loss_output.detach()
