import torch
from typing import Callable


def compute_fixed_effects_gradient(
    loss_fn: Callable, psi: torch.Tensor, eps_base
) -> tuple[torch.Tensor, torch.Tensor]:
    nb_params = psi.shape[0]
    loss = loss_fn(psi)
    eps_scaled = eps_base * torch.clamp(psi.abs(), min=1.0)
    perturbation_matrix = torch.diag(eps_scaled)
    perturbed_psi = psi.unsqueeze(0) + perturbation_matrix
    loss_eps = loss_fn(perturbed_psi)
    assert loss_eps.shape == (nb_params,), (
        f"Unexpected perturbed loss shape in gradient calculation: {loss_eps.shape} ."
    )
    grad = (loss_eps - loss) / eps_scaled
    return grad, loss


def take_fixed_effects_step(
    loss_fn: Callable,
    psi0: torch.Tensor,
    lr: float,
    eps_grad: float,
    step_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take one scaled finite-difference gradient step and return its baseline loss.

    ``lr`` includes the outer stochastic-approximation rate. ``step_scale``
    is a fixed positive diagonal preconditioner in the coordinates of ``psi0``.
    """
    assert psi0.dim() == 1
    if step_scale is None:
        step_scale = torch.ones_like(psi0)
    assert step_scale.shape == psi0.shape

    with torch.no_grad():
        grad, loss = compute_fixed_effects_gradient(
            loss_fn=loss_fn, psi=psi0.detach(), eps_base=eps_grad
        )
        fixed_effects = psi0 - lr * step_scale * grad
    return fixed_effects.detach(), loss.detach()
