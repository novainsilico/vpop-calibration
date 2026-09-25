import torch
from typing import Callable, NamedTuple


class FixedEffectsEvaluation(NamedTuple):
    """Fixed-effects objective evaluated at one or several candidate values.

    Leading dimensions index the candidates. ``patient_loss`` is the negative
    observation log-likelihood of each patient, ``survival_loss`` its survival part.
    ``precision`` is the inverse residual variance of each observation row, and is 0
    on rows which are not normally distributed.
    """

    patient_loss: torch.Tensor  # (..., nb_patients)
    survival_loss: torch.Tensor  # (..., nb_patients)
    predictions: torch.Tensor  # (..., nb_obs)
    precision: torch.Tensor  # (..., nb_obs)


def compute_fixed_effects_gradient(
    loss_fn: Callable, psi: torch.Tensor, eps_base
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward finite-difference gradient of ``loss_fn`` at ``psi``.

    ``loss_fn`` returns a tensor, or a tuple of tensors, with a leading candidate
    dimension. Trailing dimensions (e.g. one loss per patient) are kept: each
    derivative has shape ``(nb_params, *output_shape)``.
    """
    nb_params = psi.shape[0]
    loss = loss_fn(psi)
    eps_scaled = eps_base * torch.clamp(psi.abs(), min=1.0)
    perturbation_matrix = torch.diag(eps_scaled)
    perturbed_psi = psi.unsqueeze(0) + perturbation_matrix
    loss_eps = loss_fn(perturbed_psi)

    def difference(baseline: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
        assert perturbed.shape[0] == nb_params, (
            f"Unexpected perturbed loss shape in gradient calculation: {perturbed.shape} ."
        )
        eps_expanded = eps_scaled.view(nb_params, *[1] * (perturbed.dim() - 1))
        return (perturbed - baseline) / eps_expanded

    if isinstance(loss, tuple):
        grad = type(loss)(*map(difference, loss, loss_eps))
    else:
        grad = difference(loss, loss_eps)
    return grad, loss


def solve_damped_fisher(
    fisher: torch.Tensor, grad: torch.Tensor, damping: float
) -> torch.Tensor:
    """Solve ``(F + damping * diag(F)) x = grad``.

    Marquardt damping keeps the step invariant to rescaling each coordinate. A
    null diagonal entry implies a null gradient entry, so a relative floor suffices.
    """
    diagonal = fisher.diagonal()
    scale = diagonal.max()
    if scale == 0:
        # The likelihood does not depend on the fixed effects
        return torch.zeros_like(grad)
    floor = torch.finfo(fisher.dtype).eps * scale
    damped = fisher + torch.diag(damping * diagonal + floor)
    return torch.linalg.solve(damped, grad)


def estimate_fisher(
    derivatives: FixedEffectsEvaluation, baseline: FixedEffectsEvaluation
) -> torch.Tensor:
    """Fisher matrix of the mean per-patient loss.

    Normally distributed observations contribute their Gauss-Newton (expected Fisher)
    term, with the residual variance held at its baseline value. Survival data
    contribute the empirical Fisher of the per-patient survival log-likelihood.
    """
    nb_patients = baseline.patient_loss.shape[-1]
    jacobian = derivatives.predictions  # (nb_params, nb_obs)
    precision = baseline.precision.reshape(-1)  # (nb_obs,)
    gauss_newton = (jacobian * precision) @ jacobian.T
    survival_grads = derivatives.survival_loss  # (nb_params, nb_patients)
    empirical = survival_grads @ survival_grads.T
    return (gauss_newton + empirical) / nb_patients


def take_fixed_effects_step(
    loss_fn: Callable[[torch.Tensor], FixedEffectsEvaluation],
    psi0: torch.Tensor,
    lr: float,
    eps_grad: float,
    step_scale: torch.Tensor | None = None,
    fisher: torch.Tensor | None = None,
    fisher_rate: float | None = None,
    fisher_damping: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Take one finite-difference step on the mean of per-patient losses.

    Return the new fixed effects, the baseline mean loss and the Fisher estimate.
    ``lr`` includes the outer stochastic-approximation rate. ``step_scale`` is a
    fixed positive diagonal preconditioner in the coordinates of ``psi0``.

    If ``fisher_rate`` is set, the gradient is also preconditioned by the Fisher
    matrix (see ``estimate_fisher``), averaged into the previous estimate ``fisher``
    (None on the first step) with rate ``fisher_rate``. Otherwise the returned
    estimate is None.
    """
    assert psi0.dim() == 1
    if step_scale is None:
        step_scale = torch.ones_like(psi0)
    assert step_scale.shape == psi0.shape

    with torch.no_grad():
        derivatives, baseline = compute_fixed_effects_gradient(
            loss_fn=loss_fn, psi=psi0.detach(), eps_base=eps_grad
        )
        grad = derivatives.patient_loss.mean(dim=-1)
        if fisher_rate is None:
            direction = grad
            fisher = None
        else:
            batch_fisher = estimate_fisher(derivatives, baseline)
            if fisher is None:
                fisher = batch_fisher
            else:
                fisher = fisher + fisher_rate * (batch_fisher - fisher)
            direction = solve_damped_fisher(fisher, grad, fisher_damping)
        fixed_effects = psi0 - lr * step_scale * direction
    return (
        fixed_effects.detach(),
        baseline.patient_loss.mean(dim=-1).detach(),
        None if fisher is None else fisher.detach(),
    )
