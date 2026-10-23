import numpy as np
import torch
from torch.testing import assert_close
from vpop_calibration.saem.fixed_effects import (
    FixedEffectsEvaluation,
    fixed_effects_gradient,
    minimize_fixed_effects_loss,
)


def test_fixed_effects_gradient():
    """
    Here we test our implementation of the gradient descent using a quadratic loss function for which we know
    the solution in closed form.
    All P=5 patients have different fixed effects targets. Because the fixed effects
    optimization procedure uses the gradient averaged over patients for the gradient descent,
    it converges to the targets mean over patients.
    Here that's $\\star \\psi = [1, -1, 2]$
    """
    P, F = 5, 3
    TARGETS = torch.tensor(  # (P, F)
        [
            [0.5, -2.0, 2.0],
            [1.5, 0.0, 3.0],
            [1.5, -1.0, 1.0],
            [2.0, -1.0, 2.0],
            [-0.5, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    EXPECTED_SOLUTION = TARGETS.mean(0)
    EXPECTED_LOSS = ((EXPECTED_SOLUTION - TARGETS) ** 2).sum(-1).mean()

    def loss_fn(psi: torch.Tensor):
        # (1 + F, F), the baseline (1 row) + perturbed (F rows) psi
        assert psi.shape == (1 + F, F), f"{psi.shape=}"
        patient_loss = ((psi[:, None, :] - TARGETS[None, :, :]) ** 2).sum(-1)
        # patient_loss should have shape (1 + F, P): baseline + perturbed losses, per patient
        assert patient_loss.shape == (1 + F, P), f"{patient_loss.shape=}"
        return FixedEffectsEvaluation(
            patient_loss=patient_loss,
            predictions=torch.tensor(np.nan),
        )

    psi0 = torch.tensor([-3, 3, -2], dtype=torch.float64)
    psi0_copy = psi0.clone()
    eps_grad = 1e-6

    # Since the gradient has a closed form, we can check that
    # the finite difference approximation is exact up to the +h (eps_scaled) term
    grad, loss_baseline = fixed_effects_gradient(loss_fn, psi0, eps_base=eps_grad)
    assert grad.shape == (F, P), f"{grad.shape=}"
    eps_scaled = eps_grad * psi0.abs().clamp(min=1.0)
    expected_grad = 2.0 * (psi0 - TARGETS).T + eps_scaled.unsqueeze(-1)
    assert_close(grad, expected_grad)
    assert_close(loss_baseline, ((psi0 - TARGETS) ** 2).sum(-1))

    psi, loss = minimize_fixed_effects_loss(
        loss_fn, psi0, lr=5e-2, nb_iter=0, eps_grad=eps_grad
    )
    # nb_iter=0 must return psi0 unchanged instead of crashing
    torch.testing.assert_close(psi, psi0)

    # Run the gradient descent for 300 iterations
    psi, loss = minimize_fixed_effects_loss(
        loss_fn, psi0, lr=5e-2, nb_iter=300, eps_grad=eps_grad
    )
    # Checks that psi0 is not modified
    assert_close(psi0, psi0_copy)
    # final psi is close enough to theoretical global minimum
    torch.testing.assert_close(psi, EXPECTED_SOLUTION, atol=1e-3, rtol=0)
    # final loss function is close enough to its theoretical minimum
    torch.testing.assert_close(loss, EXPECTED_LOSS, atol=1e-3, rtol=0)
