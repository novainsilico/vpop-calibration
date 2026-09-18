"""Regression test documenting the M-step population regression bug.

See BUG_FINDINGS.md and IMPLEMENTATION_REVIEW.md #4.

`MStepState.update()` (vpop_calibration/saem/m_step.py) solves the
population regression for `beta` using ordinary least squares (`X^T X`).
With correlated PDUs and different covariate sets per PDU, the Gaussian
likelihood instead requires covariance weighting (`X^T Omega^-1 X`, a
generalized least-squares / GLS solve). The plain OLS solution need not be
a stationary point of the actual likelihood.

This test builds a case with an intercept-only PDU and an intercept+slope
PDU, and checks that the `beta` returned by `MStepState.update()` is (close
to) a stationary point of the deviance under the covariance the same update
step returns (zero gradient at the optimum). It currently FAILS: the
returned slope is 0, but the deviance's gradient there is nonzero (the
GLS-weighted solution has strictly lower deviance).
"""

import torch

from vpop_calibration.saem.m_step import MStepState


def test_mstep_beta_is_a_stationary_point_of_the_weighted_deviance():
    # First PDU has intercept + covariate x; second PDU has only an intercept.
    X = torch.tensor([[[1.0, x, 0.0], [0.0, 0.0, 1.0]] for x in [-1.0, 0.0, 1.0]])
    y = torch.tensor([[[0.0, -1.0], [1.0, 1.0], [0.0, 1.0]]])

    state = MStepState.from_init_gaussian_params(X, 1, 3, 2, y)
    proposal = state.update(y, 1.0)

    beta = proposal.beta.clone().requires_grad_()
    omega = proposal.omega
    inverse = torch.linalg.inv(omega)

    def deviance(coefficients):
        residual = y[0] - X @ coefficients
        return 3 * torch.linalg.slogdet(omega)[1] + (
            residual @ inverse * residual
        ).sum()

    loss = deviance(beta)
    loss.backward()

    assert beta.grad.abs().max().item() < 1e-3, (
        "The population coefficients returned by MStepState.update() should "
        "be a stationary point of the Gaussian deviance under the fitted "
        "covariance (near-zero gradient); instead the update solves an "
        "unweighted OLS regression, leaving the slope coefficient with a "
        f"large nonzero gradient ({beta.grad.tolist()})."
    )
