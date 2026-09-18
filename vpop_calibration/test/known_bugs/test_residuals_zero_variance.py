"""Regression test documenting the zero-residual-variance likelihood bug.

See BUG_FINDINGS.md #4 / IMPLEMENTATION_REVIEW.md #2.

`compute_normal_likelihood()` (vpop_calibration/pynlme/residuals.py) selects
which outputs are "continuous" via
`torch.logical_or(residual_error.sigma_add, residual_error.sigma_prop)`
instead of the purpose-built `additive_output`/`proportional_output` boolean
flags. Since `torch.logical_or` treats any nonzero float as True, an output
whose estimated/prior sigma is exactly 0.0 is silently excluded from the
likelihood regardless of how badly its predictions fit the data.

This test asserts that a badly-fitting prediction (100 vs an observed value
of 21/11) contributes a large negative log-likelihood even when the additive
sigma prior happens to be exactly zero (the variance floor `min_variance`
should still apply). It currently FAILS because the buggy mask drops the
output entirely, returning a log-likelihood of 0 regardless of fit quality.
"""

import pandas as pd
import torch

from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical
from vpop_calibration.pynlme.residuals import compute_normal_likelihood


def equations(offset, gain, t):
    return offset + gain + 0 * t


def _build_model():
    obs = ObsData(
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "output_name": ["y", "y"],
                "time": [0.0, 0.0],
                "value": [21.0, 11.0],
                "offset": [20.0, 10.0],
            }
        )
    )
    params = MixedEffectParameters(
        pdu={"gain": {"prior": 1.0, "prior_omega": 0.1}},
        pdk=["offset"],
        error_model={"y": {"error_type": "additive", "sigma": 1.0}},
    )
    return StatisticalModel(StructuralAnalytical(equations, ["y"]), obs, params)


def test_zero_additive_sigma_still_penalizes_bad_predictions():
    model = _build_model()
    min_variance = 1e-6
    zero_sigma = model.residual_var._replace(sigma_add=torch.zeros(1))

    exact_fit = torch.tensor([[21.0, 11.0]])
    bad_fit = torch.tensor([[100.0, 100.0]])

    ll_exact = compute_normal_likelihood(
        model.data.full_obs, exact_fit, zero_sigma, min_variance
    )
    ll_bad = compute_normal_likelihood(
        model.data.full_obs, bad_fit, zero_sigma, min_variance
    )

    assert ll_bad.tolist() != [[0.0, 0.0]], (
        "A wildly wrong prediction (100 vs observed 21/11) must not "
        "contribute a log-likelihood of exactly 0 just because the sigma "
        "prior is 0; the configured variance floor should still apply."
    )
    assert (ll_bad < ll_exact).all(), (
        "The exact-fit prediction must have a strictly higher log-likelihood "
        "than the wildly-wrong one, once the variance floor is honored."
    )
