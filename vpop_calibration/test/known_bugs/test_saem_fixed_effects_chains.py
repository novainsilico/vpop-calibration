"""Regression test documenting the multi-chain fixed-effects objective bug.

See BUG_FINDINGS.md and IMPLEMENTATION_REVIEW.md #3.

`PySaem.build_fixed_effects_objective_function()`
(vpop_calibration/saem/optimizer.py) averages the Gaussian latent parameters
across MH chains BEFORE evaluating the nonlinear structural model. This is
not the same as averaging the complete-data log-likelihood contributions
across chains (the actual SAEM objective), because the model is nonlinear.

This test builds a 2-chain case (`y = m * p`, observation 1, chain samples
p=0.5 and p=2) and asserts that the optimizer's fixed-effects gradient has
the same sign as the gradient of the correctly-averaged per-chain-likelihood
objective. It currently FAILS: the code's gradient points in the OPPOSITE
direction, i.e. the fixed-effect update can move away from, not toward,
the optimum.
"""

import pandas as pd
import torch

from vpop_calibration.pynlme.config import NlmeConfigDict
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.residuals import log_likelihood_observation
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.saem.fixed_effects import compute_fixed_effects_gradient
from vpop_calibration.saem.optimizer import PySaem
from vpop_calibration.structural_model.analytical import StructuralAnalytical


def equations(p, m, t):
    return p * m + torch.zeros_like(t)


def test_fixed_effects_gradient_matches_mean_chain_likelihood_gradient():
    params = MixedEffectParameters.model_validate(
        {
            "model_intrinsic": {"m": {"prior": 0.8}},
            "pdu": {"p": {"prior": 1.0, "prior_omega": 1.0}},
            "error_model": {"y": {"error_type": "additive", "sigma": 1.0}},
            "pdk": [],
        }
    )
    model = StatisticalModel(
        structural_model=StructuralAnalytical(equations=equations, variable_names=["y"]),
        dataset=ObsData(
            pd.DataFrame(
                {
                    "id": ["a", "b"],
                    "time": [0.0, 0.0],
                    "value": [1.0, 1.0],
                    "output_name": ["y", "y"],
                }
            )
        ),
        input_params=params,
        config=NlmeConfigDict(nb_chains=2),
    )
    optimizer = PySaem(model, SaemConfigDict(live_plot=False, progress_bars=False))

    psi = torch.log(torch.tensor([[[0.5], [0.5]], [[2.0], [2.0]]]))
    objective = optimizer.build_fixed_effects_objective_function(psi.mean(0, keepdim=True))

    def expected_objective(log_mi):
        physical = model.convert_gaussian_to_physical(psi, log_mi, model.surv_coeffs)
        theta = model.convert_physical_to_thetas_all_patients(physical)
        pred, _ = model.predict_all_patients(
            model.convert_thetas_to_model_parameters_all_patients(theta)
        )
        return -log_likelihood_observation(
            model.data.full_obs, pred, model.residual_var, model.config.residual_min_variance
        ).sum(1).mean()

    log_mi = model.log_mi
    actual_gradient = compute_fixed_effects_gradient(objective, log_mi, 1e-5)[0].item()
    expected_gradient = (
        (expected_objective(log_mi + 1e-5) - expected_objective(log_mi)) / 1e-5
    ).item()

    assert (actual_gradient > 0) == (expected_gradient > 0), (
        "The fixed-effects objective's gradient should have the same sign as "
        "the gradient of the correctly mean-chain-likelihood objective; "
        f"instead they point in opposite directions (code={actual_gradient}, "
        f"expected={expected_gradient}), because the code averages the "
        "Gaussian latent parameters across chains before evaluating the "
        "nonlinear model instead of averaging per-chain likelihoods."
    )
