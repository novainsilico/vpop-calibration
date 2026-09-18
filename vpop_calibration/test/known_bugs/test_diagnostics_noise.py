"""Regression tests documenting diagnostics that ignore residual/predictive
noise correctly.

See BUG_FINDINGS.md #2 and IMPLEMENTATION_REVIEW.md #9, #10, #11.

- `compute_pwres()` (vpop_calibration/pynlme/diagnostics.py) computes
  `torch.cov(obs_patient.T)` (variance of the *observed* values across time)
  instead of the variance of the Monte-Carlo *predictive* distribution. Since
  `obs_patient` is 1-D, this collapses to a near-meaningless scalar and is
  insensitive to the actual residual variance.
- `compute_npde()` compares observations against noise-free structural
  predictions without adding residual observation noise, so small
  observation errors look like extreme discrepancies.
- `compute_vpc()` uses deterministic conditional predictions without residual
  observation noise, so prediction intervals collapse to a point regardless
  of the configured residual variance.

All three tests assert that these diagnostics respond to the configured
residual variance the way they are documented to. They currently FAIL.
"""

import numpy as np
import pandas as pd
import torch

from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.config import NlmeConfigDict
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics
from vpop_calibration.structural_model.analytical import StructuralAnalytical


def equations(p, t):
    return torch.ones_like(t)


def _make_diagnostics(values, sigma):
    df = pd.DataFrame(
        {
            "id": ["a"] * len(values),
            "time": [float(i) for i in range(len(values))],
            "output_name": ["y"] * len(values),
            "value": values,
        }
    )
    params = MixedEffectParameters.model_validate(
        {
            "pdu": {"p": {"prior": 1.0, "prior_omega": 0.1}},
            "error_model": {"y": {"error_type": "additive", "sigma": sigma}},
        }
    )
    model = StatisticalModel(
        StructuralAnalytical(equations, ["y"]),
        ObsData(df),
        params,
        NlmeConfigDict(live_plot=False, progress_bar=False),
    )
    return ModelDiagnostics(model)


def test_pwres_scales_with_residual_variance():
    """PWRES should shrink as the assumed residual variance grows (a fixed
    residual, weighted by a larger predictive variance, gives a smaller
    weighted residual). It currently does not respond to sigma at all,
    because it is computed from the observed values' own variance across
    time instead of the predictive variance.
    """
    torch.manual_seed(0)
    small_sigma = _make_diagnostics([0.9, 1.1, 0.9], 1.0)
    small_sigma.compute_pwres(nb_samples=1000)

    torch.manual_seed(0)
    large_sigma = _make_diagnostics([0.9, 1.1, 0.9], 100.0)
    large_sigma.compute_pwres(nb_samples=1000)

    assert not np.allclose(
        small_sigma.pwres.residual_value.to_numpy(),
        large_sigma.pwres.residual_value.to_numpy(),
    ), (
        "PWRES must depend on the configured residual variance; instead it "
        "is computed from the variance of the observed values across time "
        "and is insensitive to sigma."
    )


def test_npde_accounts_for_residual_noise():
    """Increasing the residual sigma from 1 to 100 should make the same
    0.1-magnitude observation deviations look far less extreme (closer to
    0) in the normalized prediction discrepancy errors (NPDE), since a
    larger assumed observation noise better explains the deviation.
    NPDE currently ignores sigma entirely (it compares observations against
    noise-free structural predictions), so the two results are identical.
    """
    torch.manual_seed(0)
    small_sigma = _make_diagnostics([0.9, 1.1, 0.9], 1.0)
    small_sigma.compute_npde(nb_samples=1000)

    torch.manual_seed(0)
    large_sigma = _make_diagnostics([0.9, 1.1, 0.9], 100.0)
    large_sigma.compute_npde(nb_samples=1000)

    assert not np.allclose(
        small_sigma.npde.residual_value.to_numpy(),
        large_sigma.npde.residual_value.to_numpy(),
    ), (
        "NPDE must depend on the configured residual variance; instead it "
        "compares observations against noise-free structural predictions "
        "and is insensitive to sigma."
    )


def test_vpc_prediction_interval_reflects_residual_variance():
    """With nonzero residual variance, the VPC's predicted interval
    (pred_lower to pred_upper) must have positive width; it currently
    collapses to a point because the deterministic conditional predictions
    used for the interval have no residual noise added.
    """
    torch.manual_seed(0)
    diag = _make_diagnostics([0.9, 1.1, 0.9], 1.0)
    diag.sample_conditional_distribution(nb_samples=50)
    diag.compute_vpc(nb_bins=1, quantiles=[0.1, 0.5, 0.9])

    interval_width = (diag.vpc["pred_upper"] - diag.vpc["pred_lower"]).abs()
    assert (interval_width > 1e-6).all(), (
        "VPC prediction intervals must have nonzero width when residual "
        "variance is nonzero; instead they collapse to a point because "
        "residual observation noise is never added to the replicated "
        "predictions."
    )
