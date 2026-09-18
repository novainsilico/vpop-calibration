"""Regression test documenting the importance-sampling Student-t scale bug.

See BUG_FINDINGS.md #3 / IMPLEMENTATION_REVIEW.md #8.

`ImportanceSampler.fit_student_t_proposal()`
(vpop_calibration/pynlme/importance_sampling.py) passes
`torch.var(etas, 0)` directly as the Student-t `scale` parameter, which is a
standard-deviation-like scale, not a variance. It should take the square
root of the variance.

This test fits the proposal to conditional samples with a known standard
deviation (0.01) and asserts the fitted proposal's standard deviation is
close to that value. It currently FAILS because the proposal ends up
~100x (1/0.01) too narrow (fitted std ~= variance = 0.0001 instead of 0.01).
"""

from types import SimpleNamespace

import pytest
import torch

from vpop_calibration.pynlme.importance_sampling import ImportanceSampler
from vpop_calibration.pynlme.conditional_distribution import ConditionalDistribSamples


def test_student_t_proposal_scale_matches_sample_std():
    torch.manual_seed(0)
    true_std = 0.01
    target = torch.distributions.Normal(0.0, true_std)

    fake_model = SimpleNamespace(
        log_posterior_etas_all_patients=lambda x: SimpleNamespace(
            log_posterior=target.log_prob(x).sum(-1)
        )
    )
    etas = target.sample((10_000, 1, 1))

    sampler = ImportanceSampler(fake_model)
    sampler.fit_student_t_proposal(
        ConditionalDistribSamples(etas, etas, etas, etas)
    )

    fitted_std = sampler.dist.stddev.item()
    empirical_std = etas.std().item()

    assert fitted_std == pytest.approx(empirical_std, rel=0.2), (
        f"Fitted Student-t proposal std ({fitted_std}) should be close to "
        f"the empirical sample std ({empirical_std}); instead the code "
        "assigns the sample VARIANCE directly to `scale`, producing a "
        "proposal that is far too narrow."
    )
