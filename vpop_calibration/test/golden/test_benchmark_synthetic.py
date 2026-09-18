from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from deepdiff import DeepDiff

import vpop_calibration.saem.optimizer as saem_optimizer
from vpop_calibration.api import (
    Config,
    NlmeConfigDict,
    NlmeModel,
    SaemConfigDict,
    StructuralAnalytical,
)

# Mirrors examples/benchmarking/benchmark_synthetic.ipynb: a one-observation-per-output
# log-normal NLME identifiability benchmark (5 correlated PDUs, additive noise).
DIMENSION = 5
OUTPUT_NAMES = [f"y_{j + 1}" for j in range(DIMENSION)]


@pytest.fixture
def benchmark_data() -> pd.DataFrame:
    # Generated once from the notebook's data-generating code (seed=0) and stored as a
    # static asset, so the test doesn't depend on numpy's RNG stream staying stable
    # across versions. Holds both the noisy observations SAEM is given (`value`) and the
    # realized latent process (`log_latent_value`, `measurement_error`), used only by the
    # `realized_latent` fixture below -- SAEM never sees those two columns.
    return pd.read_csv(
        Path(__file__).parent / "assets" / "benchmark_synthetic_data.csv"
    )


@pytest.fixture
def obs_data(benchmark_data) -> pd.DataFrame:
    return benchmark_data[["id", "time", "protocol_arm", "output_name", "value"]]


@pytest.fixture
def realized_latent(benchmark_data) -> dict:
    # Why not compared against the "TRUE" mean and covariance used to generate the input data?
    # Because SAEM recovers the parameters that best explain the actual
    # REALIZED latent values, not the generating distribution, so that's what a
    # recovery check should compare against.
    # For instance, the realized (resp. true) mean for x_4 is 0.21813133 (resp. 0.1) when it is
    # 0.10563274 (resp. 0.2) for x_5.
    log_latent = benchmark_data.pivot(
        index="id", columns="output_name", values="log_latent_value"
    )[OUTPUT_NAMES]
    measurement_error = benchmark_data.pivot(
        index="id", columns="output_name", values="measurement_error"
    )[OUTPUT_NAMES]
    return {
        "log_mean": log_latent.mean(axis=0).to_numpy(),
        "log_cov": log_latent.cov().to_numpy(),
        "noise_variance": measurement_error.var(axis=0, ddof=1).to_numpy(),
    }


@pytest.fixture
def struct_model() -> StructuralAnalytical:
    def one_point_model(x_1, x_2, x_3, x_4, x_5, t):
        levels = torch.cat((x_1, x_2, x_3, x_4, x_5), dim=-1)
        return levels + torch.zeros_like(t)

    return StructuralAnalytical(
        one_point_model, [f"y_{j + 1}" for j in range(DIMENSION)]
    )


@pytest.fixture
def synthetic_config() -> Config:
    return Config(
        seed=2026,
        nlme=NlmeConfigDict(nb_chains=1, live_plot=False, progress_bar=False),
        saem=SaemConfigDict(
            nb_iter_burnin=30,
            nb_iter_learning=270,
            nb_iter_smoothing=200,
            nb_mcmc_transitions=5,
            learning_rate_power=0.7,
            live_plot=False,
            progress_bars=False,
            logging=False,
        ),
    )


@pytest.fixture
def model_params() -> dict:
    return {
        "pdu": {
            f"x_{j + 1}": {"prior": 10, "prior_omega": 3} for j in range(DIMENSION)
        },
        "error_model": {
            f"y_{j + 1}": {"error_type": "additive", "sigma": 4.0}
            for j in range(DIMENSION)
        },
    }


@pytest.mark.golden_test("stored_results/test_benchmark_synthetic.yml")
def test_benchmark_synthetic(
    obs_data,
    realized_latent,
    struct_model,
    synthetic_config,
    model_params,
    monkeypatch,
    golden,
    request,
):
    # Test runs always set IS_PYTEST_RUNNING (see pyproject.toml), which normally forces
    # PySaem down to a token 1/2/2-iteration schedule regardless of the passed config.
    # This benchmark needs its full schedule to actually converge, so bypass it here.
    monkeypatch.setattr(saem_optimizer, "smoke_test", False)

    nlme_model = NlmeModel(
        structural_model=struct_model,
        df=obs_data,
        input_params=model_params,
        config=synthetic_config,
    )
    nlme_model.optimizer.run()
    nlme_model.diagnostics.sample_conditional_distribution(nb_samples=500)
    nlme_model.diagnostics.compute_log_likelihood_importance_sampling(
        nb_proposal_samples=1000
    )

    recovered_beta = (
        nlme_model.statistical_model.population_betas.detach().cpu().numpy()
    )
    recovered_omega = nlme_model.statistical_model.omega_pop.detach().cpu().numpy()
    recovered_noise_variance = (
        nlme_model.statistical_model.residual_var.sigma_add.detach().cpu().numpy()
    )
    log_lik = nlme_model.diagnostics.importance_sampler.log_lik

    # Recovery check: SAEM should land close to the *realized* latent population (see
    # `realized_latent` fixture) rather than the generating distribution's parameters.
    np.testing.assert_allclose(recovered_beta, realized_latent["log_mean"], atol=0.1)
    omega_rel_error = np.linalg.norm(
        recovered_omega - realized_latent["log_cov"], ord="fro"
    ) / np.linalg.norm(realized_latent["log_cov"], ord="fro")
    assert omega_rel_error < 0.2
    # Unlike beta/omega, the residual variance doesn't track this draw's realized
    # measurement-error variance nearly as tightly (consistently 2-6x higher here,
    # stable across iteration budgets we tried) -- any correlation the fit doesn't
    # fully capture (e.g. the strong x_4/x_5 coupling) shows up as extra residual
    # variance instead. So this is a loose sanity bound, not a recovery check.
    assert np.all((recovered_noise_variance > 0) & (recovered_noise_variance < 0.1))
    assert np.isfinite(log_lik)

    # Golden-snapshot check: catches any change to the algorithm's output, intended or
    # not, independently of the recovery tolerances above.
    actual = nlme_model.get_state_dict()
    expected = golden.out["output"]

    if request.config.getoption("--update-goldens"):
        assert actual == expected
    else:
        diff = DeepDiff(
            actual,
            expected,
            ignore_type_in_groups=[(tuple, list), (float, np.float64)],
            math_epsilon=1e-16,
            ignore_nan_inequality=True,
        )
        assert diff == {}
