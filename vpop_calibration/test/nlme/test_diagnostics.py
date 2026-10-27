from pathlib import Path

from pandas.testing import assert_frame_equal
import pytest
import pandas as pd
import numpy as np
from pandera.typing import DataFrame
import torch

from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics
from vpop_calibration.pynlme.plot import PlottingUtility
from vpop_calibration.pynlme.conditional_distribution import (
    ConditionalDistributionSampler,
)
from vpop_calibration.utils import seed_everything


@pytest.fixture
def sample_nlme_params() -> MixedEffectParameters:
    input = {
        "model_intrinsic": {"mi_1": {"prior": 10.0}},
        "pdu": {
            "pdu_1": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu1", "prior": 0.5}},
            },
            "pdu_2": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu2", "prior": 0.5}},
                "constraint": {"low": 1.0, "high": 100.0},
            },
        },
        "error_model": {
            "out_1": {"error_type": "additive", "initial_variance": 0.1},
            "out_2": {"error_type": "proportional", "initial_variance": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    return MixedEffectParameters.model_validate(input)


@pytest.fixture(scope="function")
def obs_data(np_rng) -> ObsData:
    protocol_arms = ["arm-A", "arm-B"]
    patients = {
        "id": ["p1", "p2"],
        "foo": [0.0, 5.0],
        "pdk_1": [0.0, 0.0],
        "protocol_arm": protocol_arms,
    }
    outputs = ["out_1", "out_2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))
    data = ObsData(DataFrame(df))
    return data


@pytest.fixture
def struct_model() -> StructuralModel:
    def equations(mi_1, pdu_1, pdu_2, pdk_1, t, protocol_ovr_1):
        out = torch.ones_like(t)
        return torch.cat((out, out), dim=-1)

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "protocol_ovr_1": [1, 2]}
    )
    struct_model = StructuralAnalytical(
        equations=equations,
        variable_names=["out_1", "out_2"],
        protocol_design=protocol_design,
    )
    return struct_model


GOLDEN_MAP_PATH = Path(__file__).parent / "stored_results" / "map_jinko_vpop.csv"
GOLDEN_COND_PATH = Path(__file__).parent / "stored_results" / "cond_jinko_vpop.csv"
GOLDEN_POP_PATH = Path(__file__).parent / "stored_results" / "pop_jinko_vpop.csv"


def test_diagnostics(sample_nlme_params, obs_data, struct_model, request):
    seed_everything(0)
    nlme_model = StatisticalModel(
        structural_model=struct_model, dataset=obs_data, input_params=sample_nlme_params
    )
    diagnostics = ModelDiagnostics(nlme_model)
    diagnostics.sample_conditional_distribution()
    diagnostics.compute_iwres()
    diagnostics.compute_population_residuals()
    diagnostics.compute_shrinkage()
    diagnostics.compute_vpc()
    diagnostics.compute_log_likelihood_importance_sampling()
    # Test the plotter in isolation
    plotter = PlottingUtility(diagnostics=diagnostics)
    plotter.map_estimates()
    plotter.individual_map_estimates()
    plotter.all_individual_map_estimates()
    plotter.map_estimates_gof()
    plotter.weighted_residuals(res_type="iwres")
    plotter.weighted_residuals(res_type="pwres")
    plotter.weighted_residuals(res_type="npde")
    plotter.map_vs_posterior()
    plotter.vpc()
    plotter.conditional_codistributions()
    cond_sampler = ConditionalDistributionSampler(nlme_model)
    cond_sampler.live_plot = True
    cond_sampler.run_sampler()

    map_jinko_vpop = nlme_model.to_jinko_vpop(
        input_df=diagnostics.sampler.map_parameters_df
    )
    if request.config.getoption("--update-goldens"):
        map_jinko_vpop.to_csv(GOLDEN_MAP_PATH, index=False)
    expected_map = pd.read_csv(GOLDEN_MAP_PATH)
    assert_frame_equal(map_jinko_vpop, expected_map)

    cond_jinko_vpop = nlme_model.to_jinko_vpop(
        input_df=diagnostics.sampler.total_samples_parameters_df
    )
    if request.config.getoption("--update-goldens"):
        cond_jinko_vpop.to_csv(GOLDEN_COND_PATH, index=False)
    expected_cond = pd.read_csv(GOLDEN_COND_PATH)
    assert_frame_equal(cond_jinko_vpop, expected_cond)

    pop_jinko_vpop = nlme_model.to_jinko_vpop(
        input_df=nlme_model.population_model_samples(nb_samples=5)
    )
    if request.config.getoption("--update-goldens"):
        pop_jinko_vpop.to_csv(GOLDEN_POP_PATH, index=False)
    expected_pop = pd.read_csv(GOLDEN_POP_PATH)
    assert_frame_equal(pop_jinko_vpop, expected_pop)
