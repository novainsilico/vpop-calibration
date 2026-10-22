from math import ceil
from pathlib import Path

from pandas.testing import assert_frame_equal
import pytest
import pandas as pd
import numpy as np
from pandera.typing import DataFrame
import torch

from vpop_calibration.pynlme.config import NlmeConfigDict
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.saem.optimizer import PySaem
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical
from vpop_calibration.utils import seed_everything


@pytest.fixture
def nlme_params() -> MixedEffectParameters:
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
        "pdk_1": [0.0, 1.0],
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


GOLDEN_MAP_PATH = Path(__file__).parent / "stored_results" / "map.csv"
GOLDEN_CONDITIONAL_PATH = (
    Path(__file__).parent / "stored_results" / "conditional_samples.csv"
)


def test_conditional_sampling(nlme_params, obs_data, struct_model, request):
    seed_everything(0)
    statistical_model = StatisticalModel(
        structural_model=struct_model,
        dataset=obs_data,
        input_params=nlme_params,
        config=NlmeConfigDict(nb_chains=3, live_plot=False, progress_bar=False),
    )
    diagnostics = ModelDiagnostics(statistical_model)
    optimizer = PySaem(
        model=statistical_model,
        config=SaemConfigDict(
            nb_iter_burnin=1,
            nb_iter_learning=2,
            nb_iter_smoothing=3,
            learning_rate_power=0.7,
            live_plot=False,
            progress_bars=False,
            logging=False,
        ),
    )
    optimizer.run()
    nb_desired_samples = 5
    diagnostics.sample_conditional_distribution(
        nb_samples=nb_desired_samples, optimizer_last_state=optimizer.mh_state
    )
    expected_nb_samples = (
        ceil(nb_desired_samples / statistical_model.nb_chains)
        * statistical_model.nb_chains
    )
    sampler = diagnostics.sampler
    total = sampler.total_samples
    assert total.eta_samples.shape == (
        expected_nb_samples,
        statistical_model.nb_patients,
        statistical_model.nb_pdu,
    )

    params_df = sampler.total_samples_parameters_df
    assert len(params_df) == expected_nb_samples * statistical_model.nb_patients
    assert (
        params_df["id"].nunique() == expected_nb_samples * statistical_model.nb_patients
    )
    assert params_df["iteration"].nunique() == len(sampler.samples)
    assert params_df["chain"].nunique() == statistical_model.nb_chains

    if request.config.getoption("--update-goldens"):
        params_df.to_csv(GOLDEN_CONDITIONAL_PATH, index=False)
    expected_conditional_samples = pd.read_csv(GOLDEN_CONDITIONAL_PATH)
    assert_frame_equal(params_df, expected_conditional_samples)

    map_df = sampler.map_parameters_df
    assert len(map_df) == statistical_model.nb_patients
    assert map_df["id"].nunique() == statistical_model.nb_patients

    if request.config.getoption("--update-goldens"):
        map_df.to_csv(GOLDEN_MAP_PATH, index=False)
    expected_map = pd.read_csv(GOLDEN_MAP_PATH)
    assert_frame_equal(map_df, expected_map)
