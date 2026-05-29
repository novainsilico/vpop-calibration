import pytest
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
import torch

from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.model import NlmeModel
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step


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
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    return MixedEffectParameters.model_validate(input)


@pytest.fixture
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
        out = torch.zeros_like(t)
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


def test_mh_step(sample_nlme_params, obs_data, struct_model):
    nlme_model = NlmeModel(
        structural_model=struct_model, dataset=obs_data, prior_params=sample_nlme_params
    )
    nb_samples = 1
    etas = nlme_model.sample_etas(nb_samples)
    etas = torch.zeros_like(etas)
    gaussian_params = nlme_model.convert_etas_to_gaussian_all_patients(etas)
    # Test the log prior function for etas
    predictions = nlme_model.log_posterior_etas_all_patients(etas)
    assert predictions.log_posterior.shape == (nb_samples, nlme_model.nb_patients)
    init_state = MetropolisHastingsState(
        etas=etas,
        gaussian_params=gaussian_params,
        log_prob=predictions.log_posterior,
        step_size=0.1,
        complete_likelihood=predictions.log_posterior.mean(dim=0).sum(dim=0),
        prediction=predictions.predictions,
    )

    new_state = mh_step(
        nlme_model=nlme_model, previous_state=init_state, learning_rate=0.1
    )
