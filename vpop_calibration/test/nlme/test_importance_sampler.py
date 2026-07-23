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
from vpop_calibration.pynlme.importance_sampling import ImportanceSampler
from vpop_calibration.pynlme.conditional_distribution import (
    ConditionalDistributionSampler,
)


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


def test_importance_sampling(sample_nlme_params, obs_data, struct_model):
    nlme_model = StatisticalModel(
        structural_model=struct_model, dataset=obs_data, input_params=sample_nlme_params
    )
    cond_sampler = ConditionalDistributionSampler(nlme_model)
    cond_sampler.run_sampler()
    sampler = ImportanceSampler(nlme_model)
    sampler.fit_student_t_proposal(conditional_samples=cond_sampler.total_samples)
    sampler.compute_likelihood()


def test_state_dict(sample_nlme_params, obs_data, struct_model):
    nlme_model = StatisticalModel(
        structural_model=struct_model, dataset=obs_data, input_params=sample_nlme_params
    )
    cond_sampler = ConditionalDistributionSampler(nlme_model)
    cond_sampler.run_sampler()
    sampler = ImportanceSampler(nlme_model)
    state_dict_empty = sampler.get_state_dict()

    new_sampler_empty = ImportanceSampler.from_state_dict(
        model=nlme_model, state_dict=state_dict_empty
    )
    assert new_sampler_empty.dist is None

    sampler.fit_student_t_proposal(conditional_samples=cond_sampler.total_samples)
    sampler.compute_likelihood()
    state_dict_not_empty = sampler.get_state_dict()

    new_sampler_not_empty = ImportanceSampler.from_state_dict(
        model=nlme_model, state_dict=state_dict_not_empty
    )
    assert new_sampler_not_empty.dist is not None
    assert new_sampler_not_empty.log_lik == sampler.log_lik
