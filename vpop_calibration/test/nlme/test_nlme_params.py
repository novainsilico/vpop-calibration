from vpop_calibration.nlme_model.params import MixedEffectParameters
from vpop_calibration.nlme_model.data import ObsData

import pytest
import numpy as np
import pandas as pd
from pandera.typing import DataFrame


@pytest.fixture
def sample_nlme_params() -> dict:
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
            "out_1": {"type": "additive", "sigma": 0.1},
            "out_2": {"type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    return input


@pytest.fixture
def obs_data(np_rng) -> ObsData:
    patients = {"id": ["p1", "p2"], "foo": [0.0, 0.0], "pdk_1": [0.0, 0.0]}
    protocol_arms = ["arm-A", "arm-B"]
    outputs = ["out_1", "out_2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(protocol_arms, columns=["protocol_arm"]), how="cross")
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))
    data = ObsData(DataFrame(df))
    return data


def test_nlme_params(sample_nlme_params, obs_data):
    nlme_params = MixedEffectParameters.model_validate(sample_nlme_params)
    assert nlme_params.pdu_names == ["pdu_1", "pdu_2"]
    assert nlme_params.mi_names == ["mi_1"]
    assert nlme_params.beta_names == [
        "pdu_1",
        "coef_foo_pdu1",
        "pdu_2",
        "coef_foo_pdu2",
    ]
    transformed_prior_pdu1 = np.log(10.0)
    shifted_pdu2 = (10.0 - 1.0) / (100.0 - 1.0)
    transformed_prior_pdu2 = np.log(shifted_pdu2 / (1 - shifted_pdu2))
    assert nlme_params.beta_init == [
        transformed_prior_pdu1,
        0.5,
        transformed_prior_pdu2,
        0.5,
    ]
    assert nlme_params.covariate_names == ["foo"]
    assert nlme_params.pdk == ["pdk_1"]

    nlme_params.validate_data(obs_data)
