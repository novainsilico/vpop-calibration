from vpop_calibration.pynlme.params import ErrorModel, MixedEffectParameters
from vpop_calibration.pynlme.data import ObsData

import pytest
from pydantic import ValidationError
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
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    return input


@pytest.fixture(scope="function")
def obs_data(np_rng) -> ObsData:
    protocol_arms = ["arm-A", "arm-B"]
    patients = {
        "id": ["p1", "p2"],
        "foo": [0.0, 0.0],
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


def test_duplicate_names():
    input1 = {
        "model_intrinsic": {"foo": {"prior": 10.0}},
        "pdu": {
            "foo": {
                "prior": 10.0,
                "prior_omega": 0.1,
            },
        },
        "error_model": {
            "out_1": {"error_type": "additive", "sigma": 0.1},
        },
    }

    input2 = {
        "pdu": {
            "foo": {
                "prior": 10.0,
                "prior_omega": 0.1,
            },
        },
        "error_model": {
            "out_1": {"error_type": "additive", "sigma": 0.1},
        },
        "pdk": ["foo"],
    }

    input3 = {
        "pdu": {
            "pdu_1": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu1", "prior": 0.5}},
            },
        },
        "error_model": {
            "out_1": {"error_type": "additive", "sigma": 0.1},
        },
        "pdk": ["foo"],
    }

    for input in [input1, input2, input3]:
        with pytest.raises(ValidationError):
            _params = MixedEffectParameters.model_validate(input)


def test_state_dict(sample_nlme_params):
    nlme_params = MixedEffectParameters.model_validate(sample_nlme_params)
    state_dict = nlme_params.get_state_dict()
    new_params = MixedEffectParameters.from_state_dict(state_dict)
    assert nlme_params == new_params


VALID_ERROR_MODELS = [
    {"error_type": "additive", "sigma": 0.1},
    {"error_type": "proportional", "sigma": 0.5},
    {"error_type": "combined", "sigma_add": 0.1, "sigma_prop": 0.5},
]

INVALID_ERROR_MODELS = [
    # An additive or proportional error model is parametrized by `sigma` alone
    {"error_type": "additive"},
    {"error_type": "additive", "sigma_add": 0.1},
    {"error_type": "additive", "sigma": 0.1, "sigma_add": 0.1},
    {"error_type": "proportional"},
    {"error_type": "proportional", "sigma_prop": 0.5},
    {"error_type": "proportional", "sigma": 0.5, "sigma_prop": 0.5},
    # A combined error model needs both components, and no `sigma`
    {"error_type": "combined"},
    {"error_type": "combined", "sigma_add": 0.1},
    {"error_type": "combined", "sigma_prop": 0.5},
    {"error_type": "combined", "sigma": 0.1, "sigma_add": 0.1, "sigma_prop": 0.5},
    # Variances are non-negative
    {"error_type": "additive", "sigma": -0.1},
    {"error_type": "combined", "sigma_add": -0.1, "sigma_prop": 0.5},
    # Unknown error type, and unexpected field
    {"error_type": "exponential", "sigma": 0.1},
    {"error_type": "additive", "sigma": 0.1, "sigmaa": 0.1},
]


@pytest.mark.parametrize("params", VALID_ERROR_MODELS)
def test_valid_error_model(params: dict):
    error_model = ErrorModel.model_validate(params)

    for variance, active in zip(
        error_model.variance_components, error_model.active_components
    ):
        if active:
            assert isinstance(variance, float) and variance >= 0.0
        else:
            assert variance == 0.0


@pytest.mark.parametrize("params", INVALID_ERROR_MODELS)
def test_invalid_error_model(params: dict):
    with pytest.raises(ValidationError):
        ErrorModel.model_validate(params)
