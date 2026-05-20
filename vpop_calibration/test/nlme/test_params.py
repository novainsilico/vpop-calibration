from vpop_calibration.nlme_model.params import MixedEffectParameters

import pytest
import numpy as np


@pytest.fixture
def sample_nlme_params() -> dict:
    input = {
        "model_intrinsic": {"mi_1": {"prior": 10.0}},
        "pdu": {
            "pdu_1": {
                "prior_mean": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu1", "prior": 0.5}},
            },
            "pdu_2": {
                "prior_mean": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu2", "prior": 0.5}},
            },
        },
        "error_model": {
            "out_1": {"type": "additive", "sigma": 0.1},
            "out_2": {"type": "proportional", "sigma": 0.5},
        },
    }
    return input


def test_nlme_params(sample_nlme_params):
    nlme_params = MixedEffectParameters.model_validate(sample_nlme_params)
    assert nlme_params.pdu_names == ["pdu_1", "pdu_2"]
    assert nlme_params.mi_names == ["mi_1"]
    assert nlme_params.beta_names == [
        "pdu_1",
        "coef_foo_pdu1",
        "pdu_2",
        "coef_foo_pdu2",
    ]
    assert nlme_params.beta_init == [np.log(10.0), 0.5, np.log(10.0), 0.5]
    assert nlme_params.covariate_names == ["foo"]
