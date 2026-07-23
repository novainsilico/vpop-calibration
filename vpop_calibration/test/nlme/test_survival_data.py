from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.data import ObsData

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
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
        "time_to_event": {
            "hazard_name": "hazard",
            "coefficients": {"beta_1": {"prior": -0.2}},
        },
    }
    return input


@pytest.fixture(scope="function")
def obs_data(np_rng) -> pd.DataFrame:
    protocol_arms = ["arm-A", "arm-B"]
    patients = {
        "id": ["p1", "p2"],
        "foo": [0.0, 0.0],
        "pdk_1": [0.0, 0.0],
        "protocol_arm": protocol_arms,
        "event_time": [10.0, 20.0],
        "event_status": [True, False],
        "hazard_name": ["hazard", "hazard"],
    }
    outputs = ["out_1", "out_2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))
    return df


def test_nlme_survival_inputs(sample_nlme_params, obs_data):
    # A data set with complete survival inputs is accepted
    obs_data_complete = ObsData(DataFrame(obs_data))
    # A data set with one missing column is rejected
    with pytest.raises(Exception):
        _obs_data_incomplete = ObsData(DataFrame(obs_data.drop(columns=["event_time"])))
    # A data set with no survival inputs at all is accepted
    obs_data_no_survival = ObsData(
        DataFrame(obs_data.drop(columns=["event_time", "event_status", "hazard_name"]))
    )

    nlme_params_complete = MixedEffectParameters.model_validate(sample_nlme_params)

    assert nlme_params_complete.surv_coeff_init == [-0.2]
    assert nlme_params_complete.surv_coeff_names == ["beta_1"]
    sample_nlme_params.pop("time_to_event")
    nlme_params_no_survival = MixedEffectParameters.model_validate(sample_nlme_params)

    assert nlme_params_no_survival.surv_coeff_init == []
    assert nlme_params_no_survival.surv_coeff_names == []

    # Compatible params and data set
    nlme_params_complete.validate_data(obs_data_complete)
    # Incompatible params and data set
    with pytest.raises(Exception):
        nlme_params_no_survival.validate_data(obs_data_complete)

    with pytest.raises(Exception):
        nlme_params_complete.validate_data(obs_data_no_survival)

    # Compatible params and data
    nlme_params_no_survival.validate_data(obs_data_no_survival)
