import pandas as pd
import pytest
import numpy as np

from vpop_calibration.interface import NlmeModel
from vpop_calibration.structural_model.gp import StructuralGp
from vpop_calibration.model.gp import GP
from vpop_calibration.pynlme.plot import PlottingUtility


@pytest.fixture
def sample_nlme_params() -> dict:
    input = {
        "model_intrinsic": {"k3": {"prior": 10.0}},
        "pdu": {
            "k1": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu1", "prior": 0.5}},
            },
            "k2": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu2", "prior": 0.5}},
                "constraint": {"low": 1.0, "high": 100.0},
            },
        },
        "error_model": {
            "s1": {"error_type": "additive", "sigma": 0.1},
            "s2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": [],
    }
    return input


@pytest.fixture
def obs_data(np_rng) -> pd.DataFrame:
    protocol_arms = ["arm-A", "arm-B"]
    patients = {
        "id": ["p1", "p2"],
        "foo": [0.0, 5.0],
        "protocol_arm": protocol_arms,
    }
    outputs = ["s1", "s2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))
    return df


def test_gp_saem(np_rng, obs_data, sample_nlme_params):
    time_steps = pd.DataFrame({"time": [0.0, 1.0]})
    params = ["k1", "k2", "k3", "time"]
    patients_training = pd.DataFrame(
        {"id": ["p1", "p2"], "k1": [1.0, 2.0], "k2": [3.0, 4.0], "k3": [5.0, 6.0]}
    )
    outputs = pd.DataFrame({"output_name": ["s1", "s2"]})
    protocol_arms = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"]})

    training_df = (
        patients_training.merge(outputs, how="cross")
        .merge(protocol_arms, how="cross")
        .merge(time_steps, how="cross")
    )
    training_df["value"] = np_rng.normal()

    gp = GP(training_df, params)

    struct_model = StructuralGp(gp)

    nlme_model = NlmeModel(
        structural_model=struct_model,
        df=obs_data,
        prior_params=sample_nlme_params,
    )
    nlme_model.optimizer.run()


def test_gp_diagnostics(np_rng, obs_data, sample_nlme_params):
    time_steps = pd.DataFrame({"time": [0.0, 1.0]})
    params = ["k1", "k2", "k3", "time"]
    patients_training = pd.DataFrame(
        {"id": ["p1", "p2"], "k1": [1.0, 2.0], "k2": [3.0, 4.0], "k3": [5.0, 6.0]}
    )
    outputs = pd.DataFrame({"output_name": ["s1", "s2"]})
    protocol_arms = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"]})

    training_df = (
        patients_training.merge(outputs, how="cross")
        .merge(protocol_arms, how="cross")
        .merge(time_steps, how="cross")
    )
    training_df["value"] = np_rng.normal()

    gp = GP(training_df, params)

    struct_model = StructuralGp(gp)

    nlme_model = NlmeModel(
        structural_model=struct_model,
        df=obs_data,
        prior_params=sample_nlme_params,
    )
    nlme_model.plot.check_surrogate_validity_gp()
