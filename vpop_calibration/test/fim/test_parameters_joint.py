import pytest
import pandas as pd
import numpy as np
from pandera.typing import DataFrame
import torch

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.structural_model.analytical import StructuralAnalytical


@pytest.fixture(scope="function")
def sample_model(np_rng) -> StatisticalModel:
    def equations(mi_1, pdu_1, pdu_2, pdk_1, t, protocol_ovr_1, beta_0):
        out = torch.ones_like(t)
        return torch.cat((out, out, out, out), dim=-1)

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "protocol_ovr_1": [1, 2]}
    )
    struct_model = StructuralAnalytical(
        equations=equations,
        variable_names=["out_1", "out_2", "log_hazard", "cumulative_hazard"],
        protocol_design=protocol_design,
    )
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
            },
        },
        "error_model": {
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
        "time_to_event": {
            "hazard_name": "hazard",
            "coefficients": {"beta_0": {"prior": -0.5}},
        },
    }

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
    df["event_time"] = 10.0
    df["event_status"] = True
    df["hazard_name"] = "hazard"
    data = ObsData(DataFrame(df))
    model = StatisticalModel(
        structural_model=struct_model,
        input_params=MixedEffectParameters.model_validate(input),
        dataset=data,
    )

    return model


def test_flat_parameter_roundtrip(sample_model):
    theta = sample_model.flatten()
    params = sample_model.unflatten(theta)

    torch.testing.assert_close(params.beta, sample_model.population_betas)
    torch.testing.assert_close(
        params.omega_lower_chol, sample_model.omega_pop_lower_chol
    )
    torch.testing.assert_close(params.log_mi, sample_model.log_mi)
    torch.testing.assert_close(params.surv_coeffs, sample_model.surv_coeffs)
    assert params.res_var == sample_model.residual_var


def test_parameter_names(sample_model):
    names = sample_model.flat_parameter_names
    assert names == [
        "pdu_1",
        "coef_foo_pdu1",
        "pdu_2",
        "coef_foo_pdu2",
        "omega_log_diag_pdu_1",
        "omega_log_diag_pdu_2",
        "omega_pdu_2_pdu_1",
        "mi_1",
        "beta_0",
        "sigma_add_out_1",
        "sigma_prop_out_2",
    ]
