import pytest
import numpy as np
import pandas as pd

from vpop_calibration import *


@pytest.fixture
def ode_model_setup():
    def equations_with_abs(t, y, k_a, k_12, k_21, k_el, dose):
        A_absorption, A_central, A_peripheral = y[0], y[1], y[2]
        dA_absorption_dt = -k_a * A_absorption
        dA_central_dt = (
            k_a * A_absorption
            + k_21 * A_peripheral
            - k_12 * A_central
            - k_el * A_central
        )
        dA_peripheral_dt = k_12 * A_central - k_21 * A_peripheral

        ydot = [dA_absorption_dt, dA_central_dt, dA_peripheral_dt]
        return ydot

    def init_assignment(k_a, k_12, k_21, k_el, dose):
        return [dose, 0.0, 0.0]

    variable_names = ["A0", "A1", "A2"]
    parameter_names = ["k_a", "k_12", "k_21", "k_el", "dose"]

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "dose": [0.5, 10.0]}
    )

    return (
        equations_with_abs,
        init_assignment,
        variable_names,
        parameter_names,
        protocol_design,
    )


@pytest.fixture
def param_structure(use_case):
    match use_case:
        case 1:
            log_mi = {}
            log_pdu = {
                "k_12": {"mean": -1.0, "sd": 0.25},
                "k_21": {"mean": -1.0, "sd": 0.25},
                "k_a": {"mean": -1.0, "sd": 0.25},
                "k_el": {"mean": -1.0, "sd": 0.25},
            }
        case _:
            log_mi = {"k_21": 0.0, "k_el": 0.0}
            log_pdu = {
                "k_12": {"mean": -1.0, "sd": 0.25},
                "k_a": {"mean": -1.0, "sd": 0.25},
            }
    return log_mi, log_pdu


@pytest.fixture
def covariate_map_for_tests(include_cov):
    if include_cov:
        cov_map = {
            "k_12": {"foo": {"coef": "cov_foo_k12", "value": 0.2}},
            "k_21": {},
            "k_a": {},
            "k_el": {},
        }
    else:
        cov_map = None
    return cov_map


@pytest.fixture
def patients_df_for_tests(include_cov):
    patients_df = pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": ["arm-A", "arm-B"]})
    if include_cov:
        patients_df["foo"] = [1.0, 2.0]
    return patients_df


@pytest.mark.parametrize("error_model", ["additive", "proportional"])
@pytest.mark.parametrize("include_cov", [True, False])
@pytest.mark.parametrize("use_case", [1, 2])
def test_generate_data_omega(
    ode_model_setup,
    patients_df_for_tests,
    covariate_map_for_tests,
    param_structure,
    error_model,
):
    (
        equations,
        init_assignment,
        variable_names,
        parameter_names,
        protocol_design,
    ) = ode_model_setup
    log_mi, log_pdu = param_structure

    pk_model = OdeModel(
        equations, init_assignment, variable_names, parameter_names, multithreaded=False
    )
    time_steps = [0.0, 1.0]
    # Parameter definitions
    true_res_var = [0.5, 0.02, 0.1]
    time_steps = np.arange(0.0, 10.0, 4.0)
    patients_df = patients_df_for_tests
    covariate_map = covariate_map_for_tests

    obs_df = simulate_dataset_from_omega(
        pk_model,
        protocol_design,
        time_steps,
        log_mi,
        log_pdu,
        error_model,
        true_res_var,
        covariate_map,
        patients_df,
    )
