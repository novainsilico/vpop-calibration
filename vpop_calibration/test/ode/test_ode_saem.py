import numpy as np
import pandas as pd
import uuid

from vpop_calibration import *


def test_ode_saem(np_rng):
    variable_names = ["A0", "A1", "A2"]
    parameter_names = ["k_a", "k_12", "k_21", "k_el", "dose"]

    def equations(t, y, k_a, k_12, k_21, k_el, dose):
        # y[0] is A_absorption, y[1] is A_central, y[2] is A_peripheral
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

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "dose": [0.5, 10.0]}
    )

    pk_two_compartments_model = OdeModel(
        equations, init_assignment, variable_names, parameter_names
    )

    # Parameter definitions
    error_model_type = "additive"

    # Create a patient data frame
    # It should contain at the very minimum one `id` per patient
    nb_patients = 3
    patients_df = pd.DataFrame({"id": [str(uuid.uuid4()) for _ in range(nb_patients)]})
    patients_df["protocol_arm"] = np_rng.binomial(1, 0.5, nb_patients)
    patients_df["protocol_arm"] = patients_df["protocol_arm"].apply(
        lambda x: "arm-A" if x == 0 else "arm-B"
    )
    patients_df["k_a"] = np_rng.lognormal(-1, 0.1, nb_patients)
    patients_df["foo"] = np_rng.lognormal(0.1, 0.1, nb_patients)

    obs_df = (
        patients_df[["id", "protocol_arm"]]
        .merge(pd.DataFrame({"output_name": variable_names}), how="cross")
        .merge(pd.DataFrame({"time": [0.0, 1.0, 2.0]}), how="cross")
    )
    obs_df["value"] = np_rng.normal()

    # Initial pop estimates
    # Parameter definitions
    init_log_MI = {"k_12": -1.0}
    init_log_PDU = {
        "k_21": {"mean": -1.0, "sd": 0.2},
        "k_el": {"mean": -1.0, "sd": 0.2},
    }
    error_model_type = "additive"
    init_res_var = [0.1, 0.05, 0.5]
    init_covariate_map = {
        "k_21": {"foo": {"coef": "cov_foo_k12", "value": 0.1}},
        "k_el": {},
    }

    # Create a structural model
    structural_ode = StructuralOdeModel(pk_two_compartments_model, protocol_design)
    # Create a NLME moedl
    nlme = NlmeModel(
        structural_ode,
        patients_df,
        init_log_MI,
        init_log_PDU,
        init_res_var,
        init_covariate_map,
        error_model_type,
    )
    # Create an optimizer: here we use SAEM
    optimizer = PySaem(
        nlme,
        obs_df,
    )

    optimizer.run()
