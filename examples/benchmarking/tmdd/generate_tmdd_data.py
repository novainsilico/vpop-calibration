import numpy as np
import pandas as pd
import uuid

from vpop_calibration import *

true_params = {
    "R0_mean": 3.0,
    "R0_sd": 0.1,
    "inj": 10.0,
    "Vc_mean": 3.0,
    "Vc_sd": 0.1,
    "k_off": 1.0,
    "k_D": 1.0,
    "k_deg": 1e-2,
    "k_eL_mean": 5e-2,
    "k_eL_sd": 0.25,
    "k_eP_mean": 1e-1,
    "k_eP_sd": 0.2,
}


# Define the ODE model
def equations(t, y, k_eL, k_eP, inj, Vc, R0):
    """TMDD model equations.

    Concentrations are expressed in `nM` and time in `h`.

    """
    yL, yR, yP = y
    k_on = true_params["k_off"] / true_params["k_D"]
    k_syn = R0 * true_params["k_deg"]
    dyL = -k_eL * yL - k_on * yL * yR + true_params["k_off"] * yP
    dyR = k_syn - true_params["k_deg"] * yR - k_on * yL * yR + true_params["k_off"] * yP
    dyP = k_on * yL * yR - true_params["k_off"] * yP - k_eP * yP
    ydot = [dyL, dyR, dyP]
    return ydot


def generate_benchmark_data():
    # Define the parameters values

    tmax = 10.0  # hours
    nb_steps = 10
    time_steps = np.linspace(0.0, tmax, nb_steps).tolist()
    protocol_design = pd.DataFrame(
        {"protocol_arm": ["identity"], "inj": [true_params["inj"]]}
    )

    true_res_var = [0.05, 0.01, 0.01]
    error_model_type = "proportional"

    def init_assignment(k_eL, k_eP, inj, Vc, R0):
        return [inj / Vc, R0, 0.0]

    variable_names = ["L", "R", "P"]
    parameter_names = ["k_eL", "k_eP", "inj", "Vc", "R0"]

    tmdd_model = OdeModel(equations, init_assignment, variable_names, parameter_names)

    # NLME model parameters

    true_log_MI = {}
    true_log_PDU = {
        "k_eL": {
            "mean": np.log(true_params["k_eL_mean"]),
            "sd": true_params["k_eL_sd"],
        },
        "k_eP": {
            "mean": np.log(true_params["k_eP_mean"]),
            "sd": true_params["k_eP_sd"],
        },
        "R0": {"mean": np.log(true_params["R0_mean"]), "sd": true_params["R0_sd"]},
        "Vc": {"mean": np.log(true_params["Vc_mean"]), "sd": true_params["Vc_sd"]},
    }

    # Define the patients data frame
    nb_patients = 100
    # Give them a unique id
    patients_df = pd.DataFrame({"id": [str(uuid.uuid4()) for _ in range(nb_patients)]})
    patients_df["protocol_arm"] = "identity"

    obs_df_full = simulate_dataset_from_omega(
        tmdd_model,
        protocol_design,
        time_steps,
        true_log_MI,
        true_log_PDU,
        error_model_type,
        true_res_var,
        None,
        patients_df,
    )

    # Filter and format the data set for exporting

    # Only the antibody concentration is considered
    obs_df = obs_df_full.loc[obs_df_full["output_name"] == "L"][
        ["id", "time", "protocol_arm", "output_name", "value"]
    ]
    output_names = ["L"]
    return obs_df, patients_df, tmdd_model, time_steps, protocol_design, output_names


def export_saemix_data(df):
    df_saemix = df[["id", "time", "value"]]
    df_saemix.to_csv(
        "./tmdd_synthetic_data_saemix.csv", float_format="%.3f", index=False
    )


def export_nlmixr2_data(df):
    df_nlmixr2 = df[["id", "time", "value"]].rename(
        columns={"value": "DV", "time": "TIME", "id": "ID"}
    )
    df_nlmixr2["EVID"] = 0
    df_nlmixr2["CMT"] = 1
    df_nlmixr2["AMT"] = 0
    df_nlmixr2.loc[df_nlmixr2["TIME"] == 0, "EVID"] = 1
    df_nlmixr2.to_csv(
        "./tmdd_synthetic_data_nlmixr2.csv", sep=" ", float_format="%.3f", index=False
    )
