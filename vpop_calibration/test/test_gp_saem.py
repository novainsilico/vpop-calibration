import numpy as np
import pandas as pd
import pickle
import uuid

from vpop_calibration import *


def equations_with_abs(t, y, k_a, k_12, k_21, k_el):
    # y[0] is A_absorption, y[1] is A_central, y[2] is A_peripheral
    A_absorption, A_central, A_peripheral = y[0], y[1], y[2]
    dA_absorption_dt = -k_a * A_absorption
    dA_central_dt = (
        k_a * A_absorption + k_21 * A_peripheral - k_12 * A_central - k_el * A_central
    )
    dA_peripheral_dt = k_12 * A_central - k_21 * A_peripheral

    ydot = [dA_absorption_dt, dA_central_dt, dA_peripheral_dt]
    return ydot


variable_names = ["A0", "A1", "A2"]
parameter_names = ["k_a", "k_12", "k_21", "k_el"]

tmax = 24.0
initial_conditions = np.array([10.0, 0.0, 0.0])

protocol_design = pd.DataFrame(
    {"protocol_arm": ["arm-A", "arm-B"], "k_el": [0.5, 10.0]}
)
nb_protocols = len(protocol_design)

pk_two_compartments_model = OdeModel(
    equations_with_abs, variable_names, parameter_names
)

model_file = "vpop_calibration/test/gp_model_for_tests.pkl"


def test_gp_training():
    # Define the ode model

    nb_timesteps = 15
    time_steps = np.linspace(0.0, tmax, nb_timesteps)

    log_nb_patients = 3
    param_ranges = {
        "k_12": {"low": 0.02, "high": 0.07, "log": False},
        "k_21": {"low": 0.1, "high": 0.3, "log": False},
        "k_a": {"low": -1.0, "high": 0.0, "log": True},
    }

    dataset = simulate_dataset_from_ranges(
        pk_two_compartments_model,
        log_nb_patients,
        param_ranges,
        initial_conditions,
        protocol_design,
        None,
        None,
        time_steps,
    )

    learned_ode_params = list(param_ranges.keys())
    descriptors = learned_ode_params + ["time"]

    # initiate our GP class
    myGP = GP(
        dataset,
        descriptors,
        var_strat="IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
        kernel="RBF",  # Either RBF or SMK
        data_already_normalized=False,  # default
        nb_inducing_points=10,
        mll="ELBO",  # default, otherwise PLL
        nb_training_iter=10,
        training_proportion=0.7,
        learning_rate=0.1,
        lr_decay=0.99,
        jitter=1e-6,
        log_inputs=learned_ode_params,
    )
    myGP.train()
    myGP.plot_loss()
    myGP.plot_obs_vs_predicted("training")
    myGP.plot_individual_solution(0)
    myGP.plot_all_solutions("training")
    with open(model_file, "wb") as file:
        pickle.dump(myGP, file)


def test_gp_saem():
    time_span_rw = (0, 24)
    nb_steps_rw = 5

    # For each output and for each patient, give a list of time steps to be simulated
    time_steps_rw = np.linspace(time_span_rw[0], time_span_rw[1], nb_steps_rw).tolist()

    # Parameter definitions
    true_log_MI = {"k_21": 0.0}
    true_log_PDU = {
        "k_12": {"mean": -1.0, "sd": 0.25},
    }
    error_model_type = "additive"
    true_res_var = [0.5, 0.02, 0.1]
    true_covariate_map = {
        "k_12": {"foo": {"coef": "cov_foo_k12", "value": 0.2}},
    }

    # Create a patient data frame
    # It should contain at the very minimum one `id` per patient
    nb_patients = 5
    patients_df = pd.DataFrame({"id": [str(uuid.uuid4()) for _ in range(nb_patients)]})
    rng = np.random.default_rng()
    patients_df["protocol_arm"] = rng.binomial(1, 0.5, nb_patients)
    patients_df["protocol_arm"] = patients_df["protocol_arm"].apply(
        lambda x: "arm-A" if x == 0 else "arm-B"
    )
    patients_df["k_a"] = rng.lognormal(-1, 0.1, nb_patients)
    patients_df["foo"] = rng.lognormal(0.1, 0.1, nb_patients)

    print(f"Simulating {nb_patients} patients on {nb_protocols} protocol arms")
    obs_df = simulate_dataset_from_omega(
        pk_two_compartments_model,
        protocol_design,
        time_steps_rw,
        initial_conditions,
        true_log_MI,
        true_log_PDU,
        error_model_type,
        true_res_var,
        true_covariate_map,
        patients_df,
    )

    # Initial pop estimates
    # Parameter definitions
    init_log_MI = {}  # {"k_21": -1.0}
    init_log_PDU = {
        "k_21": {"mean": -1.0, "sd": 0.2},
        "k_12": {"mean": -0.1, "sd": 0.1},
    }
    error_model_type = "additive"
    init_res_var = [0.1, 0.05, 0.5]
    init_covariate_map = {
        "k_12": {"foo": {"coef": "cov_foo_k12", "value": -0.1}},
        "k_21": {},
    }

    with open(model_file, "rb") as file:
        myGP = pickle.load(file)
    # Create a structural model
    structural_gp = StructuralGp(myGP)
    # Create a NLME moedl
    nlme_surrogate = NlmeModel(
        structural_gp,
        patients_df,
        init_log_MI,
        init_log_PDU,
        init_res_var,
        init_covariate_map,
        error_model_type,
    )
    # Create an optimizer: here we use SAEM
    optimizer = PySAEM(
        nlme_surrogate,
        obs_df,
        mcmc_burn_in=0,
        mcmc_first_burn_in=0,
        mcmc_nb_samples=1,
        mcmc_proposal_var_scaling_factor=0.5,
        nb_phase1_iterations=1,
        nb_phase2_iterations=None,
        convergence_threshold=1e-4,
        patience=5,
        learning_rate_power=0.8,
        annealing_factor=0.95,
        verbose=False,
    )

    optimizer.run()
    optimizer.continue_iterating(nb_phase2_further_iterations=1)
    optimizer.plot_convergence_history()
    optimizer.plot_map_estimates()
