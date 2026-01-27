import numpy as np
import pandas as pd
from typing import Optional

from .ode import OdeModel
from .vpop import generate_vpop_from_ranges
from .structural_model import StructuralOdeModel
from .nlme import NlmeModel


def simulate_dataset_from_ranges(
    ode_model: OdeModel,
    log_nb_individuals: int,
    param_ranges: dict[str, dict[str, float | bool]],
    initial_conditions: np.ndarray,
    protocol_design: Optional[pd.DataFrame],
    residual_error_variance: Optional[np.ndarray],
    error_model: Optional[str],  # "additive" or "proportional"
    time_steps: np.ndarray,
) -> pd.DataFrame:
    """Generate a simulated data set with an ODE model

    Simulates a dataset for training a surrogate model. Timesteps can be different for each output.
    The parameter space is explored with Sobol sequences.

    Args:
        log_nb_individuals (int): The number of simulated patients will be 2^this parameter
        param_ranges (list[dict]): For each parameter in the model, a dict describing the search space 'low': low bound, 'high': high bound, and 'log': True if the search space is log-scaled
        initial_conditions (array): set of initial conditions, one for each variable
        protocol_design (optional): a DataFrame with a `protocol_arm` column, and one column per parameter override
        residual_error_variance (np.array): A 1D array of residual error variances for each output.
        error_model (str): the type of error model ("additive" or "proportional").
        time_steps (np.array): an array with the time points
    Returns:
        pd.DataFrame: A DataFrame with columns 'id', parameter names, 'time', 'output_name', and 'value'.

    Notes:
        If a parameter appears both in the ranges and in the protocol design, the ranges take precedence.
    """

    # Validate input data
    params_to_explore = list(param_ranges.keys())

    if protocol_design is None:
        print("No protocol")
        params = params_to_explore
        params_in_protocol = []
        protocol_design_filt = pd.DataFrame({"protocol_arm": ["identity"]})
    else:
        params_in_protocol = protocol_design.drop(
            "protocol_arm", axis=1
        ).columns.tolist()
        # Find the paramaters that appear both in the ranges and the protocol
        overlap = set(params_to_explore) & set(params_in_protocol)
        if overlap != set():
            protocol_design_filt = protocol_design.drop(list(overlap), axis=1)
            print(
                f"Warning: ignoring entries {overlap} from the protocol design (already defined in the ranges)."
            )
        else:
            protocol_design_filt = protocol_design

        params = params_to_explore + params_in_protocol
    if set(params) != set(ode_model.param_names):
        raise ValueError(
            f"Under-defined system: missing {set(ode_model.param_names) - set(params)}"
        )
    # Generate the vpop using sobol sequences
    patients_df = generate_vpop_from_ranges(log_nb_individuals, param_ranges)

    # Add a choice of protocol arm for each patient
    protocol_arms = pd.DataFrame(protocol_design_filt["protocol_arm"].drop_duplicates())
    patients_df = patients_df.merge(protocol_arms, how="cross")
    # Add the outputs for each patient
    outputs = pd.DataFrame({"output_name": ode_model.variable_names})
    patients_df = patients_df.merge(outputs, how="cross")
    # Simulate the ODE model
    output_df = ode_model.run_trial(
        patients_df, initial_conditions, protocol_design_filt, time_steps
    )
    # Pivot to wide to add noise per model output
    wide_output = output_df.pivot_table(
        index=["id", *ode_model.param_names, "time", "protocol_arm"],
        columns="output_name",
        values="predicted_value",
    ).reset_index()

    if error_model is None:
        pass
    else:
        if residual_error_variance is None:
            raise ValueError("Undefined residual error variance.")
        else:
            # Add noise to the data
            noise = np.random.normal(
                np.zeros_like(residual_error_variance),
                np.sqrt(residual_error_variance),
                (wide_output.shape[0], ode_model.nb_outputs),
            )
            if error_model == "additive":
                wide_output[ode_model.variable_names] += noise
            elif error_model == "proportional":
                wide_output[ode_model.variable_names] += (
                    noise * wide_output[ode_model.variable_names]
                )
            else:
                raise ValueError(f"Incorrect error_model choice: {error_model}")
    # Pivot back to long format
    long_output = wide_output.melt(
        id_vars=[
            "id",
            "protocol_arm",
            "time",
            *ode_model.param_names,
        ],
        value_vars=ode_model.variable_names,
        var_name="output_name",
        value_name="value",
    )
    # Remove the protocol arm overrides from the data set, they described by the protocol_arm column now
    long_output = long_output.drop(params_in_protocol, axis=1)
    return long_output


def simulate_dataset_from_omega(
    ode_model: OdeModel,
    protocol_design: pd.DataFrame,
    time_steps: np.ndarray,
    init_conditions: np.ndarray,
    log_mi: dict[str, float],
    log_pdu: dict[str, dict[str, float]],
    error_model: str,
    res_var: list[float],
    covariate_map: dict[str, dict[str, dict[str, str | float]]],
    patient_covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic data set using an ODE model and population distributions of parameters

    Args:
        ode_model (OdeModel): The equations to be simulated
        protocol_design (pd.DataFrame): _description_
        time_steps (np.ndarray): _description_
        init_conditions (np.ndarray): _description_
        log_mi (dict[str, float]): _description_
        log_pdu (dict[str, dict[str, float]]): _description_
        error_model (str): _description_
        res_var (list[float]): _description_
        covariate_map (dict[str, dict[str, dict[str, str  |  float]]]): _description_
        patient_covariates (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    structural_model = StructuralOdeModel(ode_model, protocol_design, init_conditions)
    nlme_model = NlmeModel(
        structural_model,
        patient_covariates,
        log_mi,
        log_pdu,
        res_var,
        covariate_map,
        error_model,
        num_chains=1,
    )
    etas = nlme_model.sample_etas(nlme_model.num_chains)
    phi = nlme_model.etas_to_gaussian_params(etas)
    pdu = nlme_model.gaussian_to_physical_params(phi, nlme_model.log_MI)
    theta = nlme_model.assemble_individual_parameters(pdu).squeeze(0)
    vpop = pd.DataFrame(data=theta.cpu().numpy(), columns=nlme_model.descriptors)
    vpop["id"] = nlme_model.patients
    protocol_arms = patient_covariates[["id", "protocol_arm"]]
    vpop = vpop.merge(protocol_arms, on=["id"], how="left")
    vpop = vpop.merge(
        pd.DataFrame(data=nlme_model.outputs_names, columns=["output_name"]),
        how="cross",
    )
    time_df = pd.DataFrame(data=time_steps, columns=["time"])
    vpop = vpop.merge(time_df, how="cross")
    # add a dummy observation value
    vpop["value"] = 1.0
    nlme_model.add_observations(vpop)

    out_tensor, _ = nlme_model.predict_outputs_from_theta(theta)
    out_with_noise = nlme_model.add_residual_error(out_tensor)
    out_df = nlme_model.outputs_to_df(out_with_noise)
    out_df = out_df.rename(columns={"predicted_value": "value"})

    return out_df
