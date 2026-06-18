import pandas as pd
import uuid
import numpy as np
from scipy.stats.qmc import Sobol, scale
from pydantic import BaseModel, TypeAdapter
from pandera.typing import DataFrame

from vpop_calibration.structural_model import StructuralModel
from vpop_calibration.pynlme.schemas import ObsDataSchema, patientDataSchema
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.residuals import add_predictive_error
from vpop_calibration.config import smoke_test


class ParamBounds(BaseModel):
    low: float
    high: float
    log: bool


ParamRanges = dict[str, ParamBounds]
paramRangesAdapter = TypeAdapter(ParamRanges)


def init_patient_ids(nb_individuals: int) -> pd.DataFrame:
    """Initiate a single column data frame with unique patient ids."""
    ids = [str(uuid.uuid4()) for _ in range(nb_individuals)]
    return pd.DataFrame({"id": ids})


def sample_descriptors_sobol_sequences(
    log_nb_individuals: int, param_ranges: ParamRanges
) -> pd.DataFrame:
    """Given parameter ranges, generate individual patients by Sobol sampling"""
    nb_individuals = np.power(2, log_nb_individuals)
    params_to_explore = list(param_ranges.keys())
    nb_parameters = len(params_to_explore)
    if nb_parameters != 0:

        # Create a sobol sampler to generate parameter values
        sobol_engine = Sobol(d=nb_parameters, scramble=True)
        sobol_sequence = sobol_engine.random_base2(log_nb_individuals)
        samples = scale(
            sobol_sequence,
            [param_ranges[param_name].low for param_name in params_to_explore],
            [param_ranges[param_name].high for param_name in params_to_explore],
        )

        # Handle log-scaled parameters
        for j, param_name in enumerate(params_to_explore):
            if param_ranges[param_name].log:
                samples[:, j] = np.exp(samples[:, j])
        # Create the full data frame of patient descriptors
        patients_df = pd.DataFrame(data=samples, columns=params_to_explore)
    else:
        # No parameter requested, create empty data frame
        patients_df = pd.DataFrame()

    ids_df = init_patient_ids(nb_individuals)
    patients_df = pd.concat((ids_df, patients_df), axis=1)
    return patients_df


def sample_protocol_arms(
    patients: pd.DataFrame,
    protocol_arms: list[str],
    np_rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Given a set of patients and a set of protocol arms, associate each patient with a single arm randomly."""
    if np_rng == None:
        rng = np.random.default_rng()
    else:
        rng = np_rng
    nb_protocols = len(protocol_arms)
    new_patients = patients.assign(
        protocol_arm=np.array(protocol_arms)[
            rng.integers(nb_protocols, size=patients.shape[0])
        ]
    )
    return new_patients


def cross_join_protocol_arms(
    patients: pd.DataFrame,
    protocol_arms: list[str],
):
    """Given a set of patients and a set of protocol arms, cross join the set of protocol arms to have all patients simulated on all arms. A new `id` column is created to uniquely identify them, storing the existing id in `previous_id`."""

    new_patients = patients.merge(
        pd.DataFrame({"protocol_arm": protocol_arms}), how="cross"
    ).rename(columns={"id": "previous_id"})
    # A new id column is created to have the 1:1 patient-protocol arm structure
    new_patients["id"] = [str(uuid.uuid4()) for _ in range(new_patients.shape[0])]

    return new_patients


def patients_to_obs_df(
    vpop: pd.DataFrame,
    time_steps: list[float],
    output_names: list[str],
    dummy_value: float = 0.0,
) -> pd.DataFrame:
    """Add necessary columns to transform a patient data frame to an observation data frame.

    Patients will be simulated on all output_names and all time_steps, and a dummy observed value of 0 is added.
    """
    patients = patientDataSchema.validate(vpop)
    outputs_df = pd.DataFrame({"output_name": output_names})
    time_df = pd.DataFrame({"time": time_steps})

    obs_df = patients.merge(outputs_df, how="cross").merge(time_df, how="cross")
    obs_df["value"] = dummy_value
    obs_df = obs_df.assign(task=lambda r: r.output_name + "_" + r.protocol_arm)
    validated_output = ObsDataSchema.validate(obs_df)
    return validated_output


def generate_training_data(
    struct_model: StructuralModel,
    ranges: dict,
    log_nb_ind: int,
    time: list[float],
) -> pd.DataFrame:
    """Given a structural model and parameter ranges, generate a training data set."""
    if smoke_test:
        log_nb_ind = 1

    param_ranges = paramRangesAdapter.validate_python(ranges)
    # Sample the patient descriptors using Sobol sequences
    vpop = sample_descriptors_sobol_sequences(
        log_nb_individuals=log_nb_ind, param_ranges=param_ranges
    )
    # Assign each patient to all arms of the protocol design
    extended_vpop = cross_join_protocol_arms(
        patients=vpop, protocol_arms=struct_model.protocol_arms
    )
    # Add output names and timepoints
    obs_df = patients_to_obs_df(
        vpop=extended_vpop, time_steps=time, output_names=struct_model.output_names
    )
    # Simulate the structural model
    sim_df = (
        struct_model.simulate_from_df(vpop=extended_vpop, obs_df=obs_df)
        .drop(columns=["value"])
        .rename(columns={"predicted_value": "value"})
    )
    # Rename the previous id into `id` (not unique)
    training_df = (
        sim_df[["previous_id", "output_name", "time", "id", "value"]]
        .merge(extended_vpop, on=["id", "previous_id"])
        .drop(columns=["id"])
        .rename(columns={"previous_id": "id"})
    )
    return training_df


def generate_synthetic_data(
    struct_model: StructuralModel,
    param_distrib: dict,
    nb_patients: int,
    time: list[float],
    np_rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if smoke_test:
        nb_patients = 2
    # Initiate the patient data frame
    raw_vpop = init_patient_ids(nb_individuals=nb_patients)
    # Sample a protocol arm for each patient
    patients_with_protocol = sample_protocol_arms(
        patients=raw_vpop, protocol_arms=struct_model.protocol_arms, np_rng=np_rng
    )
    # Add output names and time steps to create an actual observation df
    obs_df = patients_to_obs_df(
        vpop=patients_with_protocol,
        time_steps=time,
        output_names=struct_model.output_names,
    )
    # Initiate the elements required to construct a mixed effects model
    data = ObsData(DataFrame(obs_df))
    params = MixedEffectParameters.model_validate(param_distrib)
    assert params.pdk == [], "PDK are not yet supported in data generation."
    assert params.covariate_names == [], "Covariates are not yet supported."
    # Create the nlme model
    nlme_model = StatisticalModel(
        structural_model=struct_model, dataset=data, prior_params=params, nb_chains=1
    )
    # Prepare model inputs
    eta = nlme_model.eta_samples_chains
    gaussian = nlme_model.convert_etas_to_gaussian_all_patients(eta)
    physical = nlme_model.convert_gaussian_to_physical(
        psi=gaussian, log_mi=nlme_model.log_mi
    )
    theta = nlme_model.convert_physical_to_thetas_all_patients(physical_params=physical)
    inputs = nlme_model.convert_thetas_to_model_parameters_all_patients(theta)
    # Run the model
    outputs, _ = nlme_model.predict_all_patients(inputs)
    # Add noise
    noisy_pred = add_predictive_error(
        observations=nlme_model.data.full_obs,
        predictions=outputs,
        error_model_selector=nlme_model.error_model_selector,
        sigma=nlme_model.residual_var,
    )
    # Convert back to pandas
    df = (
        nlme_model.data.full_obs.to_pandas(prediction=noisy_pred)
        .drop(columns=["value"])
        .rename(columns={"predicted_value": "value"})
    )
    # validate the output
    validated_out = ObsDataSchema.validate(df)
    return validated_out
