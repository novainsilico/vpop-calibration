import pandas as pd
import uuid
import numpy as np
from scipy.stats.qmc import Sobol, scale


def generate_vpop_from_ranges(
    log_nb_individuals: int, param_ranges: dict[str, dict[str, float | bool]]
) -> pd.DataFrame:
    """Generate a vpop of patients from parameter ranges using Sobol sequences

    Args:
        log_nb_individuals (int): The vpop size will be 2^log_nb_individuals
        param_ranges (dict[str, dict[str, float  |  bool]]): One entry for each parameter to be explored
        - `param_name`: {`low`: float, `high`: float, `log`: bool}. Turn `log` to true to define log-scaled ranges

    Returns:
        pd.DataFrame: A set of patients with a generated `id`, and a column per descriptor

    Note:
        This method may be called with an empty dict, to return a list of patient ids.
    """

    nb_individuals = np.power(2, log_nb_individuals)
    params_to_explore = list(param_ranges.keys())
    nb_parameters = len(params_to_explore)
    if nb_parameters != 0:

        # Create a sobol sampler to generate parameter values
        sobol_engine = Sobol(d=nb_parameters, scramble=True)
        sobol_sequence = sobol_engine.random_base2(log_nb_individuals)
        samples = scale(
            sobol_sequence,
            [param_ranges[param_name]["low"] for param_name in params_to_explore],
            [param_ranges[param_name]["high"] for param_name in params_to_explore],
        )

        # Handle log-scaled parameters
        for j, param_name in enumerate(params_to_explore):
            if param_ranges[param_name]["log"] == True:
                samples[:, j] = np.exp(np.log(10) * samples[:, j])
        # Create the full data frame of patient descriptors
        patients_df = pd.DataFrame(data=samples, columns=params_to_explore)
    else:
        # No parameter requested, create empty data frame
        patients_df = pd.DataFrame()

    ids = [str(uuid.uuid4()) for _ in range(nb_individuals)]
    patients_df.insert(0, "id", ids)
    return patients_df
