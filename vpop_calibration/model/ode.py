import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool
from typing import List, Any, Callable, Optional

from ..vpop import generate_vpop_from_ranges


class OdeModel:
    def __init__(
        self,
        equations: Callable,
        variable_names: List[str],
        param_names: List[str],
        tol: Optional[float] = 1e-6,
    ):
        """OdeModel

        Create a computational model given a set of equations.

        Args:
            equations (callable): A function describing the right hand side of the ODE system
            variable_names (List[str]): The names of the outputs of the system
            param_names (List[str]): The name of the parameters of the system
        """
        self.equations = equations
        self.variable_names = variable_names
        self.nb_outputs = len(variable_names)

        self.param_names = param_names
        self.nb_parameters = len(param_names)
        # Define the name of initial conditions as `<variable>_0`
        self.initial_cond_names = [v + "_0" for v in self.variable_names]

        self.tol = tol

    def simulate_model(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Solve the ODE model using formatted input data

        Args:
            input_data (pd.DataFrame): a DataFrame containing 'id', 'output_name', 'time', and columns for all individual parameters and initial conditions ('var_0', for var in variable_names) for each patient

        Returns:
            pd.DataFrame: a DataFrame with the same inputs and a new 'predicted_value' column
        """

        # group the data by individual to create tasks for each process
        tasks: list[Any] = []
        for _, ind_df in input_data.groupby("id"):
            for _, filtered_df in ind_df.groupby("protocol_arm"):
                params = filtered_df[self.param_names].iloc[0].values
                initial_conditions = filtered_df[self.initial_cond_names].iloc[0].values
                input_df = filtered_df[["id", "protocol_arm", "output_name", "time"]]
                indiv_task = {
                    "patient_inputs": input_df,
                    "initial_conditions": initial_conditions,
                    "params": params,
                    "equations": self.equations,
                    "output_names": self.variable_names,
                    "tol": self.tol,
                }
                tasks.append(indiv_task)
        with Pool() as pool:
            all_solutions: list[pd.DataFrame] = pool.map(_simulate_patient, tasks)
        output_data = pd.concat(all_solutions)
        return output_data

    def run_trial(
        self,
        vpop: pd.DataFrame,
        initial_conditions: np.ndarray,
        protocol_design: Optional[pd.DataFrame],
        time_steps: np.ndarray,
    ) -> pd.DataFrame:
        """Run a trial given a vpop, protocol and solving times

        Args:
            vpop (pd.DataFrame): The patient descriptors. Should contain the following columns
            - `id`
            - `protocol_arm`
            - `output_name`
            - one column per patient descriptor
            initial_conditions (np.ndarray): one set of initial conditions (same for all patients)
            protocol_design (Optional[pd.DataFrame]): Protocol design linking `protocol_arm` to actual parameter overrides
            time_steps (np.ndarray): The requested observation times. Same for all outputs

        Returns:
            pd.DataFrame: A merged output containing the following columns
            - `id`
            - one column per patient descriptor
            - `protocol_arm`
            - `output_name`
            - `predicted_value`: the simulated value

        Notes:
            Each patient will be run on each protocol arm, and all outputs will be included
        """

        # List the requested time steps for each output (here we use same solving times for all outputs)
        time_steps_df = pd.DataFrame({"time": time_steps})
        # Assemble the initial conditions in a dataframe
        init_cond_df = pd.DataFrame(
            data=[initial_conditions], columns=self.initial_cond_names
        )
        if protocol_design is None:
            protocol_design_to_use = pd.DataFrame({"protocol_arm": "identity"})
        else:
            protocol_design_to_use = protocol_design

        # Merge the data frames together
        # Add time steps and output names for all patients
        full_input_data = vpop.merge(time_steps_df, how="cross")
        # Add initial conditions for all patients
        full_input_data = full_input_data.merge(init_cond_df, how="cross")
        # Add protocol arm info by merging the protocol design
        full_input_data = full_input_data.merge(
            protocol_design_to_use, how="left", on="protocol_arm"
        )
        # Run the model
        output = self.simulate_model(full_input_data)

        merged_df = pd.merge(
            full_input_data,
            output,
            on=["id", "output_name", "time", "protocol_arm"],
            how="left",
        )
        return merged_df


def _simulate_patient(args: dict) -> pd.DataFrame:
    """Worker function to simulate a model on a single patient

    Args:
        args (dict): describes the simulation to be performed. Requires the following
            patient_inputs (pd.DataFrame): a data frame describing the patient to be simulated. The output data frame will be identical, with an additional `predicted_value` column. The inputs expect the following columns
                `id`
                `protocol_arm`
                `output_name`
                `time`
            initial_conditions (dict[str,float]): the initial conditions for each variable
            params (dict[str,float]): the patients descriptors
            equations (Callable): system right-hand side
            output_names (list[str]): the model output names, in the same order as in the equations
            tol (float): solver tolerance

    Returns:
        List(dict): A list of model result entries
    """

    # extract args
    input_df: pd.DataFrame = args["patient_inputs"]
    ind_id: pd.Series = input_df["id"].drop_duplicates()
    if ind_id.shape[0] > 1:
        raise ValueError("More than 1 patient was provided to `simulate_patient`")

    time_steps: List[float] = input_df["time"].drop_duplicates().to_list()
    initial_conditions: np.ndarray = args["initial_conditions"]
    params: np.ndarray = args["params"]
    equations: Callable = args["equations"]
    output_names: List[str] = args["output_names"]
    tol: float = args["tol"]

    time_span = (time_steps[0], time_steps[-1])

    sol = solve_ivp(
        equations,
        time_span,
        initial_conditions,
        method="LSODA",
        t_eval=time_steps,
        rtol=tol,
        atol=tol,
        args=params,
    )
    if not sol.success:
        raise ValueError(f"ODE integration failed: {sol.message}")

    # Filter the solver output to keep only the requested time steps for each output
    simulation_outputs_df = pd.DataFrame(data=sol.y.transpose(), columns=output_names)
    simulation_outputs_df["time"] = time_steps
    simulation_outputs_df = simulation_outputs_df.melt(
        id_vars=["time"],
        value_vars=list(output_names),
        var_name="output_name",
        value_name="predicted_value",
    )
    full_output = input_df.merge(
        simulation_outputs_df, how="left", on=["output_name", "time"]
    )
    return full_output
