import torch
import pandas as pd
import numpy as np
import uuid
from typing import Callable

from .model.gp import GP
from .ode import OdeModel
from .utils import device


class StructuralModel:
    def __init__(
        self,
        parameter_names,
        output_names,
        protocol_arms,
        tasks,
        task_idx_to_output_idx,
        task_idx_to_protocol,
    ):
        """Initialize a structural model

        Args:
            parameter_names (list[str]): _description_
            output_names (list[str]): _description_
            protocol_arms (list[str]): _description_
            tasks (list[str]): _description_
            task_idx_to_output_idx (list[str]): _description_
            task_idx_to_protocol (list[str]): _description_
        """
        self.parameter_names: list[str] = parameter_names
        self.nb_parameters: int = len(self.parameter_names)
        self.output_names: list[str] = output_names
        self.nb_outputs: int = len(self.output_names)
        self.protocols: list[str] = protocol_arms
        self.nb_protocols: int = len(self.protocols)
        self.tasks: list[str] = tasks
        self.task_idx_to_output_idx: dict[int, int] = task_idx_to_output_idx
        self.task_idx_to_protocol: dict[int, str] = task_idx_to_protocol

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        chunks: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Not implemented")


class StructuralGp(StructuralModel):
    def __init__(self, gp_model: GP):
        """Create a structural model from a GP

        Args:
            gp_model (GP): The trained GP
        """
        # list the GP parameters, except time, as it will be handled differently in the NLME model
        parameter_names = [p for p in gp_model.data.parameter_names if p != "time"]
        super().__init__(
            parameter_names,
            gp_model.data.output_names,
            gp_model.data.protocol_arms,
            gp_model.data.tasks,
            gp_model.data.task_idx_to_output_idx,
            gp_model.data.task_idx_to_protocol,
        )
        self.gp_model = gp_model

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        chunks: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (num_chains, nb_patients, nb_timesteps, nb_params) = X.shape
        nb_obs_per_chain = prediction_index[0].shape[0]
        prediction_index_expanded = (
            torch.arange(num_chains).repeat_interleave(nb_obs_per_chain),
            prediction_index[0].repeat(num_chains),
            prediction_index[1].repeat(num_chains),
            prediction_index[2].repeat(num_chains),
        )
        # Simulate the GP
        X_vertical = X.view(-1, nb_params)
        out_cat, var_cat = self.gp_model.predict_wide_scaled(X_vertical)
        num_tasks = out_cat.shape[-1]
        out_wide = out_cat.view(num_chains, nb_patients, nb_timesteps, -1)
        var_wide = var_cat.view(num_chains, nb_patients, nb_timesteps, -1)

        # Retrieve the necessary rows and columns to transform into a single column tensor
        y = out_wide[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        var = var_wide[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        return y, var


class StructuralOdeModel(StructuralModel):
    def __init__(
        self,
        ode_model: OdeModel,
        protocol_design: pd.DataFrame,
        init_conditions: np.ndarray,
    ):
        self.ode_model = ode_model
        protocol_arms = protocol_design["protocol_arm"].drop_duplicates().to_list()
        self.protocol_design = protocol_design
        output_names: list[str] = self.ode_model.variable_names
        tasks: list[str] = [
            output + "_" + protocol
            for protocol in protocol_arms
            for output in output_names
        ]
        # Map tasks to output names
        task_to_output = {
            output_name + "_" + protocol_arm: output_name
            for output_name in output_names
            for protocol_arm in protocol_arms
        }
        # Map task index to output index
        task_idx_to_output_idx = {
            tasks.index(k): output_names.index(v) for k, v in task_to_output.items()
        }
        # Map task to protocol arm
        task_to_protocol = {
            output_name + "_" + protocol_arm: protocol_arm
            for output_name in output_names
            for protocol_arm in protocol_arms
        }
        # Map task index to protocol arm
        task_idx_to_protocol = {tasks.index(k): v for k, v in task_to_protocol.items()}

        # list the structural model parameters: the protocol overrides are ignored
        self.protocol_overrides = self.protocol_design.drop(
            columns="protocol_arm"
        ).columns.to_list()
        parameter_names = list(
            set(self.ode_model.param_names) - set(self.protocol_overrides)
        )
        self.nb_protocol_overrides = len(self.protocol_overrides)

        super().__init__(
            parameter_names,
            output_names,
            protocol_arms,
            tasks,
            task_idx_to_output_idx,
            task_idx_to_protocol,
        )

        self.init_cond_df = pd.DataFrame(
            data=[init_conditions], columns=self.ode_model.initial_cond_names
        )

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        chunks: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (nb_chains, nb_patients, nb_timesteps, nb_params) = X.shape
        patient_index_full, rows_full, tasks_full = prediction_index
        list_rows = torch.split(rows_full, chunks)
        list_tasks = torch.split(tasks_full, chunks)

        output_list = []
        for chain_X in X:  # iterate through the individual chains
            # Separate the individual patients
            list_X = [ind_X for ind_X in chain_X]

            input_df_list = []
            for ind_X, ind_rows, ind_tasks in zip(list_X, list_rows, list_tasks):
                temp_id = str(uuid.uuid4())
                # Extract the parameters and time values
                params = ind_X.index_select(0, ind_rows).cpu().detach().numpy()
                # Extract the task order
                task_index = ind_tasks.cpu().detach().numpy()
                # Format the data inputs
                # This step is where the order of parameters is implicit
                input_df_temp = pd.DataFrame(
                    data=params, columns=self.parameter_names + ["time"]
                )
                # The passed params include the _global_ time steps
                # Filter the time steps that we actually want for this patient
                input_df_temp = input_df_temp.iloc[ind_rows.cpu().numpy()]
                # Add the task index as a temporary column
                input_df_temp["task_index"] = task_index
                # Deduce protocol arm and output name from task index
                input_df_temp["protocol_arm"] = input_df_temp["task_index"].apply(
                    lambda t: self.task_idx_to_protocol[t]
                )
                input_df_temp["output_name"] = input_df_temp["task_index"].apply(
                    lambda t: self.output_names[self.task_idx_to_output_idx[t]]
                )
                # Remove the unnecessary task index column
                input_df_temp = input_df_temp.drop(columns=["task_index"])
                input_df_temp["id"] = temp_id
                # Add the protocol overrides
                if self.nb_protocol_overrides > 0:
                    input_df_temp = input_df_temp.merge(
                        self.protocol_design, how="left", on=["protocol_arm"]
                    )
                # Add the initial conditions
                input_df_temp = input_df_temp.merge(self.init_cond_df, how="cross")
                input_df_list.append(input_df_temp)

            full_input = pd.concat(input_df_list)
            # Simulate the ODE model
            output_df = self.ode_model.simulate_model(full_input)
            # Convert back to tensor
            out_tensor = torch.as_tensor(
                output_df["predicted_value"].values, device=device
            )
            output_list.append(out_tensor)
        out_full = torch.stack(output_list, dim=0).to(device)
        out_var = torch.zeros_like(out_full)
        return out_full, out_var


class StructuralAnalytical(StructuralModel):
    def __init__(
        self, equations: Callable, parameter_names: list[str], variable_names: list[str]
    ):
        self.equations = equations
        output_names: list[str] = variable_names
        protocol_arms = ["identity"]
        tasks: list[str] = [
            output + "_" + protocol
            for protocol in protocol_arms
            for output in output_names
        ]
        # Map tasks to output names
        task_to_output = {
            output_name + "_" + protocol_arm: output_name
            for output_name in output_names
            for protocol_arm in protocol_arms
        }
        # Map task index to output index
        task_idx_to_output_idx = {
            tasks.index(k): output_names.index(v) for k, v in task_to_output.items()
        }
        # Map task to protocol arm
        task_to_protocol = {
            output_name + "_" + protocol_arm: protocol_arm
            for output_name in output_names
            for protocol_arm in protocol_arms
        }
        # Map task index to protocol arm
        task_idx_to_protocol = {tasks.index(k): v for k, v in task_to_protocol.items()}

        super().__init__(
            parameter_names,
            output_names,
            protocol_arms,
            tasks,
            task_idx_to_output_idx,
            task_idx_to_protocol,
        )

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        chunks: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (num_chains, nb_patients, nb_timesteps, nb_params) = X.shape
        nb_obs_per_chain = prediction_index[0].shape[0]
        prediction_index_expanded = (
            torch.arange(num_chains).repeat_interleave(nb_obs_per_chain),
            prediction_index[0].repeat(num_chains),
            prediction_index[1].repeat(num_chains),
            prediction_index[2].repeat(num_chains),
        )
        params = X.split(1, dim=-1)
        outputs = self.equations(*params)
        y = outputs[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        pred_var = torch.zeros_like(y)

        return y, pred_var
