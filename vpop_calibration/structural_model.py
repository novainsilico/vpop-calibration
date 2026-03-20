import torch
import pandas as pd
import numpy as np
import uuid
from typing import Callable, Optional
import itertools

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
        self.training_ranges = {}
        training_samples = self.gp_model.data.full_df_raw[self.parameter_names]
        train_min = training_samples.min(axis=0)
        train_max = training_samples.max(axis=0)
        for param in self.parameter_names:
            self.training_ranges.update(
                {param: {"low": train_min[param], "high": train_max[param]}}
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
        # Simulate the GP
        X_vertical = X.view(-1, nb_params)
        out_cat, var_cat = self.gp_model.predict_wide_scaled(X_vertical)
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
        output_names: list[str],
    ):
        self.ode_model = ode_model
        protocol_arms = protocol_design["protocol_arm"].drop_duplicates().to_list()
        self.protocol_design = protocol_design
        assert set(output_names) <= set(self.ode_model.variable_names)
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
            tasks.index(task): self.ode_model.variable_names.index(output)
            for task, output in task_to_output.items()
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

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        chunks: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patient_index_full, rows_full, tasks_full = prediction_index
        list_rows = torch.split(rows_full, chunks)
        list_tasks = torch.split(tasks_full, chunks)
        list_id = [id[0].item for id in torch.split(patient_index_full, chunks)]

        output_list = []
        for chain_X in X:  # iterate through the individual chains
            # Separate the individual patients
            list_X = chain_X.split(1, dim=0)
            input_df_list = []
            # Iterate through patients
            for patient_descriptors, patient_rows, patient_tasks, p_id in zip(
                list_X, list_rows, list_tasks, list_id
            ):
                temp_id = str(uuid.uuid4())
                # Extract the parameters and time values
                params = (
                    patient_descriptors.squeeze(0)
                    .index_select(0, patient_rows)
                    .cpu()
                    .detach()
                    .numpy()
                )
                # Extract the task order
                task_index = patient_tasks.cpu().detach().numpy()
                # Format the data inputs
                # This step is where the order of parameters is implicit
                input_df_temp = pd.DataFrame(
                    data=params, columns=self.parameter_names + ["time"]
                )
                # The passed params include the _global_ time steps
                # Filter the time steps that we actually want for this patient
                input_df_temp = input_df_temp.iloc[patient_rows.cpu().numpy()]
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
                    ).reset_index(drop=True)
                input_df_list.append(input_df_temp.reset_index(drop=True))
            full_input = pd.concat(input_df_list)
            # Simulate the ODE model
            output_df = self.ode_model.simulate_model(full_input).reset_index(drop=True)
            # Place the simulated results in the right rows
            id_cols = ["id", "protocol_arm", "time", "output_name"]
            output_ordered = full_input[id_cols].merge(
                output_df[[*id_cols, "predicted_value"]],
                on=id_cols,
            )
            # Convert back to tensor
            out_tensor = torch.as_tensor(
                output_ordered["predicted_value"].values,
                device=device,
            )
            output_list.append(out_tensor)
        out_full = torch.stack(output_list, dim=0).to(device)
        out_var = torch.zeros_like(out_full)
        return out_full, out_var


class StructuralAnalytical(StructuralModel):
    def __init__(
        self,
        equations: Callable,
        variable_names: list[str],
        protocol_design: Optional[pd.DataFrame] = None,
    ):
        # Extracts the ORDERED list of arguments from the "equations" function signature
        # This will be used to map the columns of the input tensor to their corresponding positional arguments
        # when invoking "equations"
        function_arguments = list(
            # 1. ".__code__.co_varnames" is a tuple of ALL the variables (including local variables) referenced in
            # the function, STARTING with the function arguments which is the only reason this works.
            # 2. # ".__code__.co_argcount" is the number of function arguments
            # See the "inspect" doc for more details
            # https://docs.python.org/3/library/inspect.html
            equations.__code__.co_varnames[: equations.__code__.co_argcount]
        )
        self.equations = equations
        if protocol_design is None:
            protocol_design = pd.DataFrame({"protocol_arm": ["identity"]})
        protocol_arms = protocol_design["protocol_arm"].drop_duplicates().to_list()
        protocol_arm_idx = {
            arm: arm_index for arm_index, arm in enumerate(protocol_arms)
        }
        variable_idx = {
            variable_name: variable_index
            for variable_index, variable_name in enumerate(variable_names)
        }
        protocol_overrides_set = set(protocol_design.drop(columns="protocol_arm"))
        # the parameters of the "equations" function which are NOT protocol overrides and NOT time, in this order
        parameter_names_without_protocol_overrides = [
            p
            for p in function_arguments
            if p not in protocol_overrides_set and p != "t"
        ]
        # the parameters of the "equations" function which are protocol overrides, in this order
        protocol_parameters = [
            p for p in function_arguments if p in protocol_overrides_set
        ]
        self.nb_protocol_overrides = len(protocol_parameters)

        # time is special cases in "nlme.py" and always comes after the structural parameters
        input_parameters = (
            parameter_names_without_protocol_overrides + ["t"] + protocol_parameters
        )
        parameter_index = {p: index for (index, p) in enumerate(input_parameters)}
        self.input_tensor_column_index_to_function_parameter_index = [
            parameter_index[a] for a in function_arguments
        ]

        # The tensor of protocol overrides with shape nb_arms x nb_protocol_params
        # np.atleast_2d() ensures the array has at least shape (1, 0) if there are not protocol overrides
        # (e.g. in the "identity" protocol)
        if len(protocol_parameters) == 0:
            protocol_tensor = np.zeros((len(protocol_arms), 0))
        else:
            protocol_tensor = np.atleast_2d(
                np.transpose(
                    np.array([protocol_design[p] for p in protocol_parameters])
                )
            )
        # Building the tasks names and the various index maps in one go
        tasks = []
        task_idx_to_output_idx = {}
        task_idx_to_protocol = {}
        task_protocol_overrides = []
        for task_index, (arm, output) in enumerate(
            itertools.product(protocol_arms, variable_names)
        ):
            task_name = f"{output}_{arm}"
            tasks.append(task_name)
            task_idx_to_output_idx[task_index] = variable_idx[output]
            task_idx_to_protocol[task_index] = arm
            arm_idx = protocol_arm_idx[arm]
            task_protocol_overrides.append(np.atleast_1d(protocol_tensor[arm_idx, :]))
        # the protocol overrides are stored in tensor of shape (num_tasks, nb_protocol_params)
        # such that it can later be efficiently appended to the input tensor "X" in simulate()
        self.task_protocol_tensor = torch.as_tensor(
            np.array(task_protocol_overrides), device=device
        )

        super().__init__(
            parameter_names_without_protocol_overrides,
            variable_names,
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
        patient_index, timestep_index, task_index = prediction_index
        protocol_overrides = self.task_protocol_tensor[task_index].reshape(
            num_chains, nb_patients, nb_timesteps, self.nb_protocol_overrides
        )
        X_with_protocol_overrides = torch.concat((X, protocol_overrides), dim=-1)

        nb_obs_per_chain = patient_index.shape[0]
        prediction_index_expanded = (
            torch.arange(num_chains).repeat_interleave(nb_obs_per_chain),
            patient_index.repeat(num_chains),
            timestep_index.repeat(num_chains),
            task_index.apply_(lambda i: self.task_idx_to_output_idx[i]).repeat(
                num_chains
            ),
        )
        # map the "columns" (in fact the last axis which corresponds to the parameters) of X_with_protocol_overrides
        # to the positions of the corresponding arguments in the signature of the "equations" function
        params = X_with_protocol_overrides[
            :, :, :, self.input_tensor_column_index_to_function_parameter_index
        ].split(1, dim=-1)
        outputs = self.equations(*params)
        y = outputs[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        pred_var = torch.zeros_like(y)
        return y, pred_var
