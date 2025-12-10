import torch
import pandas as pd
import numpy as np
import uuid

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
        list_X: list[torch.Tensor],
        list_rows: list[torch.LongTensor],
        list_tasks: list[torch.LongTensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
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
        list_X: list[torch.Tensor],
        list_rows: list[torch.LongTensor],
        list_tasks: list[torch.LongTensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # X must be ordered like parameter names from the GP
        # Concatenate all the inputs
        X_cat = torch.cat(list_X).to(device)
        chunk_sizes = [X.shape[0] for X in list_X]
        # Simulate the GP
        out_cat, var_cat = self.gp_model.predict_wide_scaled(X_cat)
        # Split into individual chunks
        pred_wide_list = torch.split(out_cat, chunk_sizes)
        var_wide_list = torch.split(var_cat, chunk_sizes)
        pred_list = []
        var_list = []
        for pred, var_wide, rows, cols in zip(
            pred_wide_list, var_wide_list, list_rows, list_tasks
        ):
            y = (
                pred.index_select(0, rows)
                .gather(1, cols.view(-1, 1))
                .squeeze(1)
                .to(device)
            )
            var = (
                var_wide.index_select(0, rows)
                .gather(1, cols.view(-1, 1))
                .squeeze(1)
                .to(device)
            )
            pred_list.append(y)
            var_list.append(var)
        return pred_list, var_list


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
        list_X: list[torch.Tensor],
        list_rows: list[torch.LongTensor],
        list_tasks: list[torch.LongTensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # each X must be ordered like parameter names from the ode model

        input_df_list = []
        chunks_list = []
        for X, rows, tasks in zip(list_X, list_rows, list_tasks):
            temp_id = str(uuid.uuid4())
            # store the size of X for proper splitting
            chunks_list.append(rows.shape[0])
            # Extract the parameters and time values
            params = X.index_select(0, rows).cpu().detach().numpy()
            # Extract the task order
            task_index = tasks.cpu().detach().numpy()
            # Format the data inputs
            # This step is where the order of parameters is implicit
            input_df_temp = pd.DataFrame(
                data=params, columns=self.parameter_names + ["time"]
            )
            # The passed params include the _global_ time steps
            # Filter the time steps that we actually want for this patient
            input_df_temp = input_df_temp.iloc[rows.numpy()]
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
        out_tensor = torch.Tensor(output_df["predicted_value"].values, device=device)
        out_var = torch.zeros_like(out_tensor, device=device)
        # Split into chunks
        out_list = list(torch.split(out_tensor, chunks_list))
        var_list = list(torch.split(out_var, chunks_list))
        return out_list, var_list
