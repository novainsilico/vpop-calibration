import torch
import pandas as pd
import numpy as np
from typing import List
import uuid

from .model.gp import GP
from .ode import OdeModel


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
        self.parameter_names: List[str] = parameter_names
        self.nb_parameters: int = len(self.parameter_names)
        self.output_names: List[str] = output_names
        self.nb_outputs: int = len(self.output_names)
        self.protocols: List[str] = protocol_arms
        self.nb_protocols: int = len(self.protocols)
        self.tasks: List[str] = tasks
        self.task_idx_to_output_idx: dict[int, int] = task_idx_to_output_idx
        self.task_idx_to_protocol: dict[int, str] = task_idx_to_protocol

    def simulate(
        self, list_X: list[torch.Tensor], list_tasks: list[torch.LongTensor]
    ) -> list[torch.Tensor]:
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
        # self.compiled_predict = torch.compile(self.gp_model.predict_long_scaled) # not functional for the moment
        self.compiled_predict = self.gp_model.predict_long_scaled

    def simulate(
        self, list_X: list[torch.Tensor], list_tasks: list[torch.LongTensor]
    ) -> list[torch.Tensor]:
        # X must be ordered like parameter names from the GP
        # Concatenate all the inputs
        X_cat = torch.cat(list_X)
        task_cat = torch.LongTensor(torch.cat([torch.Tensor(t) for t in list_tasks]))
        chunk_sizes = [t.shape[0] for t in list_tasks]
        # Simulate the GP
        out_cat, _ = self.compiled_predict(X_cat, task_cat)
        # Split into individual chunks
        pred_list = list(torch.split(out_cat, chunk_sizes))
        return pred_list


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
        output_names: List[str] = self.ode_model.variable_names
        tasks: List[str] = [
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

        # List the structural model parameters: the protocol overrides are ignored
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
        self, list_X: list[torch.Tensor], list_tasks: list[torch.LongTensor]
    ) -> list[torch.Tensor]:
        # each X must be ordered like parameter names from the ode model

        input_df_list = []
        chunks_list = []
        for X, tasks in zip(list_X, list_tasks):
            temp_id = str(uuid.uuid4())
            # store the size of X for proper splitting
            chunks_list.append(X.shape[0])
            # Extract the parameters and time values
            params = X.detach().numpy()
            # Extract the task order
            task_index = tasks.detach().numpy()
            # Format the data inputs
            # This step is where the order of parameters is implicit
            input_df_temp = pd.DataFrame(
                data=params, columns=self.parameter_names + ["time"]
            )
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
        out_tensor = torch.Tensor(output_df["predicted_value"].values)
        # Split into chunks
        out_list = list(torch.split(out_tensor, chunks_list))
        return out_list
