import torch
import pandas as pd
from typing import Callable, Optional

from .taskmap import TaskMap


class StructuralModel:
    def __init__(
        self,
        parameter_names: list[str],
        task_map: TaskMap,
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
        self.task_map = task_map

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Not implemented")


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

        protocol_overrides_set = set(protocol_design.drop(columns="protocol_arm"))

        # the parameters of the "equations" function which are NOT protocol overrides and NOT time, in this order
        parameter_names_without_protocol_overrides = [
            p
            for p in function_arguments
            if p not in protocol_overrides_set and p != "t"
        ]
        # the parameters of the "equations" function which are protocol overrides, in this order
        self.protocol_parameters = [
            p for p in function_arguments if p in protocol_overrides_set
        ]
        self.nb_protocol_overrides = len(self.protocol_parameters)

        # Ordered list of parameters that the NLME model expects to find in the function arguments
        input_parameters = (
            parameter_names_without_protocol_overrides
            + ["t"]
            + self.protocol_parameters
        )
        self.input_to_function_arg = [
            input_parameters.index(a) for a in function_arguments
        ]

        if self.nb_protocol_overrides > 0:
            self.task_protocol_tensor = torch.Tensor(
                [
                    protocol_design.loc[
                        protocol_design["protocol_arm"]
                        == task_map.task_idx_to_protocol[task_idx],
                        self.protocol_parameters,
                    ].values.squeeze(axis=0)
                    for task_idx, _ in enumerate(task_map.tasks)
                ]
            )
        else:
            self.task_protocol_tensor = torch.empty((len(task_map.tasks), 0))

        super().__init__(
            parameter_names=parameter_names_without_protocol_overrides,
            task_map=task_map,
        )

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_chains, nb_patients, nb_timesteps, nb_params = X.shape
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
            task_index.apply_(lambda i: self.task_map.task_idx_to_output_idx[i]).repeat(
                num_chains
            ),
        )
        # map the "columns" (in fact the last axis which corresponds to the parameters) of X_with_protocol_overrides
        # to the positions of the corresponding arguments in the signature of the "equations" function
        params = X_with_protocol_overrides[:, :, :, self.input_to_function_arg].split(
            1, dim=-1
        )
        outputs = self.equations(*params)
        y = outputs[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        pred_var = torch.zeros_like(y)
        return y, pred_var
