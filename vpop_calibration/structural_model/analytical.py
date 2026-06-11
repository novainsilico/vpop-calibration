from typing import Callable, Optional
import pandas as pd
import pandera.pandas as pa
import torch

from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.utils import extend_schema
from vpop_calibration.config import device


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

        protocol_overrides = protocol_design.drop(
            columns="protocol_arm"
        ).columns.to_list()

        base_protocol_schema = pa.DataFrameSchema(
            {
                "protocol_arm": pa.Column(str, default="identity"),
            },
            coerce=True,
        )
        self.protocol_schema = extend_schema(
            base_protocol_schema, column_list=protocol_overrides, type="float"
        )
        self.protocol_design = self.protocol_schema.validate(protocol_design)
        protocol_arms = protocol_design["protocol_arm"].drop_duplicates().tolist()
        # Create the protocol overrides tensor
        # Indexed by protocol index:
        # protocol_overrides_tensor[protocol_index,:] = parameter overrides for this protocol
        self.protocol_overrides_tensor = torch.as_tensor(
            protocol_design.drop_duplicates()
            .set_index("protocol_arm")
            .loc[protocol_arms]
            .reset_index()
            .drop(columns="protocol_arm")
            .values,
            device=device,
        )
        # the parameters of the "equations" function which are NOT protocol overrides and NOT time, in this order
        parameter_names_without_protocol_overrides = [
            p for p in function_arguments if p not in protocol_overrides and p != "t"
        ]
        # the parameters of the "equations" function which are protocol overrides, in this order
        self.protocol_parameters = [
            p for p in function_arguments if p in protocol_overrides
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

        self.task_names = [
            output + "_" + protocol
            for output in variable_names
            for protocol in protocol_arms
        ]

        super().__init__(
            parameter_names=parameter_names_without_protocol_overrides,
            output_names=variable_names,
            protocol_arms=protocol_arms,
            task_names=self.task_names,
        )

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: ObservationIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_chains, nb_patients, nb_timesteps, nb_params = X.shape
        map_patient_to_protocol = {
            t[0].item(): t[1].item()
            for t in (
                torch.stack(
                    (
                        prediction_index.id.index_values,
                        prediction_index.protocol_arm.index_values,
                    )
                )
                .unique(dim=-1)
                .unbind(dim=-1)
            )
        }
        actual_protocol_indices = [
            map_patient_to_protocol[p_ind] for p_ind in range(nb_patients)
        ]
        protocol_overrides = self.protocol_overrides_tensor[actual_protocol_indices, :]
        # protocol overrides: size (nb_patients, nb_protocol_overrides)
        # expand it to (num_chains, nb_patients, nb_timesteps, nb_protocol_overrides)
        protocol_overrides_expanded = (
            protocol_overrides.unsqueeze(0)
            .unsqueeze(-2)
            .expand(num_chains, nb_patients, nb_timesteps, -1)
        )
        X_with_protocol_overrides = torch.cat((X, protocol_overrides_expanded), dim=-1)

        nb_obs_per_chain = prediction_index.id.index_values.shape[0]
        prediction_index_expanded = (
            torch.arange(num_chains).repeat_interleave(nb_obs_per_chain),
            prediction_index.id.index_values.repeat(num_chains),
            prediction_index.time.index_values.repeat(num_chains),
            prediction_index.output_name.index_values.repeat(num_chains),
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
