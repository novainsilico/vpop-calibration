from typing import Optional
import pandas as pd
import pandera.pandas as pa
import numpy as np
import torch

from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.utils import extend_schema
from vpop_calibration.config import device


def run_haskell_model_placeholder(
    patients: pd.DataFrame, time_points: list[float], outputs: list[str]
) -> np.ndarray:
    nb_patients = patients.shape[0]
    nb_timesteps = len(time_points)
    nb_outputs = len(outputs)
    dummy_output = np.zeros((nb_patients, nb_timesteps, nb_outputs))
    return dummy_output


class SimworkModelBinding:
    def __init__(self, id: str, inputs: list[str], outputs: list[str]):
        self.id = id
        self.inputs = inputs
        self.outputs = outputs

    def run(self, vpop: pd.DataFrame, time: list[float]) -> np.ndarray:
        outputs = run_haskell_model_placeholder(
            patients=vpop, time_points=time, outputs=self.outputs
        )
        # Expect an array of size (nb_patients, nb_timesteps, nb_outputs)
        return outputs


class StructuralSimwork(StructuralModel):
    def __init__(
        self,
        model: SimworkModelBinding,
        protocol_design: Optional[pd.DataFrame] = None,
    ):
        self.model = model

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
        # the parameters of the simwork model which are NOT protocol overrides
        parameter_names_without_protocol_overrides = [
            p for p in model.inputs if p not in protocol_overrides
        ]
        # the parameters of the simwork model which are protocol overrides
        self.protocol_parameters = [p for p in model.inputs if p in protocol_overrides]
        self.nb_protocol_overrides = len(self.protocol_parameters)

        # Ordered list of parameters that the NLME model expects to find in the function arguments
        input_parameters = (
            parameter_names_without_protocol_overrides + self.protocol_parameters
        )
        self.nb_parameters = len(input_parameters)
        self.input_to_function_arg = [input_parameters.index(a) for a in model.inputs]

        self.task_names = [
            output + "_" + protocol
            for output in self.model.outputs
            for protocol in protocol_arms
        ]

        super().__init__(
            parameter_names=parameter_names_without_protocol_overrides,
            output_names=self.model.outputs,
            protocol_arms=protocol_arms,
            task_names=self.task_names,
        )

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: ObservationIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nb_chains, nb_patients, nb_timesteps, _ = X.shape
        # Create a mapping from patient index to protocol index
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

        # Remove time from the X tensor (keeping only time 0 for patient overrides)
        X_without_time = X[:, :, 0, :-1].squeeze(2)
        # now size (nb_chains, nb_patients, nb_parameters)

        # expand protocol overrides tensor to (num_chains, nb_patients, nb_protocol_overrides)
        protocol_overrides_expanded = protocol_overrides.unsqueeze(0).expand(
            nb_chains, nb_patients, -1
        )
        X_with_protocol_overrides = torch.cat(
            (X_without_time, protocol_overrides_expanded), dim=-1
        )
        assert X_with_protocol_overrides.shape[2] == self.nb_parameters
        # melt the tensor to be 2d, and assemble it in a dataframe - assuming parameters are in the correct order
        vpop = pd.DataFrame(
            data=X_with_protocol_overrides.view(-1, self.nb_parameters).numpy(),
            columns=self.parameter_names,
        )
        # Assemble the time values
        time = prediction_index.time.raw_values.to_list()
        # Run the model
        outputs = self.model.run(vpop=vpop, time=time)
        outputs_tensor = torch.as_tensor(outputs, device=device)
        # Pivot to a wide tensor
        outputs_wide = outputs_tensor.view(nb_chains, nb_patients, nb_timesteps, -1)
        # Build the 4d tensor index for row observations
        nb_obs_per_chain = prediction_index.id.index_values.shape[0]
        prediction_index_expanded = (
            torch.arange(nb_chains).repeat_interleave(nb_obs_per_chain),
            prediction_index.id.index_values.repeat(nb_chains),
            prediction_index.time.index_values.repeat(nb_chains),
            prediction_index.output_name.index_values.repeat(nb_chains),
        )
        y = outputs_wide[prediction_index_expanded].view(nb_chains, nb_obs_per_chain)
        pred_var = torch.zeros_like(y)
        return y, pred_var
