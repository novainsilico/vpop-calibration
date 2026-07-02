from typing import Optional
import pandas as pd
import pandera.pandas as pa
import numpy as np
import torch
from pydantic import BaseModel, TypeAdapter
import itertools
import subprocess
import tempfile
import uuid
import json

from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.utils import extend_schema
from vpop_calibration.config import device


class TimeseriesOutput(BaseModel):
    id: str
    unit: str
    values: list[float]


PatientOutput = tuple[list[float], list[TimeseriesOutput]]
ModelOutput = dict[str, Optional[PatientOutput]]
model_output_adapter = TypeAdapter(ModelOutput)


def nix_run_command(
    executable: str,
    model_path: str,
    solving_options_path: str,
    outputs: list[str],
    times: list[float],
    vpop_path: str,
) -> list[str]:

    concat_outputs = list(
        itertools.chain.from_iterable(["--output", o] for o in outputs)
    )
    cmd = [
        executable,
        "--model",
        model_path,
        "--explicit-time",
        ",".join(str(t) for t in times),
        "--options",
        solving_options_path,
        *concat_outputs,
        "--vpop",
        vpop_path,
    ]
    return cmd


class SimworkModelBinding:
    def __init__(
        self,
        path_to_model: str,
        path_to_solving_options: str,
        inputs: list[str],
        outputs: list[str],
    ):
        self.model_path = path_to_model
        self.solving_options = path_to_solving_options
        self.inputs = inputs
        self.outputs = outputs
        self.nb_outputs = len(outputs)

        build_result = subprocess.run(
            [
                "nix",
                "build",
                ".#simwork.legacyPackages.x86_64-linux.perf.scripts.run-model-simple",
                "--print-out-paths",
            ],
            capture_output=True,
        )
        if build_result.returncode != 0:
            raise RuntimeError(build_result.stderr)
        self.executable = (
            build_result.stdout.decode().strip("\n") + "/bin/scripts.run-model-simple"
        )

    def run(
        self,
        vpop: pd.DataFrame,
        time: list[float],
        categorical_attributes: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        vpop_json = self.df_to_json_vpop(
            vpop_df=vpop, categorical_attributes=categorical_attributes
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete_on_close=False
        ) as tmp_file:
            vpop_path = tmp_file.name
            tmp_file.write(json.dumps(vpop_json))
            tmp_file.close()
            result = subprocess.run(
                nix_run_command(
                    executable=self.executable,
                    model_path=self.model_path,
                    solving_options_path=self.solving_options,
                    outputs=self.outputs,
                    times=time,
                    vpop_path=vpop_path,
                ),
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Fatal error: {result.stderr}")
        filt_result = subprocess.run(
            ["jq", "-s", "last"],
            input=result.stdout,
            capture_output=True,
            text=True,
        )
        model_output = model_output_adapter.validate_json(filt_result.stdout)
        output_df = self.parse_output_to_pandas(model_output, time)
        return output_df

    def df_to_json_vpop(
        self, vpop_df: pd.DataFrame, categorical_attributes: pd.DataFrame | None = None
    ) -> dict:
        vpop = {
            "patients": [
                {
                    "patientIndex": row["id"],
                    "patientCategoricalAttributes": (
                        [
                            {"id": param, "val": param}
                            for param in categorical_attributes.loc[
                                categorical_attributes["id"] == row["id"]
                            ]
                        ]
                        if categorical_attributes is not None
                        else []
                    ),
                    "patientAttributes": [
                        {"id": param, "val": row[param]} for param in self.inputs
                    ],
                }
                for index, row in vpop_df.iterrows()
            ]
        }
        return vpop

    def parse_output_to_pandas(
        self, simwork_output: ModelOutput, timepoints: list[float]
    ) -> pd.DataFrame:
        df_list = []
        for patient_id, patient_data in simwork_output.items():
            if patient_data is None:
                # Solving failed for this patient so let's fill the output timeseries Inf values
                for output_name in self.outputs:
                    temp_df = pd.DataFrame(
                        {
                            "id": patient_id,
                            "time": timepoints,
                            "output_name": output_name,
                            "value": [np.inf] * len(timepoints),
                        }
                    )
                    df_list.append(temp_df)
            else:
                for timeseries in patient_data[1]:
                    temp_df = pd.DataFrame(
                        {
                            "id": patient_id,
                            "time": patient_data[0],
                            "output_name": timeseries.id,
                            "value": timeseries.values,
                        }
                    )
                    df_list.append(temp_df)
        full_df = pd.concat(df_list)
        full_df_wide = full_df.pivot(
            index=["id", "time"], columns="output_name", values="value"
        ).reset_index()
        return full_df_wide


class StructuralSimwork(StructuralModel):
    def __init__(
        self,
        model: SimworkModelBinding,
        protocol_design: Optional[pd.DataFrame] = None,
        categorical_attributes: pd.DataFrame | None = None,
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
        # the parameters of the simwork model which are NOT protocol overrides
        parameter_names_without_protocol_overrides = [
            p for p in model.inputs if p not in protocol_overrides
        ]
        # the parameters of the simwork model which are protocol overrides
        self.protocol_parameters = [p for p in model.inputs if p in protocol_overrides]
        self.nb_protocol_overrides = len(self.protocol_parameters)

        # Ordered list of parameters that the NLME model expects to find in the function arguments
        self.input_parameters = (
            parameter_names_without_protocol_overrides + self.protocol_parameters
        )
        self.nb_parameters = len(self.input_parameters)

        # Create the protocol overrides tensor
        # Indexed by protocol index:
        # protocol_overrides_tensor[protocol_index,:] = parameter overrides for this protocol
        self.protocol_overrides_tensor = torch.as_tensor(
            protocol_design.drop_duplicates()
            .set_index("protocol_arm")
            .loc[protocol_arms, self.protocol_parameters]
            .reset_index()
            .drop(columns="protocol_arm")
            .values,
            device=device,
        )

        self.task_names = [
            output + "_" + protocol
            for output in self.model.outputs
            for protocol in protocol_arms
        ]

        self.categorical_attributes = categorical_attributes

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
        X_without_time = X[:, :, 0, :-1]
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
            columns=self.input_parameters,
        )
        # Add a temp patient id, to cover the fact that a single patient is simulated on each chain
        temporary_ids = [str(uuid.uuid4()) for _ in range(vpop.shape[0])]
        vpop["id"] = temporary_ids

        if self.categorical_attributes is not None:
            # Create a numpy array indexing patients from 0 to nb_patients, looping over chains
            patient_indexing = (
                torch.arange(nb_patients)
                .unsqueeze(0)
                .expand(nb_chains, -1)
                .reshape(-1)
                .numpy()
            )
            # Map the `real` id and the temporary id
            actual_patient_ids = pd.DataFrame(
                {
                    "id": np.array(prediction_index.id.ref_values)[patient_indexing],
                    "tmp_id": temporary_ids,
                }
            )

            cat_with_temp_id = (
                self.categorical_attributes.merge(actual_patient_ids, on="id")
                .drop(columns=["id"])
                .rename(columns={"tmp_id": "id"})
            )
        else:
            cat_with_temp_id = None

        # Assemble the time values
        time = prediction_index.time.ref_values
        # Run the model
        outputs_df = self.model.run(
            vpop=vpop, time=time, categorical_attributes=cat_with_temp_id
        )
        patient_id_ordered = pd.DataFrame({"id": temporary_ids})
        outputs_df_ordered = patient_id_ordered.merge(outputs_df, on="id", how="left")
        outputs_tensor = torch.as_tensor(
            outputs_df_ordered[self.output_names].values, device=device
        )
        # Pivot to a wide tensor
        outputs_wide = outputs_tensor.view(
            nb_chains,
            nb_patients,
            nb_timesteps,
            self.model.nb_outputs,
        )
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
