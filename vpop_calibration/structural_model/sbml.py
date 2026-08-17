import pandas as pd
from typing import Hashable
import roadrunner
import pandera.pandas as pa
import torch
import uuid
import yaml

from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.utils import extend_schema
from vpop_calibration.config import default_dtype, device
from vpop_calibration.pynlme.indexing import DataIndex


def simulate_rr_single_patient(
    input: tuple[str, dict[Hashable, float]],
    rr: roadrunner.RoadRunner,
    time_steps: list[float],
    outputs: list[str],
) -> pd.DataFrame:
    patient_id, patient_overrides = input
    rr.resetToOrigin()
    # Change the patient overrides
    rr.setValues(patient_overrides)
    # Reset the floating species to their init value
    rr.reset()
    try:
        out = rr.simulate(times=time_steps, selections=outputs)
    except Exception:
        raise RuntimeError(f"Solving failed for {patient_overrides}")

    patient_df = pd.DataFrame(data=out, columns=outputs)
    patient_df["time"] = time_steps
    patient_df["id"] = patient_id
    return patient_df


class StructuralSbml(StructuralModel):
    def __init__(
        self,
        model_path: str,
        inputs: list[str],
        outputs: list[str],
        solving_options_path: str | None = None,
        protocol_design: pd.DataFrame | None = None,
    ):
        self.rr = roadrunner.RoadRunner(model_path)
        self.valid_ids = self.rr.keys()
        invalid_inputs = [o for o in inputs if o not in self.valid_ids]
        if invalid_inputs:
            raise ValueError(
                f"The following inputs are not part of the SBML model: {invalid_inputs}"
            )
        invalid_outputs = [o for o in outputs if o not in self.valid_ids]
        if invalid_outputs:
            raise ValueError(
                f"The following outputs are not part of the SBML model: {invalid_outputs}"
            )

        if solving_options_path is not None:
            with open(solving_options_path, "r") as f:
                config = yaml.safe_load(f)
            integrator = self.rr.getIntegrator()

            for key, val in config.get("settings", {}).items():
                integrator.setSetting(key, val)

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
            p for p in inputs if p not in protocol_overrides
        ]
        # the parameters of the simwork model which are protocol overrides
        self.protocol_parameters = [p for p in inputs if p in protocol_overrides]
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
            dtype=default_dtype,
        )

        self.task_names = [
            output + "_" + protocol for output in outputs for protocol in protocol_arms
        ]

        super().__init__(
            parameter_names=parameter_names_without_protocol_overrides,
            output_names=outputs,
            protocol_arms=protocol_arms,
            task_names=self.task_names,
        )

    def assemble_numeric_vpop(
        self,
        X: torch.Tensor,
        prediction_index: DataIndex,
    ) -> pd.DataFrame:
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
            data=X_with_protocol_overrides.view(-1, self.nb_parameters)
            .detach()
            .cpu()
            .numpy(),
            columns=self.input_parameters,
        )
        # Add a temp patient id, to cover the fact that a single patient is simulated on each chain
        temporary_ids = [str(uuid.uuid4()) for _ in range(vpop.shape[0])]
        vpop["id"] = temporary_ids
        return vpop

    def run_vpop(self, vpop: pd.DataFrame, time_steps: list[float]) -> pd.DataFrame:
        simulation_inputs = [
            (
                row["id"],
                row.drop("id").to_dict(),
            )
            for _, row in vpop.iterrows()
        ]
        full_out = []
        for inputs in simulation_inputs:
            full_out.append(
                simulate_rr_single_patient(
                    inputs, rr=self.rr, time_steps=time_steps, outputs=self.output_names
                )
            )
        output_df = pd.concat(full_out)
        return output_df

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: DataIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nb_chains, nb_patients, nb_timesteps, _ = X.shape
        vpop = self.assemble_numeric_vpop(X, prediction_index)
        temporary_ids = vpop["id"]
        # Assemble the time values
        time = prediction_index.time.ref_values
        # Run the model
        outputs_df = self.run_vpop(vpop=vpop, time_steps=time)
        patient_id_ordered = pd.DataFrame({"id": temporary_ids})
        outputs_df_ordered = patient_id_ordered.merge(outputs_df, on="id", how="left")
        outputs_tensor = torch.as_tensor(
            outputs_df_ordered[self.output_names].values,
            device=device,
            dtype=default_dtype,
        )
        # Pivot to a wide tensor
        outputs_wide = outputs_tensor.view(
            nb_chains,
            nb_patients,
            nb_timesteps,
            self.nb_outputs,
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
