import torch
import pandas as pd

from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.config import device
from vpop_calibration.pynlme.schemas import ObsDataSchema, patientDataSchema


class StructuralModel:
    def __init__(
        self,
        parameter_names: list[str],
        output_names: list[str],
        protocol_arms: list[str],
        task_names: list[str],
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
        self.output_names: list[str] = output_names
        self.protocol_arms: list[str] = protocol_arms
        self.task_names: list[str] = task_names

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: ObservationIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Not implemented")

    def simulate_from_df(
        self, vpop: pd.DataFrame, obs_df: pd.DataFrame
    ) -> pd.DataFrame:
        obs_df_validated = ObsDataSchema.validate(obs_df)
        obs_index = ObservationIndex.from_dataframe(obs_df_validated)
        # The patients in the vpop data frame have no reason to be in the correct order, so enforce the ordering here
        patient_data_validated = patientDataSchema.validate(vpop)
        patients_order = pd.DataFrame({"id": obs_index.id.ref_values})
        order_patient_data = patients_order.merge(
            patient_data_validated, on="id", how="left"
        )

        patient_descriptors = torch.as_tensor(
            order_patient_data[self.parameter_names].values, device=device
        )
        nb_patients, nb_parameters = patient_descriptors.shape
        timesteps = obs_index.time.index_values
        nb_timesteps = timesteps.shape[0]
        X = torch.cat(
            (
                patient_descriptors.unsqueeze(-2).expand(-1, nb_timesteps, -1),
                timesteps.unsqueeze(0).unsqueeze(-1).expand(nb_patients, -1, 1),
            ),
            dim=-1,
        ).unsqueeze(0)
        assert X.shape == (1, nb_patients, nb_timesteps, nb_parameters + 1)

        output, _ = self.simulate(X=X, prediction_index=obs_index)
        output_df = obs_df
        output_df["predicted_value"] = output.squeeze(0).detach().numpy()

        return output_df
