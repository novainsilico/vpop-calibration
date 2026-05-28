import pandera.pandas as pa
import pandas as pd
import torch

from vpop_calibration.utils import extend_schema
from vpop_calibration.config import device
from vpop_calibration.nlme_model.indexing import ObservationIndex, IndexedObservations

obsDataSchemaLong = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "output_name": pa.Column(str),
        "time": pa.Column(pd.Float64Dtype),
        "protocol_arm": pa.Column(str, default="identity"),
        "value": pa.Column(pd.Float64Dtype),
    },
    coerce=True,
    add_missing_columns=True,
)

patientDataSchema = pa.DataFrameSchema(
    {"id": pa.Column(str, unique=True), "protocol_arm": pa.Column(str, unique=True)}
)


class ObsData:
    def __init__(self, data: pa.typing.DataFrame):
        """Load and process an observed data set

        Args:
            data (pa.typing.DataFrame): The observed data. Should contain at least the columns ["id", "output_name", "time", "value"].
        """
        # Initial validation
        self.obs_schema = obsDataSchemaLong
        self.input_df = self.obs_schema.validate(data)
        self.patients: list[str] = self.input_df.id.drop_duplicates().to_list()

        # Create the patient data frame (id, protocol_arm and descriptors)
        patients_df_raw = self.input_df.drop(
            columns=["output_name", "time", "value"]
        ).drop_duplicates()
        self.descriptors_known: list[str] = patients_df_raw.columns.to_list()
        self.descriptors_known.remove("id")
        self.descriptors_known.remove("protocol_arm")
        self.patients_schema = extend_schema(
            patientDataSchema, self.descriptors_known, "float"
        )
        self.patients_df = self.patients_schema.validate(patients_df_raw)

        self.input_df = self.input_df.assign(
            task=lambda x: x.output_name + "_" + x.protocol_arm
        )

        self.full_obs = IndexedObservations(
            obs_index=ObservationIndex.from_dataframe(self.input_df),
            obs_values=torch.as_tensor(self.input_df["value"].to_list(), device=device),
        )
        self.global_timesteps = torch.tensor(
            self.full_obs.obs_index.time.ref_values, device=device
        )
        self.nb_global_timesteps = self.global_timesteps.shape[0]
        self.nb_total_observations = self.full_obs.obs_values.shape[0]
        self.observed_output_names = self.full_obs.obs_index.output_name.ref_values

        self.individual_observations: dict[str, IndexedObservations] = {}
        for p in self.patients:
            patient_data = self.input_df.loc[self.input_df["id"] == p]
            index_values_p = ObservationIndex.from_dataframe(patient_data)
            obs_values_p = torch.as_tensor(
                patient_data["value"].to_list(), device=device
            )
            self.individual_observations.update(
                {
                    p: IndexedObservations(
                        obs_index=index_values_p, obs_values=obs_values_p
                    )
                }
            )

    def init_pdk_values(self, pdk_names: list[str]) -> None:
        """Generate per-patient PDK tensors

        Once initialized they are stored in `self.patients_pdk[patient_id]` and `self.patients_pdk_full`.

        Args:
            pdk_names (list[str]): The name of the known parameters which are to be assembled as pdk. Must appear in the data set columns.
        """
        assert set(pdk_names) <= set(
            self.descriptors_known
        ), f"Unknown PDK: {set(pdk_names) - set(self.descriptors_known)}"
        self.pdk_names = pdk_names
        self.nb_pdk = len(pdk_names)
        self.patients_pdk = {}
        for patient in self.patients:
            if self.nb_pdk > 0:
                row = self.patients_df.loc[
                    self.patients_df["id"] == patient
                ].drop_duplicates()
                self.patients_pdk.update(
                    {
                        patient: torch.as_tensor(
                            row[self.pdk_names].values, device=device
                        )
                    }
                )
            else:
                self.patients_pdk.update({patient: torch.empty((1, 0), device=device)})
        # Store the full pdk tensor on the device
        self.patients_pdk_full = torch.cat(
            [self.patients_pdk[ind] for ind in self.patients]
        ).to(device)

    def remap_all_indexings(
        self,
        new_patient_ids: list | None = None,
        new_output_names: list | None = None,
        new_protocol_arms: list | None = None,
        new_tasks: list | None = None,
        new_times: list | None = None,
    ):
        args = (
            new_patient_ids,
            new_output_names,
            new_protocol_arms,
            new_tasks,
            new_times,
        )
        self.full_obs.obs_index = self.full_obs.obs_index.remap_observation_index(*args)
        for p in self.patients:
            self.individual_observations[p].obs_index = self.individual_observations[
                p
            ].obs_index.remap_observation_index(*args)
