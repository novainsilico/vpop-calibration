import pandera.pandas as pa
import pandas as pd
import torch

from vpop_calibration.utils import extend_schema
from vpop_calibration.config import device

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
        # Initial validation
        self.obs_schema = obsDataSchemaLong
        self.input_df = self.obs_schema.validate(data)
        self.patients = self.input_df.id.drop_duplicates().to_list()
        self.nb_patients = len(self.patients)

        # Create the patient data frame (id, protocol_arm and descriptors)
        patients_df_raw = self.input_df.drop(
            columns=["output_name", "time", "value"]
        ).drop_duplicates()
        self.descriptors_known = patients_df_raw.columns.to_list()
        self.descriptors_known.remove("id")
        self.descriptors_known.remove("protocol_arm")
        self.patients_schema = extend_schema(
            patientDataSchema, self.descriptors_known, "float"
        )
        self.patients_df = self.patients_schema.validate(patients_df_raw)
        # Gather the reference lists for indexing:
        # These are sorted list of unique elements for the three columns [protocol_arm, output_name and time]
        self.protocol_arms, self.output_names, self.global_time_steps = tuple(
            map(
                lambda col: self.input_df[col]
                .drop_duplicates()
                .sort_values()
                .to_list(),
                ["protocol_arm", "output_name", "time"],
            )
        )
        self.nb_global_time_steps = len(self.global_time_steps)
        self.time_steps_tensor = torch.as_tensor(self.global_time_steps, device=device)

        # Create indexing columns
        # Avoiding code repetition with a config tuple list. Each element of the list is:
        # (name of the added indexing column, name of the existing column, corresponding indexing list)
        indexings = [
            ("patient_index", "id", self.patients),
            ("output_index", "output_name", self.output_names),
            ("protocol_index", "protocol_arm", self.protocol_arms),
            ("timestep_index", "time", self.global_time_steps),
        ]
        self.indexing_columns = [indexing[0] for indexing in indexings]

        for index_name, index_variable, indexing_list in indexings:
            self.input_df[index_name] = self.input_df[index_variable].apply(
                lambda x: indexing_list.index(x)
            )
        # Assemble the per-patient observations
        self.observations = {}
        for p in self.patients:
            patient_data = self.input_df.loc[self.input_df["id"] == p]
            # Column extraction as separate tensors is now immediate with list comprehension:
            self.observations.update(
                {
                    p: {
                        "prediction_index": tuple(
                            torch.as_tensor(patient_data[col].to_list(), device=device)
                            for col in self.indexing_columns
                        ),
                        "value": torch.as_tensor(
                            patient_data["value"].to_list(), device=device
                        ),
                    }
                }
            )
        # Assemble the full prediction index by concatenating separate tensors into one per index
        # Important: the prediction index then contains one LongTensor per indexing column
        # (patient_index, output_index, protocol_index, timestep_index) -> todo: replace with a dataclass
        self.prediction_index = tuple(
            map(
                torch.cat,
                zip(*[self.observations[p]["prediction_index"] for p in self.patients]),
            )
        )
        # Assemble the full observation tensor
        self.full_observations = torch.cat(
            [self.observations[p]["value"] for p in self.patients]
        )
        self.nb_total_observations = self.full_observations.shape[0]

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
