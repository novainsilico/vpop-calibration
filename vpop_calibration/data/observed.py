from torch.utils.data import Dataset
import pandera.pandas as pa
import pandas as pd
import torch

from .utils import TaskMap, extend_schema
from ..utils import device

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

patientDataSchema = pa.DataFrameSchema({"id": pa.Column(str, unique=True)})


class ObsData(Dataset):
    def __init__(self, data: pa.typing.DataFrame, task_map: TaskMap | None = None):
        self.obs_schema = obsDataSchemaLong
        # initial validation
        self.input_df = self.obs_schema.validate(data)
        self.patients = self.input_df.id.drop_duplicates().to_list()
        self.nb_patients = len(self.patients)

        # Create the patient data frame (id and descriptors)
        patients_df_raw = self.input_df.drop(
            columns=["output_name", "time", "value", "protocol_arm"]
        ).drop_duplicates()
        self.descriptors_known = patients_df_raw.columns.to_list()
        self.descriptors_known.remove("id")
        self.patients_schema = extend_schema(
            patientDataSchema, self.descriptors_known, "float"
        )
        self.patients_df = self.patients_schema.validate(patients_df_raw)
        # Gather the observed outputs and protocols
        self.protocol_arms = self.input_df["protocol_arm"].drop_duplicates().to_list()
        self.output_names = self.input_df["output_name"].drop_duplicates().to_list()
        # Create task column
        self.input_df["task"] = self.input_df[["output_name", "protocol_arm"]].apply(
            lambda r: "_".join(r), axis=1
        )
        if task_map is not None:
            self.task_map = task_map
            # Validate against the provided task map
            self.task_map.validate_tasks(self.protocol_arms, self.output_names)
        else:
            # Create the task map
            self.task_map = TaskMap(
                protocol_arms=self.protocol_arms, output_names=self.output_names
            )

        # Create indexing columns
        self.input_df["task_index"] = self.input_df["task"].apply(
            lambda task: self.task_map.tasks.index(task)
        )
        self.input_df["output_index"] = self.input_df["output_name"].apply(
            lambda output: self.task_map.output_names.index(output)
        )
        # Common list of time steps
        self.global_time_steps = (
            self.input_df["time"].drop_duplicates().sort_values().to_list()
        )
        self.input_df["t_index"] = self.input_df["time"].apply(
            lambda t: self.global_time_steps.index(t)
        )
        self.input_df["patient_index"] = self.input_df["id"].apply(
            lambda p: self.patients.index(p)
        )

        self.observations = []
        for p in self.patients:
            patient_data = self.input_df.loc[self.input_df["id"] == p]
            p_index = torch.as_tensor(
                patient_data["patient_index"].values, device=device
            )
            task_index = torch.as_tensor(
                patient_data["task_index"].values, device=device
            )
            time_index = torch.as_tensor(patient_data["t_index"].values, device=device)
            observed_value = torch.as_tensor(
                patient_data["value"].values, device=device
            )
            self.observations.append(
                ((p_index, task_index, time_index), observed_value)
            )

    def __getitem__(
        self, index
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.observations[index]

    def __len__(self):
        return self.nb_patients

    def collate_fn(self, batch):
        p_index = torch.cat([item[0][0] for item in batch])
        task_index = torch.cat([item[0][1] for item in batch])
        time_index = torch.cat([item[0][2] for item in batch])
        y = torch.cat([item[1] for item in batch])
        return (p_index, task_index, time_index), y

    def to_dataloader(self, batch_size: int | None = None):
        if batch_size is None:
            use_batch_size = len(self)
        else:
            use_batch_size = batch_size
        return torch.utils.data.DataLoader(
            dataset=self, batch_size=use_batch_size, collate_fn=self.collate_fn
        )
