from torch.utils.data import Dataset
import pandera.pandas as pa
import pandas as pd
import numpy as np
from typing import Callable
import torch

from vpop_calibration.utils import (
    join_if_two,
    TaskMap,
    extend_schema,
)
from vpop_calibration.config import device

trainingDataSchemaLong = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "output_name": pa.Column(str),
        "protocol_arm": pa.Column(str, default="identity"),
        "value": pa.Column(pd.Float64Dtype),
    },
    coerce=True,
    add_missing_columns=True,
    strict=True,
)

trainingDataSchemaWide = pa.DataFrameSchema(
    {"id": pa.Column(str)},
    coerce=True,
    add_missing_columns=False,
    strict=True,
)


def pivot_input_data(data_in: pd.DataFrame, descriptors: list[str]) -> pd.DataFrame:
    """Pivot and reorder columns from a data frame to feed to the model

    This method is used at initialization on the training data frame), and when plotting the model performance against existing data.

    Args:
        data_in (pd.DataFrame): Input data frame, containing the following columns
        - `id`: patient id
        - one column per descriptor, the same descriptors as self.parameter_names should be present
        - `output_name`: the name of the output
        - `protocol_arm`: the name of the protocol arm
        - `value`: the observed value

    Returns:
        pd.DataFrame: Pivotted dataframe with one column per task (`outputName_protocolArm`), and one row per observation
    """

    # Pivot the data set
    reshaped_df = data_in.pivot(
        index=["id"] + descriptors,
        columns=["output_name", "protocol_arm"],
        values="value",
    ).reset_index()
    nested_column_names = reshaped_df.columns.to_list()
    flat_column_names = list(map(join_if_two, nested_column_names))
    reshaped_df.columns = flat_column_names

    return reshaped_df


def normalize_dataframe(
    data_in: pd.DataFrame, ignore: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalize a data frame with respect to its mean and std, ignoring certain columns, and output the corresponding mean and std."""
    selected_columns = data_in.columns.difference(ignore)
    norm_data = data_in
    mean = data_in[selected_columns].mean()
    std = data_in[selected_columns].std()
    norm_data[selected_columns] = (norm_data[selected_columns] - mean) / std
    return norm_data, mean, std


class TrainingData(Dataset):
    def __init__(
        self,
        data: pa.typing.DataFrame,
        descriptors: list[str],
        log_descriptors: list[str] = [],
        log_outputs: list[str] = [],
    ):
        # Preliminary data validation on long data frame
        self.descriptors: list[str] = descriptors
        self.nb_descriptors: int = len(descriptors)
        # Add the descriptors to the validation schema
        self.long_schema = extend_schema(trainingDataSchemaLong, descriptors, "float")
        validated_df = self.long_schema.validate(data)

        # Extract all relevant metadata
        self.output_names: list[str] = validated_df.output_name.unique().tolist()
        self.nb_outputs: int = len(self.output_names)

        self.log_descriptors: list[str] = log_descriptors
        self.log_descriptors_idx: list[int] = [
            self.descriptors.index(desc) for desc in self.log_descriptors
        ]
        self.log_outputs: list[str] = log_outputs
        self.log_outputs_idx: list[int] = [
            self.output_names.index(out) for out in self.log_outputs
        ]

        self.patients: list[str] = validated_df.id.unique().tolist()
        self.nb_patients: int = len(self.patients)
        self.protocol_arms: list[str] = validated_df.protocol_arm.unique().tolist()

        # Create maps between tasks and output/protocol arm
        self.task_map: TaskMap = TaskMap(self.protocol_arms, self.output_names)
        self.nb_tasks: int = len(self.task_map.tasks)
        self.log_tasks_idx: list[int] = [
            idx
            for idx in self.task_map.task_idx_to_output_idx
            if self.task_map.task_idx_to_output_idx[idx] in self.log_outputs_idx
        ]
        self.log_tasks: list[str] = [
            self.task_map.tasks[idx] for idx in self.log_tasks_idx
        ]

        # Pivot the input data to a wide format
        pivoted_df = pivot_input_data(validated_df, self.descriptors)
        # Apply log transform where necessary
        pivoted_df.loc[:, self.log_descriptors + self.log_tasks] = pivoted_df.loc[
            :, self.log_descriptors + self.log_tasks
        ].apply(np.log)

        # Validate the wide data
        self.wide_schema = extend_schema(
            trainingDataSchemaWide, self.descriptors + self.task_map.tasks, "float"
        )
        final_df = self.wide_schema.validate(pivoted_df)
        self.patients_idx: torch.LongTensor = torch.LongTensor(
            final_df.id.apply(lambda p_id: self.patients.index(p_id)).values,
            device=device,
        )
        # Normalize the inputs and outputs
        self.normalized_df, mean, std = normalize_dataframe(final_df, ["id"])
        # Store normalizing values as tensors
        self.input_mean, self.input_std = (
            torch.as_tensor(mean[self.descriptors].values, device=device),
            torch.as_tensor(std[self.descriptors].values, device=device),
        )
        self.output_mean, self.output_std = (
            torch.as_tensor(mean[self.task_map.tasks].values, device=device),
            torch.as_tensor(std[self.task_map.tasks].values, device=device),
        )

        self.X_full: torch.Tensor = torch.as_tensor(
            self.normalized_df[self.descriptors].values, device=device
        )
        self.Y_full: torch.Tensor = torch.as_tensor(
            self.normalized_df[self.task_map.tasks].values, device=device
        )
        self.data = []
        for i, _ in enumerate(self.patients):
            rows = torch.tensor(self.patients_idx == i, device=device)
            x = self.X_full[rows, :]
            y = self.Y_full[rows, :]
            self.data.append((x, y))

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self):
        return self.nb_patients

    def collate_fn(self, batch):
        x = torch.cat([item[0] for item in batch])
        y = torch.cat([item[1] for item in batch])
        return x, y

    def to_loader(self, batch_size: int | None = None):
        if batch_size is None:
            use_batch_size = len(self)
        else:
            use_batch_size = batch_size
        return torch.utils.data.DataLoader(dataset=self, batch_size=use_batch_size)

    def get_processing_functions(self) -> tuple[Callable, Callable]:
        @torch.compile
        def normalize_inputs(inputs: torch.Tensor) -> torch.Tensor:
            """Normalize new inputs provided to the model as a tensor. The columns of the input tensor should be the same as [self.descriptors]"""
            X = inputs.to(device)
            X[:, self.log_descriptors_idx] = torch.log(X[:, self.log_descriptors_idx])
            mean = self.input_mean
            std = self.input_std
            norm_X = (X - mean) / std

            return norm_X

        @torch.compile
        def unnormalize_outputs(
            data: torch.Tensor, task_indices: torch.LongTensor
        ) -> torch.Tensor:
            """Unnormalize long outputs (one row per task) from the model."""
            rescaled_data = data
            for task_idx, task in enumerate(self.task_map.tasks):
                log_task = task in self.log_tasks
                mask = torch.tensor(task_indices == task_idx, device=device).bool()
                rescaled_data[mask] = (
                    rescaled_data[mask] * self.output_std[task_idx]
                    + self.output_mean[task_idx]
                )
                if log_task:
                    rescaled_data[mask] = torch.exp(rescaled_data[mask])
            return rescaled_data

        return normalize_inputs, unnormalize_outputs
