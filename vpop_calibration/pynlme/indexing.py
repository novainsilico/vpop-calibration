from typing import NamedTuple
import torch
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np


class IndexedValues(NamedTuple):
    index_values: torch.Tensor
    ref_values: list


def remap_single_index(
    input_index: torch.Tensor, mapping: dict[int, int]
) -> torch.Tensor:
    assert (
        input_index.dim() == 1
    ), f"Unexpected indexing tensor dimension {input_index.dim()}"
    return input_index.apply_(mapping.get)


def remap_indexed_values(
    source_index: IndexedValues,
    dest_ref_values: list | None,
) -> IndexedValues:
    if dest_ref_values is None:
        return source_index

    assert set(source_index.ref_values) <= set(
        dest_ref_values
    ), f"Incompatible indexing lists provided:\nSource: {source_index.ref_values}\nDestination: {dest_ref_values}"
    mapping = {
        i: dest_ref_values.index(val) for i, val in enumerate(source_index.ref_values)
    }
    new_index_values = remap_single_index(source_index.index_values, mapping)
    new_index = IndexedValues(
        index_values=new_index_values,
        ref_values=dest_ref_values,
    )
    return new_index


class ObservationIndex(NamedTuple):
    """Utility class to store and manipulate tensor indexings"""

    # The field names correspond to actual column names in ObsData
    id: IndexedValues
    output_name: IndexedValues
    protocol_arm: IndexedValues
    task: IndexedValues
    time: IndexedValues

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """Instantiate an ObservationIndex from an observed dataframe."""
        indexes = []
        for field in cls._fields:
            # This only works if df contains one column per field in the observation index
            ref_values = df[field].drop_duplicates().sort_values().tolist()
            indexed_values = torch.tensor(
                df[field].apply(lambda x: ref_values.index(x)).values
            )
            indexes.append(
                IndexedValues(
                    index_values=indexed_values,
                    ref_values=ref_values,
                )
            )

        prediction_index = cls(*indexes)
        return prediction_index

    def remap_observation_index(
        self,
        new_patient_ids: list | None = None,
        new_output_names: list | None = None,
        new_protocol_arms: list | None = None,
        new_tasks: list | None = None,
        new_times: list | None = None,
    ) -> "ObservationIndex":
        """Given an existing indexing, remap to new (compatible) reference values."""
        replacement_map = [
            (self.id, new_patient_ids),
            (self.output_name, new_output_names),
            (self.protocol_arm, new_protocol_arms),
            (self.task, new_tasks),
            (self.time, new_times),
        ]
        new_obs_index = ObservationIndex(
            *tuple(map(lambda args: remap_indexed_values(*args), replacement_map))
        )
        return new_obs_index


class IndexedObservations(BaseModel):
    obs_index: ObservationIndex
    obs_values: torch.Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_pandas(
        self,
        prediction: torch.Tensor | None = None,
    ) -> pd.DataFrame:
        nb_obs = self.obs_values.shape[0]
        if prediction is not None:
            assert (
                prediction.dim() == 2
            ), "Don't squeeze predictions before turning them into a dataframe."
            assert (
                prediction.shape[0] == 1
            ), "Cannot convert batched predictions to dataframe."
            assert (
                prediction.shape[1] == nb_obs
            ), f"Incompatible number of self ({nb_obs}) and predictions ({prediction.shape[1]})"

        id_col = np.array(self.obs_index.id.ref_values)[
            self.obs_index.id.index_values.cpu().numpy().astype(int)
        ]
        output_name_col = np.array(self.obs_index.output_name.ref_values)[
            self.obs_index.output_name.index_values.cpu().numpy().astype(int)
        ]
        protocol_arm_col = np.array(self.obs_index.protocol_arm.ref_values)[
            self.obs_index.protocol_arm.index_values.cpu().numpy().astype(int)
        ]
        time_col = np.array(self.obs_index.time.ref_values)[
            self.obs_index.time.index_values.cpu().numpy().astype(int)
        ]
        value_col = self.obs_values.cpu().numpy()
        df_long = pd.DataFrame(
            {
                "id": id_col,
                "output_name": output_name_col,
                "protocol_arm": protocol_arm_col,
                "time": time_col,
                "value": value_col,
            }
        )
        if prediction is not None:
            df_long["predicted_value"] = prediction.squeeze(0).cpu().numpy()

        return df_long
