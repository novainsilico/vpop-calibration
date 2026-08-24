from typing import NamedTuple
import torch
from pydantic import BaseModel, ConfigDict
import pandas as pd
from pandera.typing import DataFrame

from vpop_calibration.pynlme.schemas import ObsDataSchema
from vpop_calibration.config import device


class TensorIndexing(NamedTuple):
    index_values: torch.Tensor
    ref_values: list
    raw_values: pd.Series


def remap_single_index(
    input_index: torch.Tensor, mapping: dict[int, int]
) -> torch.Tensor:
    assert input_index.dim() == 1, (
        f"Unexpected indexing tensor dimension {input_index.dim()}"
    )
    new_index = torch.as_tensor([mapping[int(i.item())] for i in input_index])
    return new_index


def remap_indexed_values(
    source_index: TensorIndexing,
    dest_ref_values: list | None,
) -> TensorIndexing:
    if dest_ref_values is None:
        return source_index

    assert set(source_index.ref_values) <= set(dest_ref_values), (
        f"Incompatible indexing lists provided:\nSource: {source_index.ref_values}\nDestination: {dest_ref_values}"
    )
    mapping = {
        i: dest_ref_values.index(val) for i, val in enumerate(source_index.ref_values)
    }
    new_index_values = remap_single_index(source_index.index_values, mapping)
    new_index = TensorIndexing(
        index_values=new_index_values,
        ref_values=dest_ref_values,
        raw_values=source_index.raw_values,
    )
    return new_index


class DataIndex(NamedTuple):
    """Utility class to store and manipulate tensor indexings"""

    # The field names correspond to actual column names in ObsData
    id: TensorIndexing
    output_name: TensorIndexing
    protocol_arm: TensorIndexing
    task: TensorIndexing
    time: TensorIndexing

    @classmethod
    def from_dataframe(cls, df: DataFrame[ObsDataSchema]) -> "DataIndex":
        """Instantiate an DataIndex from an observed dataframe."""
        indexes = []
        for field in cls._fields:
            raw_values = df[field]
            ref_values = raw_values.drop_duplicates().sort_values().tolist()
            indexed_values = torch.tensor(
                raw_values.apply(lambda x: ref_values.index(x)).values, device=device
            )
            indexes.append(
                TensorIndexing(
                    index_values=indexed_values,
                    ref_values=ref_values,
                    raw_values=raw_values,
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
    ) -> "DataIndex":
        """Given an existing indexing, remap to new (compatible) reference values."""
        replacement_map = [
            (self.id, new_patient_ids),
            (self.output_name, new_output_names),
            (self.protocol_arm, new_protocol_arms),
            (self.task, new_tasks),
            (self.time, new_times),
        ]
        new_obs_index = DataIndex(
            *tuple(map(lambda args: remap_indexed_values(*args), replacement_map))
        )
        return new_obs_index


class SurvivalOutputs(NamedTuple):
    log_hazard: str
    cumulative_hazard: str


class ObservationsDataSet(BaseModel):
    obs_index: DataIndex
    obs_values: torch.Tensor
    survival_outputs: SurvivalOutputs | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_pandas(
        self,
        prediction: torch.Tensor | None = None,
    ) -> pd.DataFrame:
        nb_obs = self.obs_values.shape[0]
        if prediction is not None:
            assert prediction.dim() == 2, (
                "Don't squeeze predictions before turning them into a dataframe."
            )
            assert prediction.shape[0] == 1, (
                "Cannot convert batched predictions to dataframe."
            )
            assert prediction.shape[1] == nb_obs, (
                f"Incompatible number of self ({nb_obs}) and predictions ({prediction.shape[1]})"
            )

        id_col = self.obs_index.id.raw_values
        output_name_col = self.obs_index.output_name.raw_values
        protocol_arm_col = self.obs_index.protocol_arm.raw_values
        time_col = self.obs_index.time.raw_values
        value_col = self.obs_values.detach().cpu().numpy()
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
            df_long["predicted_value"] = prediction.squeeze(0).detach().cpu().numpy()

        return df_long
