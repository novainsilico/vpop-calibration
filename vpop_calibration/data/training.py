from torch.utils.data import Dataset
import pandera.pandas as pa
import pandas as pd

from .utils import join_if_two, create_tasks_maps, normalize_dataframe

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
    {
        "id": pa.Column(str),
    },
    coerce=True,
    add_missing_columns=True,
    strict=True,
)


def extend_schema(
    schema: pa.DataFrameSchema, column_list: list[str], type: str
) -> pa.DataFrameSchema:
    """Add user-specified columns to the training data schema."""
    return schema.add_columns(
        {col: pa.Column(type, default=pd.NA) for col in column_list}
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


class TrainingData(Dataset):
    def __init__(
        self,
        data: pa.typing.DataFrame,
        descriptors: list[str],
        log_descriptors: list[str] = [],
        log_outputs: list[str] = [],
    ):
        self.descriptors = descriptors
        self.long_schema = extend_schema(trainingDataSchemaLong, descriptors, "float")
        validated_df = self.long_schema.validate(data)

        self.output_names = validated_df.output_name.unique().tolist()
        self.nb_outputs = len(self.output_names)

        self.log_descriptors = log_descriptors
        self.log_outputs = log_outputs

        self.patients = validated_df.id.unique().tolist()
        self.nb_patients = len(self.patients)
        self.protocol_arms = validated_df.protocol_arm.unique().tolist()

        self.tasks, self.task_idx_to_output_idx, self.task_idx_to_protocol = (
            create_tasks_maps(self.protocol_arms, self.output_names)
        )

        pivoted_df = pivot_input_data(validated_df, self.descriptors)
        self.wide_schema = extend_schema(trainingDataSchemaWide, self.tasks, "float")
        final_df = self.wide_schema.validate(pivoted_df)
        normalized_df, mean, std = normalize_dataframe(final_df, ["id"])

    def __getitem__(self, index):
        return super().__getitem__(index)
