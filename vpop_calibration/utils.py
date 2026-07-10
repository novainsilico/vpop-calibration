import pandas as pd
import pandera.pandas as pa


def extend_schema(
    schema: pa.DataFrameSchema, column_list: list[str], type: str
) -> pa.DataFrameSchema:
    """Add user-specified columns to the training data schema."""
    if not column_list:
        return schema
    else:
        return schema.add_columns(
            {col: pa.Column(type, default=pd.NA) for col in column_list}
        )


def pivot_table_to_vpop(df: pd.DataFrame) -> pd.DataFrame:
    pivotted = df.melt(
        id_vars=["id", "id_ref"],
        var_name="descriptor_name",
        value_name="descriptor_value",
    )
    return pivotted
