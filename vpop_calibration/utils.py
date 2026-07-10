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
