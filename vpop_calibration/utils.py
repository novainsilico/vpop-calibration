import pandas as pd
import pandera.pandas as pa


def join_if_two(tup: str, sep: str = "_") -> str:
    """Utility to concatenate a tuple of strings with a separator. Used to flatten dataframe index after pivotting."""
    if tup[0] == "":
        return tup[1]
    elif tup[1] == "":
        return tup[0]
    else:
        return sep.join(tup)


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
