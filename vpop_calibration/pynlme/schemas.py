import pandera.pandas as pa
import pandas as pd


class ObsDataSchema(pa.DataFrameModel):
    id: str = pa.Field(coerce=True)
    output_name: str
    time: float = pa.Field(coerce=True)
    protocol_arm: str = pa.Field(default="identity")
    value: float = pa.Field(coerce=True)
    task: str

    @pa.dataframe_parser
    def task_name(cls, df):
        return df.assign(task=lambda r: r.output_name + "_" + r.protocol_arm)


patientDataSchema = pa.DataFrameSchema(
    {"id": pa.Column(str, unique=True), "protocol_arm": pa.Column(str)}
)
