import pandera.pandas as pa
from typing import Optional


class ObsDataSchema(pa.DataFrameModel):
    id: str = pa.Field(coerce=True)
    output_name: str
    time: float = pa.Field(coerce=True)
    protocol_arm: str = pa.Field(default="identity")
    value: float = pa.Field(coerce=True)
    task: str
    hazard_name: Optional[str]
    event_time: Optional[float]
    event_status: Optional[bool]

    @pa.dataframe_parser
    def task_name(cls, df):
        if "protocol_arm" not in df.columns:
            df["protocol_arm"] = cls.to_schema().columns["protocol_arm"].default
        return df.assign(task=lambda r: r.output_name + "_" + r.protocol_arm)

    @pa.dataframe_check
    def check_cooccurrence_survival(cls, df: pa.typing.DataFrame) -> bool:
        time_is_defined = "event_time" in df.columns
        status_is_defined = "event_status" in df.columns
        hazard_is_defined = "hazard_name" in df.columns
        everything_is_defined = [time_is_defined, status_is_defined, hazard_is_defined]

        return all(everything_is_defined) or not any(everything_is_defined)


patientDataSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str, unique=True),
        "protocol_arm": pa.Column(str),
        "event_time": pa.Column(float, required=False),
        "event_status": pa.Column(bool, required=False),
        "hazard_name": pa.Column(str, required=False),
    },
    checks=pa.Check(
        lambda df: ("event_time" in df.columns) == ("event_status" in df.columns),
        name="Survival variable cooccurrence.",
        error="Columns 'event_time' and 'event_status' must either both exist or both be absent.",
    ),
)
