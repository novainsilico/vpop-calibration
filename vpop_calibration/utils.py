import pandas as pd
import pandera.pandas as pa
import random
import numpy as np
import torch
import uuid

torch.use_deterministic_algorithms(True)

def extend_schema(
    schema: pa.DataFrameSchema, column_list: list[str], type: str
) -> pa.DataFrameSchema:
    """Add user-specified columns to the training data schema."""
    if not column_list:
        return schema
    else:
        return schema.add_columns(
            {col: pa.Column(type, default=pd.NA, coerce=True) for col in column_list}
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reproducible_uuid4(seed=None):
    if seed is not None:
        random.seed(seed)
    return uuid.UUID(int=random.getrandbits(128), version=4)

def known_patient_uuid4(patient_number: int):
    """
    It returns a patient UUID which is reproductible and readable based on the patient number

    The numbers are "encoded" in the uuid in the latest fields, so for example:

    >>> known_patient_uuid(5)
    UUID('00000000-0000-0000-0000-000000000005')

    Note that the value are encoded in hex, so values >9 can be confusing to read.
    """
    return uuid.UUID(int=patient_number, version=4)


def from_seconds_time_scale(time_unit: str) -> float:
    match time_unit:
        case "second":
            return 1
        case "minute":
            return 60
        case "hour":
            return 60 * 60
        case "day":
            return 60 * 60 * 24
        case "week":
            return 60 * 60 * 24 * 7
        case _:
            raise ValueError(f"Unsupported time unit: {time_unit}")


def time_scale_and_label(time_unit: str | None) -> tuple[str, float]:
    if time_unit is None:
        xlabel = "Time"
        time_scale = 1
    else:
        xlabel = f"Time ({time_unit})"
        time_scale = from_seconds_time_scale(time_unit)
    return (xlabel, time_scale)
