import pandas as pd
import pandera.pandas as pa
import random
import numpy as np
import torch
import uuid


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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reproducible_uuid4(seed=None):
    if seed is not None:
        random.seed(seed)
    return uuid.UUID(int=random.getrandbits(128), version=4)
