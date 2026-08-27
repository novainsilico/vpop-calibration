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


def stochastic_approximation(
    previous: torch.Tensor, new: torch.Tensor, learning_rate: float
) -> torch.Tensor:
    """Perform stochastic approximation

    Args:
        previous (torch.Tensor): The current value of the tensor
        new (torch.Tensor): The target value of the tensor

    Returns:
        torch.Tensor: (1 - learning_rate) * previous + learning_rate * new
    """
    assert previous.shape == new.shape, (
        f"Wrong shape in stochastic approximation: {previous.shape}, {new.shape}"
    )

    stochastic_approx = (1 - learning_rate) * previous + learning_rate * new
    return stochastic_approx
