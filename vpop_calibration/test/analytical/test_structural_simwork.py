from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
    StructuralSimwork,
)
from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.pynlme.schemas import ObsDataSchema

import torch
import pytest
import pandas as pd


@pytest.fixture
def dummy_simwork_model() -> SimworkModelBinding:
    model = SimworkModelBinding(
        id="modelid", inputs=["foo", "bar"], outputs=["out_1", "out_2"]
    )
    return model


@pytest.fixture
def observations() -> ObservationIndex:
    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": "identity"})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["out_1", "out_2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))
    return obs_index


def test_analytical_simwork(dummy_simwork_model, observations):
    struct_model = StructuralSimwork(model=dummy_simwork_model)
    X = torch.tensor(
        [[[[0, 1, 0], [0, 1, 1], [0, 1, 2]], [[1, 0, 0], [1, 0, 1], [1, 0, 2]]]]
    )
    out = struct_model.simulate(X=X, prediction_index=observations)
