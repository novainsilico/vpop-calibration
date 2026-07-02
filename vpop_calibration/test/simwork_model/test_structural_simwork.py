from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
    StructuralSimwork,
)
from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.pynlme.schemas import ObsDataSchema

import torch
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def dummy_simwork_model() -> SimworkModelBinding:
    model = SimworkModelBinding(
        path_to_model="vpop_calibration/test/simwork_model/assets/model.json",
        path_to_solving_options="vpop_calibration/test/simwork_model/assets/options.json",
        inputs=["k_12", "k_21"],
        outputs=["A0", "A1", "A2"],
    )
    return model


def test_analytical_simwork_no_protocol_override(dummy_simwork_model):
    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": "identity"})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["A0", "A1", "A2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))

    struct_model = StructuralSimwork(model=dummy_simwork_model)
    X = torch.tensor(
        [
            [  # k_12 k_21 time
                [[0, 1, 0], [0, 1, 1], [0, 1, 2]],  # p1
                [[1, 0, 0], [1, 0, 1], [1, 0, 2]],  # p2
            ]
        ]
    )
    out = struct_model.simulate(X=X, prediction_index=obs_index)


def test_analytical_simwork_two_chains(dummy_simwork_model):
    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": "identity"})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["A0", "A1", "A2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))

    struct_model = StructuralSimwork(model=dummy_simwork_model)
    X = torch.tensor(
        [  # Chain 1
            [  # k_12 k_21 time
                [[0, 1, 0], [0, 1, 1], [0, 1, 2]],  # p1
                [[1, 0, 0], [1, 0, 1], [1, 0, 2]],  # p2
            ],
            # Chain 2
            [  # k_12 k_21 time
                [[0, 1, 0], [0, 1, 1], [0, 1, 2]],  # p1
                [[1, 0, 0], [1, 0, 1], [1, 0, 2]],  # p2
            ],
        ]
    )
    out = struct_model.simulate(X=X, prediction_index=obs_index)


def test_analytical_simwork_one_protocol_override(dummy_simwork_model):
    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": ["arm-B", "arm-A"]})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["A0", "A1", "A2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))

    protocol_design = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"], "k_21": [0, 1]})

    categorical_attributes = pd.DataFrame(
        {"id": ["p1", "p2"], "foo": ["truc", "muche"]}
    )
    struct_model = StructuralSimwork(
        model=dummy_simwork_model,
        protocol_design=protocol_design,
        categorical_attributes=categorical_attributes,
    )
    X = torch.tensor(
        # k_12 time
        [[[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]]]
    )
    out = struct_model.simulate(X=X, prediction_index=obs_index)
