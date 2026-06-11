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


def run_haskell_model_placeholder(
    patients: pd.DataFrame, time_points: list[float], outputs: list[str]
) -> np.ndarray:
    nb_patients = patients.shape[0]
    nb_timesteps = len(time_points)
    nb_outputs = len(outputs)
    dummy_output = np.zeros((nb_patients, nb_timesteps, nb_outputs))
    return dummy_output


@pytest.fixture
def dummy_simwork_model() -> SimworkModelBinding:
    model = SimworkModelBinding(
        haskell_callback=run_haskell_model_placeholder,
        inputs=["foo", "bar", "baz"],
        outputs=["out_1", "out_2"],
    )
    return model


def test_analytical_simwork_no_protocol_override(dummy_simwork_model):
    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": "identity"})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["out_1", "out_2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))

    struct_model = StructuralSimwork(model=dummy_simwork_model)
    X = torch.tensor(
        [
            [  # foo bar baz time
                [[0, 1, 2, 0], [0, 1, 2, 1], [0, 1, 2, 2]],  # p1
                [[1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 2]],  # p2
            ]
        ]
    )
    out = struct_model.simulate(X=X, prediction_index=obs_index)


def test_analytical_simwork_one_protocol_override(dummy_simwork_model):
    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": ["arm-B", "arm-A"]})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["out_1", "out_2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))

    protocol_design = pd.DataFrame({"protocol_arm": ["arm-A", "arm-B"], "baz": [0, 1]})

    struct_model = StructuralSimwork(
        model=dummy_simwork_model, protocol_design=protocol_design
    )
    X = torch.tensor(
        # foo bar time
        [[[[0, 1, 0], [0, 1, 1], [0, 1, 2]], [[1, 0, 0], [1, 0, 1], [1, 0, 2]]]]
    )
    out = struct_model.simulate(X=X, prediction_index=obs_index)
