import pytest
import numpy as np
import pandas as pd

from vpop_calibration.model.data import TrainingDataSet


@pytest.fixture
def training_data(np_rng, include_protocol):
    # Use this fixture for testing GP training
    patients = {"id": ["p1", "p2"], "k1": [1.0, 2.0]}
    protocol_arms = ["arm-A", "arm-B"]
    outputs = ["s1", "s2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    if include_protocol:
        df = df.merge(
            pd.DataFrame(protocol_arms, columns=["protocol_arm"]), how="cross"
        )
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np_rng.normal()
    params = ["k1", "time"]
    return df, params


@pytest.mark.parametrize("training_proportion", [0.5, 1.0])
@pytest.mark.parametrize("log_inputs", [[], ["k1"]])
@pytest.mark.parametrize("log_outputs", [[], ["s1"]])
@pytest.mark.parametrize("data_already_normalized", [False, True])
@pytest.mark.parametrize("include_protocol", [True, False])
def test_loading(
    training_data, training_proportion, log_inputs, log_outputs, data_already_normalized
):
    df, params = training_data
    TrainingDataSet(
        training_df=df,
        descriptors=params,
        log_inputs=log_inputs,
        log_outputs=log_outputs,
        training_proportion=training_proportion,
        data_already_normalized=data_already_normalized,
    )
