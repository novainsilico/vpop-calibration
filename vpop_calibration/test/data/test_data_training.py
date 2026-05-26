from vpop_calibration.data.training import TrainingData

import pytest
import numpy as np
import pandas as pd
import torch


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
    df["value"] = np.abs(np_rng.normal())
    params = ["k1", "time"]
    return df, params


@pytest.mark.parametrize("include_protocol", [True, False])
@pytest.mark.parametrize("log_inputs", [[], ["k1"]])
@pytest.mark.parametrize("log_outputs", [[], ["s1"]])
def test_data_container(training_data, log_inputs, log_outputs):
    df, params = training_data
    ds = TrainingData(
        df,
        params,
        log_descriptors=log_inputs,
        log_outputs=log_outputs,
    )
    loader = ds.to_loader()
    x, y = next(iter(loader))

    normalize, unnormalize = ds.get_processing_functions()
    sample_inputs = torch.abs(torch.rand(3, ds.nb_descriptors))
    sample_outputs = torch.rand(3, 1)
    sample_task_indices = torch.randint(0, ds.nb_tasks, (3, 1))
    normalize(sample_inputs)
    unnormalize(sample_outputs, sample_task_indices)
