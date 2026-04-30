from vpop_calibration.data.observed import ObsData, TaskMap

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def obs_data(np_rng) -> tuple[pd.DataFrame, TaskMap]:
    # Use this fixture for testing GP training
    patients = {"id": ["p1", "p2"]}
    protocol_arms = ["arm-A", "arm-B"]
    outputs = ["s1", "s2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(protocol_arms, columns=["protocol_arm"]), how="cross")
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal())

    task_map = TaskMap(protocol_arms, outputs)
    return df, task_map


@pytest.mark.parametrize("boostrap_ratio", [1.0, 0.5])
def test_data_container(obs_data, boostrap_ratio, np_rng):
    df, task_map = obs_data
    df_boot = df.sample(frac=boostrap_ratio, random_state=np_rng)
    ds = ObsData(df_boot, task_map)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=len(ds), collate_fn=ds.collate_fn
    )
    pred_index, y = next(iter(loader))
