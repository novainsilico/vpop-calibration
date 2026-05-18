from vpop_calibration.data.observed import ObsData, TaskMap

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def obs_data(np_rng) -> tuple[pd.DataFrame, TaskMap]:
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
@pytest.mark.parametrize("provide_task_map", [True, False])
def test_data_container(obs_data, boostrap_ratio, np_rng, provide_task_map):
    df, task_map_pre = obs_data
    df_boot = df.sample(frac=boostrap_ratio, random_state=np_rng)
    if provide_task_map:
        task_map = task_map_pre
    else:
        task_map = None
    ds = ObsData(df_boot, task_map)
    loader = ds.to_dataloader()
    pred_index, y = next(iter(loader))
