from vpop_calibration.pynlme.data import ObsData

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def obs_data() -> pd.DataFrame:
    protocol_arms = ["arm-A", "arm-B"]
    patients_arms = {"id": ["p1", "p2"], "protocol_arm": protocol_arms}
    outputs = ["s1", "s2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients_arms)
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = 0.0

    return df


def test_data_container(obs_data):
    df = obs_data
    ds = ObsData(df)

    torch.testing.assert_close(
        ds.n_tot_observations_per_output, torch.tensor([6, 6]), check_dtype=False
    )


def test_incomplete_data(obs_data, np_rng):
    df = obs_data
    df_boot = df.sample(frac=0.5, random_state=np_rng)
    ds = ObsData(df_boot)
