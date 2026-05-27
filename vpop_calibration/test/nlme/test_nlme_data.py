from vpop_calibration.nlme_model.data import ObsData

import pytest
import numpy as np
import pandas as pd


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


@pytest.mark.parametrize("boostrap_ratio", [1.0, 0.5])
def test_data_container(obs_data, boostrap_ratio, np_rng):
    df = obs_data
    df_boot = df.sample(frac=boostrap_ratio, random_state=np_rng)
    ds = ObsData(df_boot)
