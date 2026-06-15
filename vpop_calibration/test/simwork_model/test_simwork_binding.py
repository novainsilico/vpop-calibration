from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
)

import pytest
import pandas as pd


@pytest.fixture
def dummy_simwork_model() -> SimworkModelBinding:
    model = SimworkModelBinding(
        path_to_model="vpop_calibration/test/simwork_model/assets/model.json",
        path_to_solving_options="vpop_calibration/test/simwork_model/assets/options.json",
        inputs=["k_12", "k_21"],
        outputs=["A0", "A1", "A2"],
    )
    return model


def test_simwork_binding(dummy_simwork_model):
    vpop = pd.DataFrame({"id": ["p1", "p2"], "k_12": [0, 0], "k_21": [1, 0]})
    json = dummy_simwork_model.df_to_json_vpop(vpop_df=vpop)

    time = [0, 1, 2]
    out = dummy_simwork_model.run(vpop=vpop, time=time)
