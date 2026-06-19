from numpy import inf

from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
    model_output_adapter,
    TimeseriesOutput,
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


def test_handle_payload(dummy_simwork_model):
    timepoints = [0, 1, 2]
    model_output_json = {
        "p1": (
            timepoints,
            [
                {"id": "A0", "unit": "mg", "values": [0, 1, 2]},
                {"id": "A1", "unit": "mg", "values": [0, 1, 2]},
                {"id": "A2", "unit": "mg", "values": [0, 1, 2]},
            ],
        ),
        "p2": (
            timepoints,
            [
                {"id": "A0", "unit": "mg", "values": [0, 1, 2]},
                {"id": "A1", "unit": "mg", "values": [0, 1, 2]},
                {"id": "A2", "unit": "mg", "values": [0, 1, 2]},
            ],
        ),
        "p3": None, # simwork returns None when it cannot simulate, for instance if the ODE solver fails
    }
    validated_output = model_output_adapter.validate_python(model_output_json)
    
    parsed_output = dummy_simwork_model.parse_output_to_pandas(validated_output, timepoints)
    expected_output = pd.DataFrame(
        {
            "id": ["p1", "p1", "p1", "p2", "p2", "p2", "p3", "p3", "p3"],
            "time": timepoints * 3,
            "A0": [0, 1, 2, 0, 1, 2, inf, inf, inf],
            "A1": [0, 1, 2, 0, 1, 2, inf, inf, inf],
            "A2": [0, 1, 2, 0, 1, 2, inf, inf, inf],
        }
    )

    pd.testing.assert_frame_equal(
        parsed_output, expected_output, check_dtype=False, check_names=False
    )
    # We don't check the `names` attributes, as the index has a different name here, but the actual column names are actually validated
