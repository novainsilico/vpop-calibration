import pandas as pd
import torch
import pytest

from vpop_calibration.structural_model.sbml import StructuralSbml
from vpop_calibration.pynlme.indexing import ObservationIndex, ObsDataSchema


def test_sbml_model_wrong_inputs():
    file = "vpop_calibration/test/sbml/assets/model.xml"

    with pytest.raises(Exception):
        _model = StructuralSbml(
            model_path=file, inputs=["incorrect_input"], outputs=["A0"]
        )


def test_sbml_model_wrong_outputs():
    file = "vpop_calibration/test/sbml/assets/model.xml"
    with pytest.raises(Exception):
        _model = StructuralSbml(
            model_path=file, inputs=[], outputs=["incorrect_output"]
        )


def test_sbml_model():
    file = "vpop_calibration/test/sbml/assets/model.xml"

    df = (
        pd.DataFrame({"id": ["p1", "p2"], "protocol_arm": ["arm-B", "arm-A"]})
        .merge(pd.DataFrame({"time": [0, 1, 2]}), how="cross")
        .merge(pd.DataFrame({"output_name": ["A0", "A1", "A2"]}), how="cross")
    )
    df["value"] = 0.0
    obs_index = ObservationIndex.from_dataframe(ObsDataSchema.validate(df))

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "k__21": [0, 1]}
    )
    struct_model = StructuralSbml(
        model_path=file,
        inputs=["k__12", "k__21"],
        outputs=["A0", "A1", "A2"],
        protocol_design=protocol_design,
    )

    X = torch.tensor(
        # k_12 time
        [[[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]]]
    )
    nb_chains, nb_patients, nb_timesteps, nb_params = X.shape
    assert nb_chains == 1
    assert nb_patients == 2
    assert nb_timesteps == 3
    assert nb_params == 2
    vpop = struct_model.assemble_numeric_vpop(X=X, prediction_index=obs_index)
    temporary_ids = vpop["id"]
    expected_vpop = pd.DataFrame(
        {"id": temporary_ids, "k__12": [0, 1], "k__21": [1, 0]}
    )
    pd.testing.assert_frame_equal(
        vpop, expected_vpop, check_like=True, check_dtype=False
    )

    _out = struct_model.simulate(X=X, prediction_index=obs_index)
