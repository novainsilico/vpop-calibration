import pandas as pd
import roadrunner
import torch
import pytest

from vpop_calibration.structural_model.sbml import StructuralSbml
from vpop_calibration.pynlme.indexing import DataIndex, ObsDataSchema


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
    obs_index = DataIndex.from_dataframe(ObsDataSchema.validate(df))

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


INITIAL_ASSIGNMENT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="initial_assignment">
    <listOfParameters>
      <parameter id="a" value="5" constant="false"/>
      <parameter id="b" constant="false"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="b">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn> 2 </cn><ci> a </ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
  </model>
</sbml>
"""


def test_roadrunner_initial_assignment_set_value_vs_init():
    """b = 2 * a is an initial assignment. Setting `a` does not propagate to b,
    while setting `init(a)` does."""
    rr = roadrunner.RoadRunner(INITIAL_ASSIGNMENT_SBML)

    # same call sequence as simulate_rr_single_patient
    rr.resetToOrigin()
    rr.setValues({"a": 1})
    rr.reset()
    value_a, value_b_from_value = rr["a"], rr["b"]

    rr.resetToOrigin()
    rr.setValues({"init(a)": 1})
    rr.reset()
    init_a, value_b_from_init = rr["a"], rr["b"]

    # setting `a` directly: b still reflects the original a = 5
    assert value_a == 1
    assert value_b_from_value == 10
    # setting `init(a)`: the initial assignment is re-evaluated with a = 1
    assert init_a == 1
    assert value_b_from_init == 2
