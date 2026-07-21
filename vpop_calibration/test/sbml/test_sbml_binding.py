import pytest
import pandas as pd

from vpop_calibration.structural_model.sbml import SbmlModelBinding


def test_sbml_model():
    file = "vpop_calibration/test/sbml/assets/model.xml"
    model = SbmlModelBinding(file=file, inputs=["k__a"], outputs=["A0", "A1", "A2"])

    time_steps: list[float] = [0, 1, 2]
    out = model.run_single_patient(
        patient_overrides={"k__a": 0.5}, time_steps=time_steps
    )

    vpop = pd.DataFrame({"id": ["p1", "p2"], "k__a": [0, 1]})

    out = model.run_vpop(vpop=vpop, time_steps=time_steps)


def test_sbml_model_wrong_inputs():
    file = "vpop_calibration/test/sbml/assets/model.xml"

    with pytest.raises(Exception):

        model = SbmlModelBinding(file=file, inputs=["incorrect_input"], outputs=["A0"])


def test_sbml_model_wrong_outputs():
    file = "vpop_calibration/test/sbml/assets/model.xml"
    with pytest.raises(Exception):
        model = SbmlModelBinding(file=file, inputs=[], outputs=["incorrect_output"])
