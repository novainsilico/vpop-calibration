"""Regression test documenting the SBML initial-time shift bug.

See BUG_FINDINGS.md and IMPLEMENTATION_REVIEW.md #5.

`StructuralSbml` (vpop_calibration/structural_model/sbml.py) resets initial
conditions and then calls `rr.simulate(times=time_steps)`. When the first
requested time is strictly positive, RoadRunner assigns the initial
conditions to that first requested time rather than integrating from time
zero first, so predictions depend incorrectly on whether time zero is
included in the observation schedule.

For the test model, `A0(t) = 0.5 * exp(-0.5 * t)`, so `A0(1) ~= 0.30326533`.
Requesting only `time_steps=[1, 2]` should give the same value at t=1 as
requesting `[0, 1, 2]` and dropping the t=0 row; instead it incorrectly uses
the t=0 initial condition (0.5) as the value AT t=1.
"""

import pandas as pd
import pytest

from vpop_calibration.structural_model.sbml import StructuralSbml


def test_sbml_prediction_at_t1_is_independent_of_including_t0():
    file = "vpop_calibration/test/sbml/assets/model.xml"
    model = StructuralSbml(file, inputs=["k__a"], outputs=["A0"])
    vpop = pd.DataFrame({"id": ["p1"], "k__a": [0.5]})

    with_t0 = model.run_vpop(vpop, time_steps=[0.0, 1.0, 2.0])["A0"].tolist()
    without_t0 = model.run_vpop(vpop, time_steps=[1.0, 2.0])["A0"].tolist()

    assert without_t0 == pytest.approx(with_t0[1:], abs=1e-6), (
        f"Prediction at t=1 should not depend on whether t=0 is included in "
        f"the requested time grid; got with_t0={with_t0}, "
        f"without_t0={without_t0}. Because RoadRunner isn't integrated from "
        "the model's initial time first, requesting [1, 2] incorrectly "
        "reuses the t=0 initial condition as the value at t=1."
    )
