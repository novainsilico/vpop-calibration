"""Regression test documenting the analytical protocol-override column bug.

See BUG_FINDINGS.md #5 and IMPLEMENTATION_REVIEW.md #7.

`StructuralAnalytical` (vpop_calibration/structural_model/analytical.py)
builds `protocol_overrides_tensor` from `protocol_design`'s columns in
DATAFRAME COLUMN ORDER, but interprets them at simulation time in FUNCTION
ARGUMENT order. If the dataframe's columns are not already in the same
order as the equation's arguments, override values get silently assigned to
the wrong parameter name.

This test declares `equations(k, t, a, b)` (so the expected override order
is a, then b) but supplies `protocol_design` with columns in the order
b, then a. It currently FAILS: `a`'s value (3) and `b`'s value (2) get
swapped internally, giving `k + 10*b + a` instead of `k + 10*a + b`.
"""

import pandas as pd
import torch

from vpop_calibration.structural_model.analytical import StructuralAnalytical


def test_protocol_override_columns_are_matched_by_name_not_position():
    def overrides_model(k, t, a, b):
        return k + 10 * a + b + 0 * t

    # Columns deliberately in the OPPOSITE order to the function's (a, b) args.
    protocol_design = pd.DataFrame({"protocol_arm": ["identity"], "b": [2.0], "a": [3.0]})
    model = StructuralAnalytical(overrides_model, ["y"], protocol_design)

    patient = pd.DataFrame({"id": ["p1"], "protocol_arm": ["identity"], "k": [2.0]})
    obs = pd.DataFrame(
        {
            "id": ["p1"],
            "protocol_arm": ["identity"],
            "output_name": ["y"],
            "time": [0.0],
            "value": [0.0],
        }
    )

    predicted = model.simulate_from_df(patient, obs)["predicted_value"].tolist()

    # k=2, a=3, b=2 -> k + 10*a + b = 2 + 30 + 2 = 34
    assert predicted == [34.0], (
        f"Expected k + 10*a + b = 34 (k=2, a=3, b=2) regardless of the "
        f"protocol_design column order, got {predicted}. The override "
        "values are being assigned by dataframe column position instead of "
        "by matching column name to argument name."
    )
