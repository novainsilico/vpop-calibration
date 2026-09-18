"""Regression tests documenting `simulate_from_df()`'s missing index remap.

See BUG_FINDINGS.md and IMPLEMENTATION_REVIEW.md #6.

`StructuralModel.simulate_from_df()` (vpop_calibration/structural_model/
base.py) builds `obs_index = DataIndex.from_dataframe(obs_df_validated)`,
whose `output_name`/`protocol_arm`/`task` reference lists are sorted
alphabetically by `DataIndex.from_dataframe`. Unlike the main SAEM/MCMC
pipeline (which calls `remap_all_indexings()` to align these with the
structural model's own `output_names`/`protocol_arms` ordering),
`simulate_from_df()` never remaps them, so a structural-model backend that
selects outputs/protocol overrides by index silently picks the wrong one
whenever alphabetical order differs from the model's declared order.

Both tests assert the correct output/value and currently FAIL.
"""

import pandas as pd
import torch

from vpop_calibration.structural_model.analytical import StructuralAnalytical


def _observations(outputs, arm="identity"):
    return pd.DataFrame(
        {
            "id": ["p1"] * len(outputs),
            "protocol_arm": [arm] * len(outputs),
            "output_name": outputs,
            "time": [0.0] * len(outputs),
            "value": [0.0] * len(outputs),
        }
    )


def _patient(arm="identity"):
    return pd.DataFrame({"id": ["p1"], "protocol_arm": [arm], "k": [2.0]})


def test_simulate_from_df_selects_the_right_output_column():
    """Model declares outputs ["z", "a"] in that order, computing z=k and
    a=10*k. Requesting both outputs (in the same ["z", "a"] order) should
    give predictions [2, 20], not the alphabetically-remapped [20, 2].
    """

    def two_outputs(k, t):
        return torch.cat([k + 0 * t, 10 * k + 0 * t], dim=-1)

    model = StructuralAnalytical(two_outputs, ["z", "a"])
    predicted = model.simulate_from_df(_patient(), _observations(["z", "a"]))[
        "predicted_value"
    ].tolist()

    assert predicted == [2.0, 20.0], (
        "simulate_from_df() must return each requested output's own "
        f"predicted value in the requested row order; got {predicted}, "
        "expected [2.0, 20.0] (z=k=2, a=10*k=20). The alphabetical "
        "DataIndex ordering used internally is never remapped to the "
        "model's own output order, so the two outputs' values get swapped."
    )


def test_simulate_from_df_selects_the_right_protocol_override():
    """Model has two protocol arms with different dose overrides
    (z: d=10, a: d=100). Requesting only arm "a" should use d=100.
    """

    def dose_model(k, t, d):
        return k * d + 0 * t

    protocol_design = pd.DataFrame({"protocol_arm": ["z", "a"], "d": [10.0, 100.0]})
    model = StructuralAnalytical(dose_model, ["y"], protocol_design)
    predicted = model.simulate_from_df(_patient("a"), _observations(["y"], "a"))[
        "predicted_value"
    ].tolist()

    assert predicted == [200.0], (
        f"simulate_from_df() must use arm 'a''s own dose override (d=100), "
        f"giving k*d=2*100=200; got {predicted}. Instead it silently picks "
        "the wrong protocol arm's override because protocol_arm indices "
        "are never remapped to the model's own protocol_arms order."
    )
