"""Regression tests documenting the patient-ID ordering bug.

See BUG_FINDINGS.md #1 / IMPLEMENTATION_REVIEW.md #1.

`ObsData.patients` (vpop_calibration/pynlme/data.py) keeps patients in
first-appearance order, while `DataIndex.from_dataframe`
(vpop_calibration/pynlme/indexing.py) sorts the `id` field. The two orderings
are never reconciled, so every structural-model backend's batched prediction
path silently pairs a patient's observations with a DIFFERENT patient's
simulated parameters whenever first-appearance order != sorted order.

These tests assert the CORRECT behavior (predictions/likelihoods keyed by
patient identity, matching the id-keyed `single_patient_likelihood_factory`
ground truth) and therefore currently FAIL against the buggy batched code
path. That is intentional: they document the bug until it is fixed.
"""

import pandas as pd
import torch

from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical


def equations(offset, gain, t):
    return offset + gain + 0 * t


def _build_model(patient_ids):
    obs = ObsData(
        pd.DataFrame(
            {
                "id": patient_ids,
                "output_name": ["y", "y"],
                "time": [0.0, 0.0],
                "value": [21.0, 11.0],
                "offset": [20.0, 10.0],
            }
        )
    )
    params = MixedEffectParameters(
        pdu={"gain": {"prior": 1.0, "prior_omega": 0.1}},
        pdk=["offset"],
        error_model={"y": {"error_type": "additive", "sigma": 1.0}},
    )
    model = StatisticalModel(StructuralAnalytical(equations, ["y"]), obs, params)
    return model


def test_batched_predictions_stay_aligned_with_patient_identity():
    """Rows for patient "b" are listed before patient "a" in the input
    dataframe, so first-appearance order (["b", "a"]) differs from the
    sorted `DataIndex` order (["a", "b"]).
    """
    model = _build_model(["b", "a"])

    batched = model.log_posterior_etas_all_patients(torch.zeros(1, 2, 1))
    singles = [
        model.single_patient_likelihood_factory(p)(torch.zeros(1, 1, 1))
        for p in model.patients
    ]
    expected_predictions = [[s.predictions.item() for s in singles]]

    assert batched.predictions.tolist() == expected_predictions, (
        "Batched predict_all_patients() must return each patient's own "
        "prediction in the same row order as the observations, matching the "
        "id-keyed single_patient_likelihood_factory ground truth."
    )


def test_patient_order_already_matches_sorted_order_is_unaffected():
    """Sanity check: when first-appearance order already equals sorted
    order (ids "a", "b" fed in that row order), there is no ordering
    mismatch and predictions are already correct. This should PASS
    both before and after any fix.
    """
    model = _build_model(["a", "b"])

    batched = model.log_posterior_etas_all_patients(torch.zeros(1, 2, 1))
    singles = [
        model.single_patient_likelihood_factory(p)(torch.zeros(1, 1, 1))
        for p in model.patients
    ]
    expected_predictions = [[s.predictions.item() for s in singles]]

    assert batched.predictions.tolist() == expected_predictions


def test_sorting_input_by_string_coerced_id_is_not_the_same_as_numeric_sort():
    """Caveat found while investigating a workaround: `id` is coerced to
    `str` by ObsDataSchema, and DataIndex sorts lexicographically. Feeding
    patient ids 2 and 10 in NUMERIC order ("2" then "10") does not match the
    lexicographic order ("10" < "2" as strings), so the mismatch still
    occurs even though the caller believes their data is "sorted".
    """
    model = _build_model([2, 10])

    batched = model.log_posterior_etas_all_patients(torch.zeros(1, 2, 1))
    singles = [
        model.single_patient_likelihood_factory(p)(torch.zeros(1, 1, 1))
        for p in model.patients
    ]
    expected_predictions = [[s.predictions.item() for s in singles]]

    assert batched.predictions.tolist() == expected_predictions
