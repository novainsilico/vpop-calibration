from vpop_calibration.pynlme.residuals import (
    calculate_residuals,
    compute_error_variance,
    sum_sq_residuals,
    log_likelihood_observation,
)
from vpop_calibration.pynlme.params import ErrorType
from vpop_calibration.pynlme.indexing import (
    ObservationIndex,
    IndexedValues,
    IndexedObservations,
)

import torch
import pandas as pd


def test_residuals():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1", "arm-2", "arm-3"]
    outputs: list = ["output_1", "output_2"]
    time: list = [0, 1, 2, 3]
    tasks: list = [
        "output_1_arm-1",
        "output_2_arm-1",
        "output_1_arm-2",
        "output_2_arm-2",
        "output_1_arm-3",
        "output_2_arm-3",
    ]

    patient_indices = IndexedValues(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1"]),
    )
    outputs_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 0, 2]),
        ref_values=protocols,
        raw_values=pd.Series(
            [
                "arm_1",
                "arm_2",
                "arm_1",
                "arm_3",
            ]
        ),
    )
    task_indices = IndexedValues(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=tasks,
        raw_values=pd.Series(
            [
                "output_1_arm-1",
                "output_2_arm-2",
                "output_1_arm-1",
                "output_2_arm-3",
            ]
        ),
    )

    obs_index = ObservationIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    obs_index = ObservationIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    vals = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    pred = torch.tensor([[1, 2, 1, 2]], dtype=torch.float32)

    obs = IndexedObservations(obs_index=obs_index, obs_values=vals)
    error_model_selector: dict[ErrorType, list[int]] = {
        "additive": [0],
        "proportional": [1],
    }

    res = calculate_residuals(obs, pred, error_model_selector)
    expected_res = torch.tensor([[-1, -1 / 2, -1, -1 / 2]], dtype=torch.float32)
    torch.testing.assert_close(res, expected_res)

    sigma = torch.tensor([1, 1], dtype=torch.float32)
    out_variance = compute_error_variance(
        observations=obs,
        predictions=pred,
        error_model_selector=error_model_selector,
        sigma=sigma,
    )
    expected_variance = torch.tensor([[1, 4, 1, 4]], dtype=torch.float32)
    torch.testing.assert_close(out_variance, expected_variance)

    sum_sq_res = sum_sq_residuals(
        observations=obs, prediction=pred, error_model_selector=error_model_selector
    )
    expected_sum_sq_res = torch.tensor([[2, 1 / 2]], dtype=torch.float32)
    torch.testing.assert_close(sum_sq_res, expected_sum_sq_res)

    log_lik = log_likelihood_observation(
        observations=obs,
        predictions=pred,
        error_model_selector=error_model_selector,
        sigma=sigma,
    )
