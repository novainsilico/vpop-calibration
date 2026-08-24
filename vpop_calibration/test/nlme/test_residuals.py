from vpop_calibration.pynlme.residuals import (
    ResidualErrorEstimates,
    calculate_residuals,
    compute_error_variance,
    log_likelihood_observation,
    add_predictive_error,
    compute_survival_likelihood,
)
from vpop_calibration.pynlme.error_estimation import estimate_error_params
from vpop_calibration.pynlme.indexing import (
    DataIndex,
    TensorIndexing,
    ObservationsDataSet,
    SurvivalOutputs,
)
from vpop_calibration.config import default_dtype

import torch
import pandas as pd
from math import log, pi, inf


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

    patient_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1", "p2", "p2"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = TensorIndexing(
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
    task_indices = TensorIndexing(
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

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    vals = torch.tensor([0, 1, 0, 1], dtype=default_dtype)
    pred = torch.tensor([[1, 2, 1, 2]], dtype=default_dtype)

    obs = ObservationsDataSet(obs_index=obs_index, obs_values=vals)

    res = calculate_residuals(obs, pred)
    expected_res = torch.tensor([[-1, -1, -1, -1]], dtype=default_dtype)
    torch.testing.assert_close(res, expected_res)

    residual_error = ResidualErrorEstimates(
        sigma_add=torch.tensor([1.0, 0.0], dtype=default_dtype),
        sigma_prop=torch.tensor([0.0, 1.0], dtype=default_dtype),
        additive_output=torch.tensor([True, False]),
        proportional_output=torch.tensor([False, True]),
    )

    out_variance = compute_error_variance(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )
    expected_variance = torch.tensor([[1, 4, 1, 4]], dtype=default_dtype)
    torch.testing.assert_close(out_variance, expected_variance)

    error_params = estimate_error_params(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )

    expected_error_params = residual_error._replace(
        sigma_add=torch.tensor([1.0, 0.0], dtype=default_dtype),
        sigma_prop=torch.tensor([0.0, 1 / 4], dtype=default_dtype),
    )
    torch.testing.assert_close(error_params, expected_error_params)

    log_lik = log_likelihood_observation(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )
    expected_log_lik = torch.tensor(
        [
            [
                -0.5 * (log(2 * pi * 1) + ((-1) ** 2 / 1))
                - 0.5 * (log(2 * pi * 4) + ((-1) ** 2 / 4)),
                -0.5 * (log(2 * pi * 1) + ((-1) ** 2 / 1))
                - 0.5 * (log(2 * pi * 4) + ((-1) ** 2 / 4)),
            ]
        ],
        dtype=default_dtype,
    )
    torch.testing.assert_close(log_lik, expected_log_lik)

    _noisy_prediction = add_predictive_error(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )


def test_residuals_with_inf():
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

    patient_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 1, 1]),
        ref_values=patient_id,
        raw_values=pd.Series(["p1", "p1", "p2", "p2"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["output_1", "output_2", "output_1", "output_2"]),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 2, 3]),
        ref_values=time,
        raw_values=pd.Series([0, 1, 2, 3]),
    )
    protocol_indices = TensorIndexing(
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
    task_indices = TensorIndexing(
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

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    obs_values = torch.tensor([0, 1, 0, 1], dtype=default_dtype)
    pred = torch.tensor([[1, 2, +inf, +inf]], dtype=default_dtype)

    obs = ObservationsDataSet(obs_index=obs_index, obs_values=obs_values)

    res = calculate_residuals(obs, pred)
    expected_res = torch.tensor([[-1, -1, -inf, -inf]], dtype=default_dtype)
    torch.testing.assert_close(res, expected_res)

    residual_error = ResidualErrorEstimates(
        sigma_add=torch.tensor([1.0, 0.0], dtype=default_dtype),
        sigma_prop=torch.tensor([0.0, 1.0], dtype=default_dtype),
        additive_output=torch.tensor([True, False]),
        proportional_output=torch.tensor([False, True]),
    )
    out_variance = compute_error_variance(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )
    expected_variance = torch.tensor([[1, 4, 1, 1]], dtype=default_dtype)
    torch.testing.assert_close(out_variance, expected_variance)

    expected_error_params = residual_error._replace(
        sigma_add=torch.tensor([1.0, 0.0], dtype=default_dtype),
        sigma_prop=torch.tensor([0.0, 1 / 4], dtype=default_dtype),
    )
    error_params = estimate_error_params(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )
    torch.testing.assert_close(error_params, expected_error_params)

    log_lik = log_likelihood_observation(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )
    expected_log_lik = torch.tensor(
        [
            [
                -0.5 * (log(2 * pi * 1) + ((-1) ** 2 / 1))
                - 0.5 * (log(2 * pi * 4) + ((-1) ** 2 / 4)),
                -inf,
            ]
        ],
        dtype=default_dtype,
    )
    torch.testing.assert_close(log_lik, expected_log_lik)

    _noisy_prediction = add_predictive_error(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )


def test_residuals_survival():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1"]
    outputs: list = ["log_hz", "cumulative_hz"]
    event_time_p1 = 2
    event_time_p2 = 3
    time: list = [event_time_p1, event_time_p2]
    tasks: list = [
        "log_hz_arm-1",
        "cumulative_hz_arm-1",
    ]

    # Patients are not ordered
    patient_indices = TensorIndexing(
        index_values=torch.tensor([1, 1, 0, 0]),
        ref_values=patient_id,
        raw_values=pd.Series(["p2", "p2", "p1", "p1"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=outputs,
        raw_values=pd.Series(["log_hz", "cumulative_hz", "log_hz", "cumulative_hz"]),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([1, 1, 0, 0]),
        ref_values=time,
        raw_values=pd.Series(
            [event_time_p2, event_time_p2, event_time_p1, event_time_p1]
        ),
    )
    protocol_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 0, 0]),
        ref_values=protocols,
        raw_values=pd.Series(
            [
                "arm_1",
                "arm_1",
                "arm_1",
                "arm_1",
            ]
        ),
    )
    task_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1]),
        ref_values=tasks,
        raw_values=pd.Series(
            [
                "log_hz_arm-1",
                "cumulative_hz_arm-1",
                "log_hz_arm-1",
                "cumulative_hz_arm-1",
            ]
        ),
    )

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    # Observations are event_status
    event_status_p1 = 1.0
    event_status_p2 = 0.0
    obs_values = torch.tensor(
        [event_status_p2, event_status_p2, event_status_p1, event_status_p1],
        dtype=default_dtype,
    )

    pred_log_hz_p1 = -1
    pred_cum_hz_p1 = 0.3

    pred_log_hz_p2 = -2
    pred_cum_hz_p2 = 0.7

    pred_one_sample = [pred_log_hz_p2, pred_cum_hz_p2, pred_log_hz_p1, pred_cum_hz_p1]
    pred = torch.tensor([pred_one_sample, pred_one_sample], dtype=default_dtype)

    obs = ObservationsDataSet(
        obs_index=obs_index,
        obs_values=obs_values,
        survival_outputs=SurvivalOutputs(
            log_hazard="log_hz", cumulative_hazard="cumulative_hz"
        ),
    )

    ll = compute_survival_likelihood(observations=obs, predictions=pred)

    ll_p1 = event_status_p1 * pred_log_hz_p1 - pred_cum_hz_p1
    ll_p2 = event_status_p2 * pred_log_hz_p2 - pred_cum_hz_p2
    expected_ll = torch.tensor([[ll_p1, ll_p2], [ll_p1, ll_p2]], dtype=default_dtype)
    torch.testing.assert_close(ll, expected_ll)


def test_joint_likelihood():
    patient_id: list = ["p1", "p2"]
    protocols: list = ["arm-1"]
    outputs: list = ["log_hz", "cumulative_hz", "out1"]
    event_time_p1 = 2
    event_time_p2 = 3
    obs_time_cont = 10
    time: list = [event_time_p1, event_time_p2, obs_time_cont]
    tasks: list = ["log_hz_arm-1", "cumulative_hz_arm-1", "out1_arm-1"]

    # Patients are not ordered
    patient_indices = TensorIndexing(
        index_values=torch.tensor([1, 1, 0, 0, 0]),
        ref_values=patient_id,
        raw_values=pd.Series(["p2", "p2", "p1", "p1", "p1"]),
    )
    outputs_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1, 2]),
        ref_values=outputs,
        raw_values=pd.Series(
            ["log_hz", "cumulative_hz", "log_hz", "cumulative_hz", "out1"]
        ),
    )
    time_indices = TensorIndexing(
        index_values=torch.tensor([1, 1, 0, 0, 2]),
        ref_values=time,
        raw_values=pd.Series(
            [event_time_p2, event_time_p2, event_time_p1, event_time_p1, obs_time_cont]
        ),
    )
    protocol_indices = TensorIndexing(
        index_values=torch.tensor([0, 0, 0, 0, 0]),
        ref_values=protocols,
        raw_values=pd.Series(["arm_1", "arm_1", "arm_1", "arm_1", "arm_1"]),
    )
    task_indices = TensorIndexing(
        index_values=torch.tensor([0, 1, 0, 1, 2]),
        ref_values=tasks,
        raw_values=pd.Series(
            [
                "log_hz_arm-1",
                "cumulative_hz_arm-1",
                "log_hz_arm-1",
                "cumulative_hz_arm-1",
                "out1_arm-1",
            ]
        ),
    )

    obs_index = DataIndex(
        id=patient_indices,
        output_name=outputs_indices,
        protocol_arm=protocol_indices,
        time=time_indices,
        task=task_indices,
    )

    # Observations are event_status
    event_status_p1 = 1.0
    event_status_p2 = 0.0
    obs_continuous_out1 = 10.0
    obs_values = torch.tensor(
        [
            event_status_p2,
            event_status_p2,
            event_status_p1,
            event_status_p1,
            obs_continuous_out1,
        ],
        dtype=default_dtype,
    )

    pred_log_hz_p1 = -1
    pred_cum_hz_p1 = 0.3

    pred_log_hz_p2 = -2
    pred_cum_hz_p2 = 0.7

    pred_continuous_out1 = 5.0
    pred_one_sample = [
        pred_log_hz_p2,
        pred_cum_hz_p2,
        pred_log_hz_p1,
        pred_cum_hz_p1,
        pred_continuous_out1,
    ]

    pred = torch.tensor([pred_one_sample, pred_one_sample], dtype=default_dtype)

    obs = ObservationsDataSet(
        obs_index=obs_index,
        obs_values=obs_values,
        survival_outputs=SurvivalOutputs(
            log_hazard="log_hz", cumulative_hazard="cumulative_hz"
        ),
    )

    sigma_val = 2.0
    residual_error = ResidualErrorEstimates(
        sigma_add=torch.tensor([0.0, 0.0, sigma_val], dtype=default_dtype),
        sigma_prop=torch.tensor([0.0, 0.0, 0.0], dtype=default_dtype),
        additive_output=torch.tensor([False, False, True]),
        proportional_output=torch.tensor([False, False, False]),
    )

    ll = log_likelihood_observation(
        observations=obs,
        predictions=pred,
        residual_error=residual_error,
        min_variance=1e-6,
    )

    ll_p1 = event_status_p1 * pred_log_hz_p1 - pred_cum_hz_p1
    ll_p2 = event_status_p2 * pred_log_hz_p2 - pred_cum_hz_p2
    ll_out1 = -0.5 * (
        log(2 * pi * sigma_val)
        + ((pred_continuous_out1 - obs_continuous_out1) ** 2 / sigma_val)
    )
    expected_ll = torch.tensor(
        [[ll_p1 + ll_out1, ll_p2], [ll_p1 + ll_out1, ll_p2]], dtype=default_dtype
    )
    torch.testing.assert_close(ll, expected_ll)
