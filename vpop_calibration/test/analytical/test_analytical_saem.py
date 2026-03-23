import pytest
import torch
import pandas as pd
from torch.testing import assert_close

from vpop_calibration import *


def test_analytical_saem(np_rng):
    def logistic_growth(lambda1, lambda2, lambda3, t):
        y = lambda1 / (1 + torch.exp(-(t - lambda2) / lambda3))
        return y

    outputs = ["circ"]
    struct_model = StructuralAnalytical(logistic_growth, outputs)

    patient_df = pd.DataFrame({"id": ["tree-1", "tree-2"]})
    df = patient_df.merge(pd.DataFrame({"output_name": outputs}), how="cross")
    time_steps = [0.0, 1.0, 3.0]
    df = df.merge(pd.DataFrame({"time": time_steps}), how="cross")
    df["value"] = np_rng.normal(100, 10, df.shape[0])

    init_log_mi = {"lambda2": 0.0, "lambda3": 0.0}
    init_log_pdu = {
        "lambda1": {"mean": 0.0, "sd": 0.5},
    }
    constraints = {"lambda1": {"low": 0.0, "high": 300}}
    covariate_map = None
    init_res_var = [100.0]
    nlme_model = NlmeModel(
        structural_model=struct_model,
        patients_df=patient_df,
        init_log_MI=init_log_mi,
        init_PDU=init_log_pdu,
        covariate_map=covariate_map,
        init_res_var=init_res_var,
        error_model_type="additive",
        num_chains=1,
        constraints=constraints,
    )

    optimizer = PySaem(nlme_model, df)
    optimizer.run()
    nlme_model.compute_ebe()
    nlme_model.sample_conditional_distribution(nb_samples=3)
    plot_individual_map_estimates(nlme_model)
    plot_all_individual_map_estimates(nlme_model)
    plot_map_estimates_gof(nlme_model)
    plot_weighted_residuals(nlme_model, "iwres")
    plot_weighted_residuals(nlme_model, "pwres")
    plot_weighted_residuals(nlme_model, "npde")


def test_analytical_no_protocol():
    def logistic_growth(lambda1, lambda2, lambda3, t):
        return lambda1 / (1 + torch.exp(-(t - lambda2) / lambda3))

    struct_model = StructuralAnalytical(
        logistic_growth,
        ["circumference"],
        protocol_design=None,
    )

    assert struct_model.parameter_names == ["lambda1", "lambda2", "lambda3"]
    assert struct_model.nb_protocol_overrides == 0
    assert struct_model.tasks == ["circumference_identity"]
    assert struct_model.task_idx_to_protocol == {0: "identity"}
    assert struct_model.task_idx_to_output_idx == {0: 0}
    num_tasks = len(struct_model.tasks)
    num_protocol_parameters = 0
    assert struct_model.task_protocol_tensor.shape == (
        num_tasks,
        num_protocol_parameters,
    )


def test_analytical_one_arm_one_override():
    def logistic_growth(lambda1, lambda2, lambda3, t, sun_power_multiplicator):
        return lambda1 / (
            1 + torch.exp(-(sun_power_multiplicator * (t - lambda2) / lambda3))
        )

    struct_model = StructuralAnalytical(
        logistic_growth,
        ["circumference"],
        protocol_design=pd.DataFrame(
            data={"protocol_arm": ["Italy"], "sun_power_multiplicator": [1.05]}
        ),
    )

    assert struct_model.parameter_names == [
        "lambda1",
        "lambda2",
        "lambda3",
    ]
    assert struct_model.nb_protocol_overrides == 1
    assert struct_model.tasks == ["circumference_Italy"]
    assert struct_model.task_idx_to_protocol == {0: "Italy"}
    assert struct_model.task_idx_to_output_idx == {0: 0}
    num_tasks = len(struct_model.tasks)
    num_protocol_parameters = 1
    assert struct_model.task_protocol_tensor.shape == (
        num_tasks,
        num_protocol_parameters,
    )


def test_analytical_one_arm_two_overrides():
    # Here we scrambled the function arguments on purpose, to check
    # that the logic of self.input_tensor_column_index_to_function_parameter_index
    # is properly implemented
    def logistic_growth(
        lambda3,
        t,
        max_circumference_multiplicator,
        lambda2,
        lambda1,
        sun_power_multiplicator,
    ):
        return (
            max_circumference_multiplicator
            * lambda1
            / (1 + torch.exp(-(sun_power_multiplicator * (t - lambda2) / lambda3)))
        )

    struct_model = StructuralAnalytical(
        logistic_growth,
        ["circumference"],
        protocol_design=pd.DataFrame(
            data={
                "protocol_arm": ["Italy"],
                "sun_power_multiplicator": [1.05],
                "max_circumference_multiplicator": [1.1],
            }
        ),
    )

    assert struct_model.parameter_names == [
        "lambda3",
        "lambda2",
        "lambda1",
    ]

    # internally we have input_parameters = parameter_names_without_protocol_overrides + ["t"] + protocol_parameters
    # ["lambda3", "lambda2", "lambda1", "t", "max_circumference_multiplicator", "sun_power_multiplicator"]
    #      0         1           2       3                    4                          5
    assert struct_model.input_tensor_column_index_to_function_parameter_index == [
        0,  # position of "lambda3"
        3,  # position of "t"
        4,  # position of "max_circumference_multiplicator"
        1,  # position of "lambda2"
        2,  # position of "lambda1"
        5,  # position of "sun_power_multiplicator"
    ]
    assert struct_model.nb_protocol_overrides == 2
    assert struct_model.tasks == ["circumference_Italy"]
    assert struct_model.task_idx_to_protocol == {0: "Italy"}
    assert struct_model.task_idx_to_output_idx == {0: 0}
    num_tasks = len(struct_model.tasks)
    num_protocol_parameters = 2
    assert struct_model.task_protocol_tensor.shape == (
        num_tasks,
        num_protocol_parameters,
    )


def test_analytical_two_arms_two_overrides():
    def logistic_growth(
        lambda1,
        lambda2,
        lambda3,
        t,
        sun_power_multiplicator,
        max_circumference_multiplicator,
    ):
        return (
            max_circumference_multiplicator
            * lambda1
            / (1 + torch.exp(-(sun_power_multiplicator * (t - lambda2) / lambda3)))
        )

    protocol_design = pd.DataFrame(
        data={
            "protocol_arm": ["Italy", "Morocco"],
            "sun_power_multiplicator": [1.05, 1.2],
            "max_circumference_multiplicator": [1.1, 1.25],
        }
    )
    struct_model = StructuralAnalytical(
        logistic_growth,
        ["circumference"],
        protocol_design=protocol_design,
    )

    assert struct_model.parameter_names == [
        "lambda1",
        "lambda2",
        "lambda3",
    ]
    assert struct_model.nb_protocol_overrides == 2
    assert struct_model.tasks == ["circumference_Italy", "circumference_Morocco"]
    assert struct_model.task_idx_to_protocol == {0: "Italy", 1: "Morocco"}
    assert struct_model.task_idx_to_output_idx == {0: 0, 1: 0}
    num_tasks = len(struct_model.tasks)
    num_protocol_parameters = 2
    assert struct_model.task_protocol_tensor.shape == (
        num_tasks,
        num_protocol_parameters,
    )
    patient_index = torch.tensor([0, 0, 0, 1, 1, 1])
    timestep_index = torch.tensor([0, 1, 2, 0, 1, 2])
    #                           Italy    Morocco
    task_index = torch.tensor([0, 0, 0, 1, 1, 1])
    X = torch.tensor(
        [
            [
                [  # lam1 lam2 lam3  t
                    [333, 101, 350, 118],  # patient 1 at t1
                    [333, 101, 350, 484],  # patient 1 at t2
                    [333, 101, 350, 664],  # patient 1 at t3
                ],
                [
                    [483, 100, 360, 118],  # patient 2 at t1
                    [483, 100, 360, 484],  # patient 2 at t2
                    [483, 100, 360, 664],  # patient 2 at t3
                ],
            ]
        ]
    )
    (num_chains, nb_patients, nb_timesteps, nb_params) = X.shape
    actual_y, _pred_var = struct_model.simulate(
        X, (patient_index, timestep_index, task_index), chunks=[6]
    )
    expected_y = logistic_growth(
        lambda1=torch.tensor([333, 333, 333, 483, 483, 483]),
        lambda2=torch.tensor([101, 101, 101, 100, 100, 100]),
        lambda3=torch.tensor([350, 350, 350, 360, 360, 360]),
        t=torch.tensor([118, 484, 664, 118, 484, 664]),
        #                                           Italy          Morocco
        sun_power_multiplicator=torch.tensor([1.05, 1.05, 1.05, 1.2, 1.2, 1.2]),
        #                                                  Italy          Morocco
        max_circumference_multiplicator=torch.tensor([1.1, 1.1, 1.1, 1.25, 1.25, 1.25]),
    ).reshape((num_chains, nb_patients * nb_timesteps))
    assert_close(actual_y, expected_y)
