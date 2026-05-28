import torch
import pandas as pd
from torch.testing import assert_close

from vpop_calibration.structural_model.analytical import StructuralAnalytical
from vpop_calibration.nlme_model.indexing import ObservationIndex, IndexedValues


def test_analytical_no_protocol():
    def logistic_growth(lambda1, lambda2, lambda3, t):
        return lambda1 / (1 + torch.exp(-(t - lambda2) / lambda3))

    struct_model = StructuralAnalytical(
        logistic_growth,
        ["circumference"],
        protocol_design=None,
    )

    assert struct_model.parameter_names == ["lambda1", "lambda2", "lambda3"]
    assert struct_model.protocol_parameters == []
    assert struct_model.nb_protocol_overrides == 0
    assert struct_model.task_names == ["circumference_identity"]
    assert struct_model.input_to_function_arg == [0, 1, 2, 3]


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
    assert struct_model.task_names == ["circumference_Italy"]
    assert struct_model.input_to_function_arg == [0, 1, 2, 3, 4]
    assert struct_model.protocol_parameters == ["sun_power_multiplicator"]
    assert_close(struct_model.protocol_overrides_tensor, torch.tensor([[1.05]]))


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
    assert struct_model.input_to_function_arg == [
        0,  # position of "lambda3"
        3,  # position of "t"
        4,  # position of "max_circumference_multiplicator"
        1,  # position of "lambda2"
        2,  # position of "lambda1"
        5,  # position of "sun_power_multiplicator"
    ]
    assert struct_model.nb_protocol_overrides == 2
    assert struct_model.task_names == ["circumference_Italy"]
    assert_close(struct_model.protocol_overrides_tensor, torch.tensor([[1.05, 1.1]]))


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
    assert struct_model.task_names == [
        "circumference_Italy",
        "circumference_Morocco",
    ]
    assert_close(
        struct_model.protocol_overrides_tensor, torch.tensor([[1.05, 1.1], [1.2, 1.25]])
    )
    patient_index = IndexedValues(torch.tensor([0, 0, 0, 1, 1, 1]), ["p1", "p2"])
    timestep_index = IndexedValues(torch.tensor([0, 1, 2, 0, 1, 2]), [0, 1, 2])
    output_index = IndexedValues(torch.tensor([0, 0, 0, 0, 0, 0]), ["circumference"])
    protocol_index = IndexedValues(
        torch.tensor([0, 0, 0, 1, 1, 1]), ["Italy", "Morocco"]
    )
    task_index = IndexedValues(
        torch.tensor([0, 0, 0, 1, 1, 1]),
        ["circumference_Italy", "circumference_Morocco"],
    )

    obs_index = ObservationIndex(
        id=patient_index,
        time=timestep_index,
        output_name=output_index,
        task=task_index,
        protocol_arm=protocol_index,
    )
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
    num_chains, nb_patients, nb_timesteps, nb_params = X.shape
    actual_y, _pred_var = struct_model.simulate(X, obs_index)
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
