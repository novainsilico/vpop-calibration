import pytest
import pandas as pd
import numpy as np
from pandera.typing import DataFrame
import torch

from vpop_calibration.data.observed import ObsData
from vpop_calibration.nlme_model.params import MixedEffectParameters
from vpop_calibration.structural_model import StructuralAnalytical, StructuralModel
from vpop_calibration.nlme_model.model import NlmeModel


@pytest.fixture
def sample_nlme_params() -> MixedEffectParameters:
    input = {
        "model_intrinsic": {"mi_1": {"prior": 10.0}},
        "pdu": {
            "pdu_1": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu1", "prior": 0.5}},
            },
            "pdu_2": {
                "prior": 10.0,
                "prior_omega": 0.1,
                "covariates": {"foo": {"coef_name": "coef_foo_pdu2", "prior": 0.5}},
                "constraint": {"low": 1.0, "high": 100.0},
            },
        },
        "error_model": {
            "out_1": {"type": "additive", "sigma": 0.1},
            "out_2": {"type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    return MixedEffectParameters.model_validate(input)


@pytest.fixture
def obs_data(np_rng) -> ObsData:
    patients = {"id": ["p1", "p2"], "foo": [0.0, 5.0], "pdk_1": [0.0, 0.0]}
    protocol_arms = ["arm-A", "arm-B"]
    outputs = ["out_1", "out_2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(protocol_arms, columns=["protocol_arm"]), how="cross")
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))
    data = ObsData(DataFrame(df))
    return data


@pytest.fixture
def struct_model() -> StructuralModel:
    def equations(pdu_1, pdu_2, foo, pdk_1, t):
        return (0.0, 0.0)

    struct_model = StructuralAnalytical(
        equations=equations, variable_names=["out_1", "out_2"]
    )
    return struct_model


def test_nlme_init(sample_nlme_params, obs_data, struct_model):
    nlme_model = NlmeModel(
        structural_model=struct_model, dataset=obs_data, prior_params=sample_nlme_params
    )
    # Ensure the initial omega matrix is diagonal with correct values
    assert torch.equal(nlme_model.omega_pop, torch.diag(torch.as_tensor([0.1, 0.1])))
    # Ensure the residual error tensor is properly initialized
    assert torch.equal(nlme_model.residual_var, torch.as_tensor([0.1, 0.5]))
    # Ensure the design matrices are properly initialized
    expected_design_matrix_p1 = torch.as_tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    assert torch.equal(nlme_model.design_matrices["p1"], expected_design_matrix_p1)
    expected_design_matrix_p2 = torch.as_tensor(
        [[1.0, 5.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0]]
    )
    assert torch.equal(nlme_model.design_matrices["p2"], expected_design_matrix_p2)
    assert torch.equal(
        nlme_model.full_design_matrix,
        torch.stack((expected_design_matrix_p1, expected_design_matrix_p2)),
    )
    # Ensure the parameter transforms are correctly translated
    assert torch.equal(nlme_model.mi_transforms["exp"], torch.Tensor([0]))
    assert torch.equal(nlme_model.mi_shift, torch.tensor([[[0]]]))
    assert torch.equal(nlme_model.mi_scale, torch.tensor([[[1]]]))
    assert torch.equal(nlme_model.pdu_transforms["exp"], torch.Tensor([0]))
    assert torch.equal(nlme_model.pdu_transforms["sigmoid"], torch.Tensor([1]))
    assert torch.equal(nlme_model.pdu_shift, torch.tensor([[[0, 1.0]]]))
    assert torch.equal(nlme_model.pdu_scale, torch.tensor([[[1, 100 - 1]]]))
