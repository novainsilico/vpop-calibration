import pytest
import numpy as np
import pandas as pd
import torch

from vpop_calibration.api.interface import NlmeModel
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical


@pytest.fixture(scope="function")
def sample_inputs(np_rng) -> tuple[dict, pd.DataFrame, StructuralModel]:
    priors = {
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
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    protocol_arms = ["arm-A", "arm-B"]
    patients = {
        "id": ["p1", "p2"],
        "foo": [0.0, 5.0],
        "pdk_1": [0.0, 0.0],
        "protocol_arm": protocol_arms,
    }
    outputs = ["out_1", "out_2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))

    def equations(mi_1, pdu_1, pdu_2, pdk_1, t, protocol_ovr_1):
        out = torch.zeros_like(t)
        return torch.cat((out, out), dim=-1)

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "protocol_ovr_1": [1, 2]}
    )
    struct_model = StructuralAnalytical(
        equations=equations,
        variable_names=["out_1", "out_2"],
        protocol_design=protocol_design,
    )

    return priors, df, struct_model


def test_nlme_interface(sample_inputs):
    priors, df, struct_model = sample_inputs
    _nlme_model = NlmeModel(df=df, prior_params=priors, structural_model=struct_model)


def test_state_dict(sample_inputs):
    priors, df, struct_model = sample_inputs
    nlme_model = NlmeModel(df=df, prior_params=priors, structural_model=struct_model)
    state_dict = nlme_model.get_state_dict()
    _new_nlme_model = NlmeModel.from_state_dict(
        state_dict=state_dict, df=df, structural_model=struct_model
    )
    nlme_model.optimizer.run()
    state_dict = nlme_model.get_state_dict()
    _new_nlme_model = NlmeModel.from_state_dict(
        state_dict=state_dict, df=df, structural_model=struct_model
    )
    nlme_model.diagnostics.sample_conditional_distribution()
    state_dict = nlme_model.get_state_dict()
    _new_nlme_model = NlmeModel.from_state_dict(
        state_dict=state_dict, df=df, structural_model=struct_model
    )


def test_save_load(sample_inputs, tmp_path):
    priors, df, struct_model = sample_inputs
    nlme_model = NlmeModel(df=df, prior_params=priors, structural_model=struct_model)
    nlme_model.optimizer.run()
    nlme_model.diagnostics.sample_conditional_distribution()

    model_path = tmp_path / "model.json"
    nlme_model.save(model_path)
    new_nlme_model = NlmeModel.load(model_path, df=df, struct_model=struct_model)

    assert (
        nlme_model.statistical_model.current_params
        == new_nlme_model.statistical_model.current_params
    )

    assert (
        nlme_model.statistical_model.prior_params
        == nlme_model.statistical_model.prior_params
    )
