from vpop_calibration.saem.optimizer import PySaem
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.structural_model import StructuralAnalytical

import pytest
import pandas as pd
import numpy as np
import torch
from pandera.typing import DataFrame


@pytest.fixture
def sample_nlme_model(np_rng) -> StatisticalModel:
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
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    params = MixedEffectParameters.model_validate(input)
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
    obs_data = ObsData(DataFrame(df))

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

    nlme_model = StatisticalModel(
        structural_model=struct_model, dataset=obs_data, prior_params=params
    )

    return nlme_model


def test_optimizer(sample_nlme_model):
    optim = PySaem(model=sample_nlme_model, config=SaemConfigDict())
    optim.run()
