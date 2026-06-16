import pytest
import pandas as pd
import numpy as np
import torch

from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
    StructuralSimwork,
)
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.interface import NlmeModel, Config, SaemConfigDict, NlmeConfigDict


@pytest.fixture
def simwork_model() -> StructuralModel:
    model = SimworkModelBinding(
        path_to_model="vpop_calibration/test/simwork_model/assets/model.json",
        path_to_solving_options="vpop_calibration/test/simwork_model/assets/options.json",
        inputs=["k_12", "k_21", "k_el", "dose"],
        outputs=["A0", "A1", "A2"],
    )

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["dose-1", "dose-10"], "dose": [1, 10]}
    )
    struct_model = StructuralSimwork(model=model, protocol_design=protocol_design)
    return struct_model


@pytest.fixture
def obs_data(np_rng) -> pd.DataFrame:
    protocol_arms = ["dose-1", "dose-10"]
    patients = {
        "id": ["p1", "p2"],
        "k_el": [0.01, 0.1],
        "protocol_arm": protocol_arms,
    }
    outputs = ["A0", "A1", "A2"]
    time_steps = np.arange(0, 3.0, 1.0)
    df = pd.DataFrame.from_dict(patients)
    df = df.merge(pd.DataFrame(outputs, columns=["output_name"]), how="cross")
    df = df.merge(pd.DataFrame(time_steps, columns=["time"]), how="cross")
    df["value"] = np.abs(np_rng.normal(0, 1, df.shape[0]))
    df["task"] = df.apply(lambda r: r["output_name"] + "_" + r["protocol_arm"], axis=1)

    return df


@pytest.fixture
def sample_nlme_params() -> dict:
    input = {
        # "model_intrinsic": {"k_a": {"prior": 10.0}},
        "pdu": {
            "k_12": {
                "prior": 10.0,
                "prior_omega": 0.1,
            },
            "k_21": {
                "prior": 10.0,
                "prior_omega": 0.1,
            },
        },
        "error_model": {
            "A0": {"error_type": "additive", "sigma": 0.1},
            "A1": {"error_type": "additive", "sigma": 0.5},
            "A2": {"error_type": "additive", "sigma": 0.5},
        },
        "pdk": ["k_el"],
    }
    return input


def test_simwork_saem(sample_nlme_params, obs_data, simwork_model):
    config = Config(saem=SaemConfigDict(verbose=True), nlme=NlmeConfigDict(nb_chains=1))
    nlme_model = NlmeModel(
        structural_model=simwork_model,
        df=obs_data,
        prior_params=sample_nlme_params,
        config=config,
    )
    nlme_model.optimizer.run()
