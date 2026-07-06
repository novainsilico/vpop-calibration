from vpop_calibration.sdk import (
    create_nlme_model,
    run_saem,
    run_diagnostics,
    DiagnosticsConfig,
)
from vpop_calibration.interface import Config

import pytest
import pandas as pd
import numpy as np


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
    df = df.sample(frac=0.7, random_state=np_rng)

    return df


@pytest.fixture
def sample_nlme_params() -> dict:
    input = {
        "model_intrinsic": {"k_a": {"prior": 10.0}},
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


def test_sdk(obs_data, sample_nlme_params):
    # Gather all inputs
    path_to_model = "vpop_calibration/test/simwork_model/assets/model.json"
    path_to_solving_options = "vpop_calibration/test/simwork_model/assets/options.json"
    inputs = ["k_12", "k_21", "k_el", "dose", "k_a"]
    outputs = ["A0", "A1", "A2"]
    protocol_design = pd.DataFrame(
        {"protocol_arm": ["dose-1", "dose-10"], "dose": [1, 10]}
    )
    config = Config()

    # 1. Create model
    model = create_nlme_model(
        data_table=obs_data,
        user_input=sample_nlme_params,
        config=config,
        model_path=path_to_model,
        solving_options_path=path_to_solving_options,
        protocol_design=protocol_design,
        struct_model_inputs=inputs,
        struct_model_outputs=outputs,
        categorical_attributes=None,
    )

    # 2. Run optimizer
    history = run_saem(model)

    # 3. Run diagnostics
    diag_config = DiagnosticsConfig()
    out = run_diagnostics(model=model, config=diag_config)
