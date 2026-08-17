import pytest
import pandas as pd
import numpy as np

from vpop_calibration.structural_model.sbml import StructuralSbml
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.api.interface import (
    NlmeModel,
    Config,
    SaemConfigDict,
    NlmeConfigDict,
)


@pytest.fixture
def sbml_model() -> StructuralModel:

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["dose-1", "dose-10"], "dose": [1, 10]}
    )
    struct_model = StructuralSbml(
        model_path="vpop_calibration/test/sbml/assets/model.xml",
        inputs=["k__12", "k__21", "k__el", "dose", "k__a"],
        outputs=["A0", "A1", "A2"],
        protocol_design=protocol_design,
    )
    return struct_model


@pytest.fixture
def obs_data(np_rng) -> pd.DataFrame:
    protocol_arms = ["dose-1", "dose-10"]
    patients = {
        "id": ["p1", "p2"],
        "k__el": [0.01, 0.1],
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
        "model_intrinsic": {"k__a": {"prior": 10.0}},
        "pdu": {
            "k__12": {
                "prior": 10.0,
                "prior_omega": 0.1,
            },
            "k__21": {
                "prior": 10.0,
                "prior_omega": 0.1,
            },
        },
        "error_model": {
            "A0": {"error_type": "additive", "sigma": 0.1},
            "A1": {"error_type": "additive", "sigma": 0.5},
            "A2": {"error_type": "additive", "sigma": 0.5},
        },
        "pdk": ["k__el"],
    }
    return input


def test_sbml_saem(sample_nlme_params, obs_data, sbml_model):
    config = Config(saem=SaemConfigDict(), nlme=NlmeConfigDict(nb_chains=1))
    nlme_model = NlmeModel(
        structural_model=sbml_model,
        df=obs_data,
        prior_params=sample_nlme_params,
        config=config,
    )
    nlme_model.optimizer.run()
    nlme_model.diagnostics.sample_conditional_distribution()
