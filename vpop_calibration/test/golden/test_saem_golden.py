import pytest
import pandas as pd
import numpy as np
import torch
from deepdiff import DeepDiff
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical
from vpop_calibration.interface import NlmeModel

torch.set_num_threads(1)


@pytest.fixture
def sample_nlme_params() -> dict:
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
    return input


@pytest.fixture(scope="function")
def obs_data() -> pd.DataFrame:
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
    df["value"] = np.linspace(0, 10, df.shape[0])
    df["task"] = df.apply(lambda r: r["output_name"] + "_" + r["protocol_arm"], axis=1)
    return df


@pytest.fixture
def struct_model() -> StructuralModel:
    def equations(mi_1, pdu_1, pdu_2, pdk_1, t, protocol_ovr_1):
        out_1 = t * pdu_1 + pdu_2
        out_2 = pdk_1 - mi_1 + t * protocol_ovr_1
        return torch.cat((out_1, out_2), dim=-1)

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["arm-A", "arm-B"], "protocol_ovr_1": [1, 2]}
    )
    struct_model = StructuralAnalytical(
        equations=equations,
        variable_names=["out_1", "out_2"],
        protocol_design=protocol_design,
    )
    return struct_model


@pytest.mark.golden_test("stored_results/test_saem_golden.yml")
def test_saem_golden(sample_nlme_params, obs_data, struct_model, golden, request):
    nlme_model = NlmeModel(
        structural_model=struct_model,
        df=obs_data,
        prior_params=sample_nlme_params,
    )
    nlme_model.optimizer.run()
    nlme_model.diagnostics.sample_conditional_distribution()

    actual = nlme_model.get_state_dict()
    expected = golden.out["output"]

    if request.config.getoption("--update-goldens"):
        # In update mode, use the native __eq__ comparison to trigger the actual golden test update from pytest-golden
        assert actual == expected
    else:
        # In normal test run, use deepdiff for a more granular comparison between the expected and actual
        diff_strict = DeepDiff(
            actual,
            expected,
            ignore_type_in_groups=[(tuple, list), (float, np.float64)],
            exclude_paths=["root['optimizer']['fim_estimator']"],
            math_epsilon=1e-16,
        )
        assert diff_strict == {}

        actual_fim = actual.get("optimizer", {}).get("fim_estimator", {})
        expected_fim = expected.get("optimizer", {}).get("fim_estimator", {})

        diff_fim = DeepDiff(
            actual_fim,
            expected_fim,
            ignore_type_in_groups=[(tuple, list), (float, np.float64)],
            math_epsilon=1e-8,
        )
        assert diff_fim == {}
