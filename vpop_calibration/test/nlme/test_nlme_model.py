import pytest
import pandas as pd
import numpy as np
from pandera.typing import DataFrame
import torch
import math

from vpop_calibration.nlme_model.data import ObsData
from vpop_calibration.nlme_model.params import MixedEffectParameters
from vpop_calibration.nlme_model.model import NlmeModel
from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.structural_model.analytical import StructuralAnalytical


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
            "out_1": {"error_type": "additive", "sigma": 0.1},
            "out_2": {"error_type": "proportional", "sigma": 0.5},
        },
        "pdk": ["pdk_1"],
    }
    return MixedEffectParameters.model_validate(input)


@pytest.fixture
def obs_data(np_rng) -> ObsData:
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
    data = ObsData(DataFrame(df))
    return data


@pytest.fixture
def struct_model() -> StructuralModel:
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
    return struct_model


def test_nlme_init(sample_nlme_params, obs_data, struct_model):
    nlme_model = NlmeModel(
        structural_model=struct_model, dataset=obs_data, prior_params=sample_nlme_params
    )
    assert nlme_model.descriptors == ["pdk_1", "pdu_1", "pdu_2", "mi_1"]

    # Ensure the initial omega matrix is diagonal with correct values
    torch.testing.assert_close(
        nlme_model.omega_pop,
        torch.diag(
            torch.as_tensor(
                [
                    sample_nlme_params.pdu["pdu_1"].prior_omega,
                    sample_nlme_params.pdu["pdu_2"].prior_omega,
                ]
            )
        ),
    )

    # Ensure the etas distribution is properly initiated
    assert nlme_model.eta_distribution.batch_shape == torch.Size(
        [nlme_model.nb_patients]
    )
    assert nlme_model.eta_distribution.event_shape == torch.Size([nlme_model.nb_pdu])

    # Ensure the residual error tensor is properly initialized
    torch.testing.assert_close(
        nlme_model.residual_var,
        torch.as_tensor(
            [
                sample_nlme_params.error_model["out_1"].sigma,
                sample_nlme_params.error_model["out_2"].sigma,
            ]
        ),
    )

    # Ensure the design matrices are properly initialized
    p1_foo = obs_data.patients_df.loc[obs_data.patients_df["id"] == "p1", "foo"].iloc[0]
    expected_design_matrix_p1 = torch.as_tensor(
        [[1.0, p1_foo, 0.0, 0.0], [0.0, 0.0, 1.0, p1_foo]]
    )
    torch.testing.assert_close(
        nlme_model.design_matrices["p1"], expected_design_matrix_p1
    )

    p2_foo = obs_data.patients_df.loc[obs_data.patients_df["id"] == "p2", "foo"].iloc[0]
    expected_design_matrix_p2 = torch.as_tensor(
        [[1.0, p2_foo, 0.0, 0.0], [0.0, 0.0, 1.0, p2_foo]]
    )
    torch.testing.assert_close(
        nlme_model.design_matrices["p2"], expected_design_matrix_p2
    )

    # Ensure full design matrix is properly stacked
    torch.testing.assert_close(
        nlme_model.full_design_matrix,
        torch.stack((expected_design_matrix_p1, expected_design_matrix_p2)),
    )

    # Ensure PDKs are properly loaded
    p1_pdk = obs_data.patients_df.loc[obs_data.patients_df["id"] == "p1", "pdk_1"].iloc[
        0
    ]
    torch.testing.assert_close(
        nlme_model.data.patients_pdk["p1"], torch.tensor([[p1_pdk]])
    )

    p2_pdk = obs_data.patients_df.loc[obs_data.patients_df["id"] == "p2", "pdk_1"].iloc[
        0
    ]
    torch.testing.assert_close(
        nlme_model.data.patients_pdk["p1"], torch.tensor([[p2_pdk]]), check_dtype=False
    )
    torch.testing.assert_close(
        nlme_model.data.patients_pdk_full, torch.tensor([[p1_pdk], [p2_pdk]])
    )


def test_nlme_simulate(sample_nlme_params, obs_data, struct_model):
    nlme_model = NlmeModel(
        structural_model=struct_model, dataset=obs_data, prior_params=sample_nlme_params
    )
    nb_samples = 1
    etas = nlme_model.sample_etas(nb_samples)
    # nullify etas to test the proper PDU and MI transformation
    etas = torch.zeros_like(etas)
    assert etas.shape == (nb_samples, nlme_model.nb_patients, nlme_model.nb_pdu)

    psi = nlme_model.convert_etas_to_gaussian(etas)
    assert psi.shape == (nb_samples, nlme_model.nb_patients, nlme_model.nb_pdu)

    phi = nlme_model.convert_gaussian_to_physical(psi, nlme_model.log_mi)

    # Ensure the parameter converted back to physical space are actually equal to the prior values: this is only valid for patient 1 as it has a foo covariate = 0.
    pdu_1_prior = sample_nlme_params.pdu["pdu_1"].prior
    pdu_2_prior = sample_nlme_params.pdu["pdu_2"].prior
    mi_1_prior = sample_nlme_params.model_intrinsic["mi_1"].prior
    assert phi[0, 0, 0].item() == pytest.approx(pdu_1_prior)
    assert phi[0, 0, 1].item() == pytest.approx(pdu_2_prior)
    assert phi[0, 0, 2].item() == pytest.approx(mi_1_prior)

    theta = nlme_model.convert_physical_to_thetas(phi)
    # Double check the thetas are properly assembled and correspond to the priors for patient 1 (0 random effect, 0 covariate effect)
    p1_pdk = obs_data.patients_df.loc[obs_data.patients_df["id"] == "p1", "pdk_1"].iloc[
        0
    ]
    torch.testing.assert_close(
        theta[0, 0, :],
        torch.tensor([p1_pdk, pdu_1_prior, pdu_2_prior, mi_1_prior]),
        check_dtype=False,
    )

    model_inputs = nlme_model.convert_thetas_to_model_parameters(theta)

    outs = nlme_model.predict(model_inputs)
    assert outs[0].shape == (nb_samples, nlme_model.data.nb_total_observations)


def test_log_prior(sample_nlme_params, obs_data, struct_model):
    nlme_model = NlmeModel(
        structural_model=struct_model, dataset=obs_data, prior_params=sample_nlme_params
    )
    nb_samples = 1
    etas = nlme_model.sample_etas(nb_samples)
    etas = torch.zeros_like(etas)
    # Test the log prior function for etas
    log_prior_etas = nlme_model.log_prior_etas(etas)
    # Omega is diagonal, computing its determinant is straightforward
    omega_det = (
        sample_nlme_params.pdu["pdu_1"].prior_omega
        * sample_nlme_params.pdu["pdu_2"].prior_omega
    )
    k = nlme_model.nb_pdu
    # Analytical formula for the log probability with samples all = 0
    expected_log_prior = -0.5 * (k * np.log(2 * math.pi) + np.log(omega_det))
    torch.testing.assert_close(
        log_prior_etas,
        torch.tensor(expected_log_prior).expand(nb_samples, nlme_model.nb_patients),
    )


def test_log_posterior(sample_nlme_params, obs_data, struct_model):
    nlme_model = NlmeModel(
        structural_model=struct_model, dataset=obs_data, prior_params=sample_nlme_params
    )
    nb_samples = 1
    etas = nlme_model.sample_etas(nb_samples)
    etas = torch.zeros_like(etas)
    # Test the log prior function for etas
    predictions = nlme_model.log_posterior_etas(etas)
    assert predictions.log_posterior.shape == (nb_samples, nlme_model.nb_patients)
    # No analytical value here :sadface:, if someone has the courage to write it feel free
