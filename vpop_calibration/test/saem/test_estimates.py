from vpop_calibration.saem.estimates import PopEstimates, check_convergence, IterSummary
from vpop_calibration.pynlme.residuals import ResidualErrorEstimates

import torch


def test_check_convergence():
    tensor_1 = torch.tensor([0, 0])
    tensor_2 = torch.tensor([0, 0.2])
    sigma = ResidualErrorEstimates(
        sigma_add=torch.tensor([1.0, 0.0]),
        sigma_prop=torch.tensor([0.0, 1.0]),
        additive_output=torch.tensor([True, False]),
        proportional_output=torch.tensor([False, True]),
    )

    prev_estimates = PopEstimates(
        beta=tensor_1,
        omega_lower_chol=tensor_1,
        ebe=tensor_1,
        sigma=sigma,
        complete_likelihood=tensor_1,
        model_intrinsic=tensor_1,
        fixed_effects_loss=tensor_1,
        surv_coeffs=tensor_1,
    )
    current_estimates = PopEstimates(
        beta=tensor_1,
        omega_lower_chol=tensor_1,
        ebe=tensor_2,
        sigma=sigma,
        complete_likelihood=tensor_1,
        model_intrinsic=tensor_2,
        fixed_effects_loss=tensor_1,
        surv_coeffs=tensor_1,
    )

    assert check_convergence(
        prev_est=prev_estimates, current_est=prev_estimates, threshold=0.01
    )
    assert check_convergence(
        prev_est=prev_estimates, current_est=current_estimates, threshold=1.0
    )
    assert not check_convergence(
        prev_est=prev_estimates, current_est=current_estimates, threshold=0.1
    )


def test_iter_summary():
    beta = torch.tensor([0.0, 0.0, 0.0])
    omega = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    beta_names = ["pdu_1", "cov_1", "pdu_2"]
    pdu_names = ["pdu_1", "pdu_2"]
    mi = torch.tensor([0.0])
    mi_names = ["mi_1"]
    cov_coeff_names = ["cov_1"]
    sigma = ResidualErrorEstimates(
        sigma_add=torch.tensor([1.0, 0.0]),
        sigma_prop=torch.tensor([0.0, 1.0]),
        additive_output=torch.tensor([True, False]),
        proportional_output=torch.tensor([False, True]),
    )
    output_names = ["out_1", "out_2"]

    pop_estimates = PopEstimates(
        beta=beta,
        omega_lower_chol=omega,
        model_intrinsic=mi,
        sigma=sigma,
        ebe=torch.tensor([0.0]),
        complete_likelihood=torch.tensor([0.0]),
        fixed_effects_loss=torch.tensor([0.0]),
        surv_coeffs=torch.tensor([]),
    )

    _summary = IterSummary.from_pop_estimates(
        iteration=0,
        estimates=pop_estimates,
        beta_names=beta_names,
        pdu_names=pdu_names,
        covariate_coeff_names=cov_coeff_names,
        mi_names=mi_names,
        output_names=output_names,
        surv_coeffs_names=[],
    )


def test_state_dict():
    beta = torch.tensor([0.0, 0.0, 0.0])
    omega = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mi = torch.tensor([0.0])
    sigma = ResidualErrorEstimates(
        sigma_add=torch.tensor([1.0, 0.0]),
        sigma_prop=torch.tensor([0.0, 1.0]),
        additive_output=torch.tensor([True, False]),
        proportional_output=torch.tensor([False, True]),
    )

    pop_estimates = PopEstimates(
        beta=beta,
        omega_lower_chol=omega,
        model_intrinsic=mi,
        sigma=sigma,
        ebe=torch.tensor([0.0]),
        complete_likelihood=torch.tensor([0.0]),
        fixed_effects_loss=torch.tensor([0.0]),
        surv_coeffs=torch.tensor([0.0]),
    )

    state_dict = pop_estimates.get_state_dict()
    new_estimates = PopEstimates.from_state_dict(state_dict)

    assert pop_estimates == new_estimates
