from vpop_calibration.saem.estimates import PopEstimates, check_convergence, IterSummary

import torch


def test_check_convergence():
    tensor_1 = torch.tensor([0, 0])
    tensor_2 = torch.tensor([0, 0.2])

    prev_estimates = PopEstimates(
        beta=tensor_1,
        omega=tensor_1,
        psi=tensor_1,
        sigma=tensor_1,
        complete_likelihood=tensor_1,
        model_intrinsic=tensor_1,
    )
    current_estimates = PopEstimates(
        beta=tensor_1,
        omega=tensor_1,
        psi=tensor_2,
        sigma=tensor_1,
        complete_likelihood=tensor_1,
        model_intrinsic=tensor_2,
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
    sigma = torch.tensor([1.0, 1.0])
    output_names = ["out_1", "out_2"]

    pop_estimates = PopEstimates(
        beta=beta,
        omega=omega,
        model_intrinsic=mi,
        sigma=sigma,
        psi=torch.tensor([0.0]),
        complete_likelihood=torch.tensor([0.0]),
    )

    summary = IterSummary.from_pop_estimates(
        iteration=0,
        estimates=pop_estimates,
        beta_names=beta_names,
        pdu_names=pdu_names,
        covariate_coeff_names=cov_coeff_names,
        mi_names=mi_names,
        output_names=output_names,
    )
