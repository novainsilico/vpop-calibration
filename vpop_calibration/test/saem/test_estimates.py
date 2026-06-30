from vpop_calibration.saem.estimates import PopEstimates, check_convergence

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
    )
    current_estimates = PopEstimates(
        beta=tensor_1,
        omega=tensor_1,
        psi=tensor_2,
        sigma=tensor_1,
        complete_likelihood=tensor_1,
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
