from vpop_calibration.saem.scheduler import (
    SaemScheduler,
    PopEstimates,
    check_convergence,
)

import torch


def test_scheduler():
    scheduler = SaemScheduler(
        nb_iter_burnin=2,
        nb_iter_learning=2,
        nb_iter_smoothing=2,
        init_step_size_adaptation=0.5,
        learning_rate_power=0.8,
        patience=5,
    )

    output = []
    for _ in scheduler:
        output.append(
            (
                scheduler.iteration,
                scheduler.phase,
                scheduler.mh_learning_rate,
                scheduler.stochastic_approximation_rate,
            )
        )

    expected_output = [
        (0, "burnin", 0.5, 1.0),
        (1, "burnin", 0.5, 1.0),
        (2, "learning", 0.5, 1.0),
        (3, "learning", 0.5 / (2**0.5), 1.0),
        (4, "smoothing", 0.0, 1.0),
        (5, "smoothing", 0.0, 1 / 2**0.8),
    ]

    assert output == expected_output


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
