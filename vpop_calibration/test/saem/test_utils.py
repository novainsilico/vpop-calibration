from vpop_calibration.saem.utils import (
    cov_to_corr,
    simulated_annealing,
    stochastic_approximation,
    covariance_matrix_simulated_annealing,
    clamp_eigen_values,
)

import torch


def test_cov_to_corr():
    cov = torch.tensor([[2, 0], [1, 2]])
    cor = torch.tensor([[1, 0], [1 / 2, 1]])
    torch.testing.assert_close(cor, cov_to_corr(cov))


def test_simulated_annealing():
    a = torch.tensor([1.0, 1.0])
    b = torch.tensor([0.2, 0.6])
    out = simulated_annealing(current=a, target=b, factor=0.5)
    expected_out = torch.tensor([0.5, 0.6])
    torch.testing.assert_close(out, expected_out)


def test_cov_simulated_annealing():
    current_cov = torch.tensor([[1, 0], [0, 2]])
    target_cov = torch.tensor([[0.2, 0], [0, 2]])
    new_cov = covariance_matrix_simulated_annealing(
        current_omega=current_cov, target_omega=target_cov, factor=0.5
    )
    expected_out = torch.tensor([[0.5, 0], [0.0, 2.0]])
    torch.testing.assert_close(new_cov, expected_out)


def test_stochastic_approx():
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([1.0, 2.0])
    lr = 0.5
    expected_out = torch.tensor([0.5, 1.0])
    out = stochastic_approximation(previous=a, new=b, learning_rate=lr)
    torch.testing.assert_close(out, expected_out)


def test_clamp_eig():
    a = torch.tensor([[1.0, 0], [0.0, 1e-3]])
    new_a = clamp_eigen_values(omega=a, min_eigenvalue=1e-2)
    torch.testing.assert_close(new_a, torch.tensor([[1.0, 0.0], [0.0, 1e-2]]))
