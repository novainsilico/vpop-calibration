import torch


def simulated_annealing(
    current: torch.Tensor, target: torch.Tensor, factor: float
) -> torch.Tensor:
    """Perform simulated annealing

    This function allows to constrain the reduction of certain values by a given factor

    Args:
        current (torch.Tensor): Current value of the tensor
        target (torch.Tensor): Target value of the tensor

    Returns:
        torch.Tensor: maximum(factor * current, target)
    """
    assert current.shape == target.shape, "Inconsistent shapes in simulated annealing"
    return torch.maximum(factor * current, target)


def cov_to_corr(covariance: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert a 2D covariance matrix into a correlation matrix, with float safeguard for division by 0."""
    assert covariance.dim() == 2

    std_dev = torch.sqrt(torch.diag(covariance))
    return covariance / (torch.outer(std_dev, std_dev) + eps)


def covariance_matrix_simulated_annealing(
    current_omega: torch.Tensor, target_omega: torch.Tensor, factor: float
):
    assert current_omega.shape == target_omega.shape, (
        "Inconsistent shapes in covariance matrix simulated annealing"
    )

    current_omega_diagonal = torch.diag(current_omega)
    target_omega_diagonal = torch.diag(target_omega)
    annealed_omega_diagonal = simulated_annealing(
        current_omega_diagonal, target_omega_diagonal, factor=factor
    )
    new_std_dev = torch.sqrt(annealed_omega_diagonal)
    corr_matrix = cov_to_corr(target_omega)
    new_omega = corr_matrix * torch.outer(new_std_dev, new_std_dev)
    return new_omega


def clamp_eigen_values(omega: torch.Tensor, min_eigenvalue: float = 1e-6):
    """
    Project a matrix onto the cone of Positive Definite matrices.
    """
    # 1. Ensure symmetry
    omega = 0.5 * (omega + omega.T)

    # 2. Eigen Decomposition
    L, V = torch.linalg.eigh(omega)

    # 3. Clamp eigenvalues
    L_clamped = torch.clamp(L, min=min_eigenvalue)

    # 4. Reconstruct
    matrix_spd = torch.matmul(V, torch.matmul(torch.diag(L_clamped), V.T))

    return matrix_spd
