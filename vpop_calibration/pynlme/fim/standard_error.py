import torch


def invert_fim(fim: torch.Tensor) -> torch.Tensor:
    """Invert the observed FIM to get the covariance matrix of the estimates."""
    eigvals = torch.linalg.eigvalsh(fim)
    min_eig = eigvals.min()
    tol = eigvals.abs().max() * fim.shape[-1] * torch.finfo(fim.dtype).eps

    if min_eig > tol:
        chol = torch.linalg.cholesky(fim)
        return torch.cholesky_inverse(chol)
    return torch.linalg.pinv(fim)


def compute_standard_errors(
    covariance_matrix: torch.Tensor, parameter_names: list[str]
) -> torch.Tensor:
    """Standard errors of the estimates, ``NaN`` where the variance is negative."""
    variances = torch.diagonal(covariance_matrix)
    negative = variances < 0
    variances = variances.masked_fill(negative, float("nan"))
    return torch.sqrt(variances)


def compute_relative_standard_errors(
    standard_errors: torch.Tensor, estimates: torch.Tensor
) -> torch.Tensor:
    """Relative Standard Errors (RSE), in percentage."""
    return (standard_errors / torch.abs(estimates)) * 100
