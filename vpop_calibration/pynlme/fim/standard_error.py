import torch


def invert_fim(fim: torch.Tensor) -> torch.Tensor:
    """Invert the observed FIM to get the covariance matrix of the estimates."""
    chol, info = torch.linalg.cholesky_ex(fim)
    if info.sum() == 0:
        inverse = torch.cholesky_inverse(chol)
    else:
        inverse = torch.linalg.pinv(fim)

    return inverse


def compute_standard_errors(covariance_matrix: torch.Tensor) -> torch.Tensor:
    """Standard errors of the estimates, NaN where the variance is negative."""
    variances = torch.diagonal(covariance_matrix)
    negative = variances < 0
    variances = variances.masked_fill(negative, float("nan"))
    return torch.sqrt(variances)


def compute_relative_standard_errors(
    standard_errors: torch.Tensor, estimates: torch.Tensor
) -> torch.Tensor:
    """Relative Standard Errors (RSE)"""
    return (standard_errors / torch.abs(estimates)) * 100
