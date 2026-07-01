from typing import NamedTuple
import torch


class PopEstimates(NamedTuple):
    beta: torch.Tensor
    omega: torch.Tensor
    psi: torch.Tensor
    sigma: torch.Tensor
    complete_likelihood: torch.Tensor


def check_convergence(
    prev_est: PopEstimates, current_est: PopEstimates, threshold: float
):
    """Checks for convergence based on the relative change in parameters."""
    all_converged = True
    variables_to_check = ["beta", "omega", "psi", "sigma"]
    for name in variables_to_check:
        current_val = current_est._asdict()[name]
        prev_val = prev_est._asdict()[name]
        abs_diff = torch.abs(current_val - prev_val)
        abs_sum = torch.abs(current_val) + torch.abs(prev_val) + 1e-9
        relative_change = abs_diff / abs_sum
        if torch.any(relative_change > threshold):
            all_converged = False
            break
    return all_converged
