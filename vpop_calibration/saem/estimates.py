from typing import NamedTuple
import torch


class PopEstimates(NamedTuple):
    beta: torch.Tensor
    omega: torch.Tensor
    psi: torch.Tensor
    sigma: torch.Tensor
    model_intrinsic: torch.Tensor
    complete_likelihood: torch.Tensor


def check_convergence(
    prev_est: PopEstimates, current_est: PopEstimates, threshold: float
):
    """Checks for convergence based on the relative change in parameters."""
    all_converged = True
    variables_to_check = ["beta", "omega", "psi", "sigma", "model_intrinsic"]
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


class IterSummary(NamedTuple):
    iteration: int
    mu: dict[str, float]
    omega: dict[str, float]
    model_intrinsic: dict[str, float]
    cov: dict[str, float]
    sigma: dict[str, float]

    @classmethod
    def from_pop_estimates(
        cls,
        iteration: int,
        estimates: PopEstimates,
        beta_names: list[str],
        pdu_names: list[str],
        covariate_coeff_names: list[str],
        mi_names: list[str],
        output_names: list[str],
    ) -> "IterSummary":
        mu_dict: dict[str, float] = {
            pdu: estimates.beta[beta_names.index(pdu)].item() for pdu in pdu_names
        }
        omega_dict: dict[str, float] = {
            pdu: estimates.omega[i, i].item() for i, pdu in enumerate(pdu_names)
        }
        mi_dict: dict[str, float] = {
            mi: estimates.model_intrinsic[i].item() for i, mi in enumerate(mi_names)
        }
        cov_dict: dict[str, float] = {
            cov: estimates.beta[beta_names.index(cov)].item()
            for cov in covariate_coeff_names
        }
        sigma_dict: dict[str, float] = {
            output: estimates.sigma[i].item() for i, output in enumerate(output_names)
        }

        return IterSummary(
            iteration=iteration,
            mu=mu_dict,
            omega=omega_dict,
            model_intrinsic=mi_dict,
            cov=cov_dict,
            sigma=sigma_dict,
        )
