from typing import NamedTuple, Any
import torch
import pandas as pd


class PopEstimates(NamedTuple):
    beta: torch.Tensor
    omega: torch.Tensor
    psi: torch.Tensor
    sigma: torch.Tensor
    model_intrinsic: torch.Tensor
    complete_likelihood: torch.Tensor

    def get_state_dict(self) -> dict[str, Any]:
        return {k: v.detach().cpu().numpy().tolist() for k, v in self._asdict().items()}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "PopEstimates":
        return cls(**{k: torch.as_tensor(v) for k, v in state_dict.items()})


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
    convergence_indicator: float

    @property
    def headers(self) -> list[tuple[dict, str]]:
        header_tuples = [
            (self.mu, "mu_"),
            (self.omega, "omega_"),
            (self.model_intrinsic, ""),
            (self.cov, ""),
            (self.sigma, "sigma_"),
        ]
        return header_tuples

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
            convergence_indicator=estimates.complete_likelihood.item(),
        )

    def print(self, width: int):
        if self.iteration == 0:
            header = self._console_header(width)
        else:
            header = ""
        out_str_list = [f"{self.iteration:<{width}}"]
        for d in [self.mu, self.omega, self.model_intrinsic, self.cov, self.sigma]:
            if d:
                out_str_list.append(dict_values_to_str(d, width))
        out_str = header + ", ".join(out_str_list)
        print(out_str)

    def _console_header(self, width: int) -> str:
        out_str_list = [
            f"{'iteration':<{width}}",
        ]
        for d, prefix in self.headers:
            if d:
                out_str_list.append(dict_keys_to_str(d, width, prefix))
        out_str = ", ".join(out_str_list) + "\n"
        return out_str

    def to_pandas(self) -> pd.DataFrame:
        combined_dicts = {}
        for d, prefix in self.headers:
            for k, v in d.items():
                combined_dicts.update({prefix + k: v})
        combined_dicts.update({"convergence_indicator": self.convergence_indicator})
        df = pd.DataFrame([combined_dicts])
        df.insert(0, "iteration", self.iteration)

        return df


def dict_values_to_str(d: dict[str, float], width: int) -> str:
    return ", ".join(f"{v:<{width}.2f}" for v in d.values())


def dict_keys_to_str(d: dict[str, float], width: int, prefix: str = "") -> str:
    return ", ".join(f"{prefix+k:<{width}}" for k in d.keys())
