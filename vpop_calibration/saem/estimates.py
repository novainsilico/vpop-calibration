from typing import NamedTuple, Any
import torch
import pandas as pd

from vpop_calibration.config import device, default_dtype
from vpop_calibration.pynlme.residuals import ResidualErrorEstimates


class PopEstimates(NamedTuple):
    beta: torch.Tensor
    omega: torch.Tensor
    ebe: torch.Tensor
    sigma: ResidualErrorEstimates
    model_intrinsic: torch.Tensor
    surv_coeffs: torch.Tensor
    complete_likelihood: torch.Tensor
    fixed_effects_loss: torch.Tensor

    def get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v.detach().cpu().numpy().tolist()
            for k, v in self._asdict().items()
            if k != "sigma"
        }
        state_dict["sigma"] = self.sigma.get_state_dict()
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "PopEstimates":
        return cls(
            sigma=ResidualErrorEstimates.from_state_dict(state_dict["sigma"]),
            **{
                k: torch.as_tensor(v, device=device, dtype=default_dtype)
                for k, v in state_dict.items()
                if k != "sigma"
            },
        )

    def __eq__(self, other) -> bool:
        compared_attributes = [
            "beta",
            "omega",
            "ebe",
            "sigma",
            "model_intrinsic",
            "surv_coeffs",
            "complete_likelihood",
            "fixed_effects_loss",
        ]

        for elem in compared_attributes:
            torch.testing.assert_close(
                getattr(self, elem), getattr(other, elem), equal_nan=True
            )
        return True


def check_convergence(
    prev_est: PopEstimates, current_est: PopEstimates, threshold: float
):
    """Checks for convergence based on the relative change in parameters."""
    variables_to_check = ["beta", "omega", "ebe", "model_intrinsic", "surv_coeffs"]
    compared_pairs = [
        (current_est._asdict()[name], prev_est._asdict()[name])
        for name in variables_to_check
    ]
    # The residual error model keeps its two variances in separate tensors
    compared_pairs += [
        (current_est.sigma.sigma_add, prev_est.sigma.sigma_add),
        (current_est.sigma.sigma_prop, prev_est.sigma.sigma_prop),
    ]
    for current_val, prev_val in compared_pairs:
        abs_diff = torch.abs(current_val - prev_val)
        abs_sum = torch.abs(current_val) + torch.abs(prev_val) + 1e-9
        relative_change = abs_diff / abs_sum
        if torch.any(relative_change > threshold):
            return False
    return True


class IterSummary(NamedTuple):
    iteration: int
    mu: dict[str, float]
    omega: dict[str, float]
    model_intrinsic: dict[str, float]
    surv_coeffs: dict[str, float]
    cov: dict[str, float]
    sigma: dict[str, float]
    convergence_indicator: float
    fixed_effects_loss: float

    @property
    def headers(self) -> list[tuple[dict, str]]:
        header_tuples = [
            (self.mu, "mu_"),
            (self.omega, "omega_"),
            (self.model_intrinsic, ""),
            (self.surv_coeffs, ""),
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
        surv_coeffs_names: list[str],
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
        surv_coeffs_dict: dict[str, float] = {
            coef: estimates.surv_coeffs[surv_coeffs_names.index(coef)].item()
            for coef in surv_coeffs_names
        }
        sigma_dict: dict[str, float] = {}
        for i, (output, error_type) in enumerate(
            zip(output_names, estimates.sigma.error_types)
        ):
            if error_type == "combined":
                sigma_dict[f"{output}_add"] = estimates.sigma.sigma_add[i].item()
                sigma_dict[f"{output}_prop"] = estimates.sigma.sigma_prop[i].item()
            elif error_type == "additive":
                sigma_dict[output] = estimates.sigma.sigma_add[i].item()
            else:
                sigma_dict[output] = estimates.sigma.sigma_prop[i].item()

        return IterSummary(
            iteration=iteration,
            mu=mu_dict,
            omega=omega_dict,
            model_intrinsic=mi_dict,
            cov=cov_dict,
            sigma=sigma_dict,
            convergence_indicator=estimates.complete_likelihood.item(),
            fixed_effects_loss=estimates.fixed_effects_loss.item(),
            surv_coeffs=surv_coeffs_dict,
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
        combined_dicts.update(
            {
                "convergence_indicator": self.convergence_indicator,
                "fixed_effects_loss": self.fixed_effects_loss,
            }
        )
        df = pd.DataFrame([combined_dicts])
        df.insert(0, "iteration", self.iteration)

        return df


def dict_values_to_str(d: dict[str, float], width: int) -> str:
    return ", ".join(f"{v:<{width}.2f}" for v in d.values())


def dict_keys_to_str(d: dict[str, float], width: int, prefix: str = "") -> str:
    return ", ".join(f"{prefix + k:<{width}}" for k in d.keys())
