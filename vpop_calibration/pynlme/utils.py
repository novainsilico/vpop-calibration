from vpop_calibration.pynlme.params import (
    PatientDescriptorUnknown,
    ModelIntrinsicParam,
    Constraint,
    MixedEffectParameters,
    TransformFunction,
)
from vpop_calibration.config import device

import torch
from typing import Callable
import numpy as np
import scipy.stats as stats
from scipy.special import expit
from typing import get_args
import matplotlib.pyplot as plt


def init_transform_tensors(
    param_dict: dict[str, PatientDescriptorUnknown] | dict[str, ModelIntrinsicParam],
    param_names: list[str],
) -> tuple[dict[str, torch.LongTensor], torch.Tensor, torch.Tensor]:
    """Extract transform functions and parameters (scale and shift) into tensors for efficient gaussian parameters transformation."""

    transforms = {
        "exp": torch.LongTensor(
            torch.tensor(
                [
                    param_names.index(p_name)
                    for p_name, p_content in param_dict.items()
                    if p_content.constraint.transform == "log"
                ],
                device=device,
                dtype=torch.long,
            )
        ),
        "sigmoid": torch.LongTensor(
            torch.tensor(
                [
                    param_names.index(p_name)
                    for p_name, p_content in param_dict.items()
                    if p_content.constraint.transform == "logit"
                ],
                device=device,
                dtype=torch.long,
            )
        ),
    }
    scale = torch.Tensor(
        [[[param_dict[param].constraint.scale for param in param_names]]]
    )

    shift = torch.Tensor(
        [[[param_dict[param].constraint.shift for param in param_names]]]
    )
    return transforms, shift, scale


def init_transform_function(
    param_dict: dict[str, PatientDescriptorUnknown] | dict[str, ModelIntrinsicParam],
    param_names: list[str],
) -> Callable:

    transforms, shift, scale = init_transform_tensors(
        param_dict=param_dict, param_names=param_names
    )

    def transform(params: torch.Tensor) -> torch.Tensor:

        new_params_raw = torch.zeros_like(params, device=device)
        new_params_raw[:, :, transforms["exp"]] = torch.exp(
            params[:, :, transforms["exp"]]
        )
        new_params_raw[:, :, transforms["sigmoid"]] = torch.sigmoid(
            params[:, :, transforms["sigmoid"]]
        )
        new_params_shifted = shift + scale * new_params_raw

        return new_params_shifted

    return transform


def inverse_transform_param(phi: np.ndarray, const: Constraint) -> np.ndarray:
    if const.transform == "log":
        return np.exp(phi) + const.shift
    elif const.transform == "logit":
        return expit(phi) * const.scale + const.shift
    else:
        raise NotImplementedError(
            f"The following transforms are currently supported: {get_args(TransformFunction)}"
        )


def theoretical_pdf(
    x: np.ndarray, mu: float, omega: float, const: Constraint
) -> np.ndarray:

    pdf = np.zeros_like(x)

    if const.transform == "log":
        mask = x > const.shift
        x_valid = x[mask]

        phi = np.log(x_valid - const.shift)
        derivative = 1.0 / (x_valid - const.shift)

        pdf[mask] = stats.norm.pdf(phi, loc=mu, scale=omega) * derivative

    elif const.transform == "logit":

        mask = (x > const.shift) & (x < const.shift + const.scale)
        x_valid = x[mask]

        shifted_x = (x_valid - const.shift) / const.scale
        phi = np.log(shifted_x / (1.0 - shifted_x))
        derivative = 1.0 / (const.scale * shifted_x * (1.0 - shifted_x))

        pdf[mask] = stats.norm.pdf(phi, loc=mu, scale=omega) * derivative

    else:
        raise NotImplementedError(
            f"The following transforms are currently supported: {get_args(TransformFunction)}"
        )

    return pdf


class PriorVisualizer:
    def __init__(self, nlme_model: Callable):
        self.nlme_model = nlme_model

    def plot_prior_distribution(self, log_scale: bool = True):

        pdu_dict = self.nlme_model.statistical_model.prior_params.pdu
        if not pdu_dict:
            print("No PDU's defined")
            return

        fig, axes = self._setup_grid(len(pdu_dict))

        for idx, (name, pdu) in enumerate(pdu_dict.items()):
            ax = axes[idx]
            mu, omega = pdu.transformed_prior, pdu.prior_omega

            x_grid = self._make_x_grid(
                mu - 3.5 * omega, mu + 3.5 * omega, pdu.constraint, log_scale
            )

            self._draw_density(
                ax,
                x_grid,
                mu,
                omega,
                pdu.constraint,
                color="steelblue",
                label=f"Prior initial: {pdu.prior:.1f}",
            )
            self._format_axis(ax, name, pdu.constraint.transform, log_scale)

        self._show_grid(fig, axes, len(pdu_dict))

    def compare_initial_estimated(self, log_scale: bool = True):

        if torch.equal(
            self.nlme_model.statistical_model.population_betas,
            self.nlme_model.statistical_model.init_beta,
        ):
            print("You have not run SAEM. Optimising now ...")
            self.nlme_model.optimizer.run()

        pdu_dict = self.nlme_model.statistical_model.prior_params.pdu

        beta_names = self.nlme_model.statistical_model.beta_names
        updated_betas = (
            self.nlme_model.statistical_model.population_betas.cpu().detach().numpy()
        )
        updated_omega = (
            self.nlme_model.statistical_model.omega_pop.cpu().detach().numpy()
        )

        fig, axes = self._setup_grid(len(pdu_dict))

        for idx, (name, pdu) in enumerate(pdu_dict.items()):
            ax = axes[idx]

            mu_init, omega_init = pdu.transformed_prior, pdu.prior_omega

            beta_idx = beta_names.index(name)
            mu_est = updated_betas[beta_idx]
            omega_est = np.sqrt(updated_omega[idx, idx])

            phi_min = min(mu_init - 3.5 * omega_init, mu_est - 3.5 * omega_est)
            phi_max = max(mu_init + 3.5 * omega_init, mu_est + 3.5 * omega_est)
            x_grid = self._make_x_grid(phi_min, phi_max, pdu.constraint, log_scale)

            self._draw_density(
                ax,
                x_grid,
                mu_init,
                omega_init,
                pdu.constraint,
                color="steelblue",
                label="Initial",
            )
            self._draw_density(
                ax,
                x_grid,
                mu_est,
                omega_est,
                pdu.constraint,
                color="darkorange",
                label="Estimated (SAEM)",
            )

            self._format_axis(ax, name, pdu.constraint.transform, log_scale)

        self._show_grid(fig, axes, len(pdu_dict))

    def _setup_grid(self, n_items: int):
        cols = min(3, n_items)
        rows = (n_items + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if n_items == 1:
            axes = np.array([axes])
        return fig, axes.flatten()

    def _make_x_grid(
        self, phi_min: float, phi_max: float, constraint: Constraint, log_scale: bool
    ):

        theta_min = inverse_transform_param(np.array(phi_min), constraint)
        theta_max = inverse_transform_param(np.array(phi_max), constraint)
        if log_scale and theta_min > 0:
            return np.geomspace(theta_min, theta_max, 500)
        else:
            return np.linspace(theta_min, theta_max, 500)

    def _draw_density(self, ax, x_grid, mu, omega, constraint, color, label):
        y = theoretical_pdf(x_grid, mu, omega, constraint)
        ax.plot(x_grid, y, color=color, label=label)
        ax.fill_between(x_grid, y, color=color, alpha=0.3)

        median = inverse_transform_param(np.array(mu), constraint)
        ax.axvline(median, color=color, linestyle="--", alpha=0.8)

    def _format_axis(self, ax, name: str, transform_name: str, log_scale: bool):
        ax.set_title(f"{name} (Transform: {transform_name})")
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Density")

        if log_scale:
            ax.set_xscale("log")
        ax.legend()

    def _show_grid(self, fig, axes, n_items: int):

        for j in range(n_items, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()
