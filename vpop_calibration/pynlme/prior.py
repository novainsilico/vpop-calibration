import numpy as np
import torch
import scipy.stats as stats
from scipy.special import expit
from typing import get_args
import matplotlib.pyplot as plt

from vpop_calibration.pynlme.params import Constraint, TransformFunction
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.config import smoke_test


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

    def __init__(self, statistical_model: StatisticalModel):
        self.model = statistical_model

    def distribution(self, log_scale: bool = True):
        pdu_dict = self.model.prior_params.pdu
        if not pdu_dict:
            raise ValueError("Aucun PDU défini sur ce modèle.")

        fig, axes = self._setup_grid(len(pdu_dict))
        for ax, (name, pdu) in zip(axes, pdu_dict.items()):
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

        return self._show_grid(fig, axes, len(pdu_dict))

    def compare(self, log_scale: bool = True):
        model = self.model
        pdu_dict = model.prior_params.pdu
        beta_names = model.beta_names
        pdu_names = model.pdu_names
        updated_betas = model.population_betas.cpu().detach().numpy()
        updated_omega = model.omega_pop.cpu().detach().numpy()

        fig, axes = self._setup_grid(len(pdu_dict))
        for ax, (name, pdu) in zip(axes, pdu_dict.items()):
            mu_init, omega_init = pdu.transformed_prior, pdu.prior_omega
            mu_est = updated_betas[beta_names.index(name)]
            omega_est = np.sqrt(
                updated_omega[pdu_names.index(name), pdu_names.index(name)]
            )
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

        return self._show_grid(fig, axes, len(pdu_dict))

    def _setup_grid(self, n_items: int):
        cols = min(3, n_items)
        rows = (n_items + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        return fig, np.atleast_1d(axes).flatten()

    def _make_x_grid(self, phi_min, phi_max, constraint: Constraint, log_scale: bool):
        theta_min = inverse_transform_param(np.array(phi_min), constraint)
        theta_max = inverse_transform_param(np.array(phi_max), constraint)
        if log_scale and theta_min > 0:
            return np.geomspace(theta_min, theta_max, 500)
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
        if not smoke_test:
            plt.show()
        plt.close(fig)
        return fig
