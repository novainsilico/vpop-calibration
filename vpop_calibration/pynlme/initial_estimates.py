import numpy as np
import scipy.stats as stats
from typing import get_args
import matplotlib.pyplot as plt

from vpop_calibration.pynlme.params import Constraint, TransformFunction
from vpop_calibration.pynlme.utils import inverse_transform_param
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.config import smoke_test


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

    def __init__(self, initial_estimates: MixedEffectParameters):
        self.params = initial_estimates

    def plot_distribution(self, log_scale: bool = True) -> None:
        pdu_dict = self.params.pdu
        if not pdu_dict:
            raise ValueError("No PDU's have been defined")

        fig, axes = self._setup_grid(len(pdu_dict))
        for ax, (name, pdu) in zip(axes, pdu_dict.items()):
            mu, omega = pdu.transformed_prior, pdu.prior_omega
            x_grid = self._make_x_grid(
                phi_min=mu - 3.5 * omega,
                phi_max=mu + 3.5 * omega,
                constraint=pdu.constraint,
                log_scale=log_scale,
            )
            self._draw_density(
                ax,
                x_grid,
                mu,
                omega,
                pdu.constraint,
                color="steelblue",
                label=f"Initial prior: {pdu.prior:.1f}",
                log_scale=log_scale,
            )
            self._format_axis(ax, name, pdu.constraint.transform, log_scale)

        self._show_grid(fig, axes, len(pdu_dict))

    def _setup_grid(self, n_items: int):
        cols = min(3, n_items)
        rows = (n_items + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        return fig, np.atleast_1d(axes).flatten()

    def _make_x_grid(
        self, phi_min: float, phi_max: float, constraint: Constraint, log_scale: bool
    ) -> np.ndarray:

        theta_min = (
            constraint.low
            if constraint.low is not None
            else inverse_transform_param(np.array(phi_min), constraint)
        )
        theta_max = (
            constraint.high
            if constraint.high is not None
            else inverse_transform_param(np.array(phi_max), constraint)
        )
        if log_scale:
            if theta_min > 0:
                return np.geomspace(theta_min, theta_max, 500)
            else:
                raise ValueError(
                    f"Unexpected negative value in physical space {theta_min}"
                )
        else:
            return np.linspace(theta_min, theta_max, 500)

    def _draw_density(
        self,
        ax,
        x_grid: np.ndarray,
        mu: float,
        omega: float,
        constraint: Constraint,
        color: str,
        label: str,
        log_scale: bool,
    ) -> None:
        y = theoretical_pdf(x_grid, mu, omega, constraint)
        if log_scale:
            y = y * x_grid * np.log(10)
        ax.plot(x_grid, y, color=color, label=label)
        ax.fill_between(x_grid, y, color=color, alpha=0.3)
        median = inverse_transform_param(np.array(mu), constraint)
        ax.axvline(median, color=color, linestyle="--", alpha=0.8)
        if constraint.low is not None:
            ax.axvline(constraint.low, color="crimson", alpha=0.8, label="constraints")
        if constraint.high is not None:
            ax.axvline(constraint.high, color="crimson", alpha=0.8)

    def _format_axis(self, ax, name: str, transform_name: str, log_scale: bool) -> None:
        ax.set_title(f"{name} (Transform: {transform_name})")
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Density")
        if log_scale:
            ax.set_xscale("log")
        ax.legend()

    def _show_grid(self, fig, axes, n_items: int) -> None:
        for j in range(n_items, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        if not smoke_test:
            plt.tight_layout()
            plt.show()
        plt.close(fig)
