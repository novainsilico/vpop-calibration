from typing import Literal, NamedTuple
import torch
import numpy as np


class SaemScheduler:
    def __init__(
        self,
        nb_iter_burnin: int,
        nb_iter_learning: int,
        nb_iter_smoothing: int,
        init_step_size_adaptation: float,
        learning_rate_power: float,
        patience: int,
    ):
        """Scheduler class for SAEM iterations and variable tuning parameters (learning rates)."""
        self.nb_iter_burnin = nb_iter_burnin
        self.nb_iter_learning = nb_iter_learning
        self.nb_iter_smoothing = nb_iter_smoothing
        self.init_step_size_adaptation = init_step_size_adaptation
        self.learning_rate_power = learning_rate_power
        self.patience = patience

        self.iteration = 0

    def __iter__(self):
        while self.iteration < self.nb_iter_tot:
            yield self.iteration
            self.iteration += 1

    @property
    def nb_iter_tot(self) -> int:
        return self.nb_iter_burnin + self.nb_iter_learning + self.nb_iter_smoothing

    @property
    def phase(self) -> Literal["burnin", "learning", "smoothing"]:
        if self.iteration < self.nb_iter_burnin:
            return "burnin"
        elif self.iteration < self.nb_iter_burnin + self.nb_iter_learning:
            return "learning"
        else:
            return "smoothing"

    @property
    def mh_learning_rate(self) -> float:
        if self.phase == "burnin":
            return self.init_step_size_adaptation
        elif self.phase == "learning":
            return self.init_step_size_adaptation / (
                np.maximum(1, self.iteration - self.nb_iter_burnin + 1) ** 0.5
            )
        elif self.phase == "smoothing":
            return 0
        else:
            raise NotImplemented

    @property
    def stochastic_approximation_rate(self) -> float:
        if self.phase == "burnin":
            return 1.0
        elif self.phase == "learning":
            return 1.0
        elif self.phase == "smoothing":
            return (
                1
                / (self.iteration - self.nb_iter_burnin - self.nb_iter_learning + 1)
                ** self.learning_rate_power
            )
        else:
            raise NotImplemented


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
