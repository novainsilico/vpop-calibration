from typing import Literal, NamedTuple
import torch
import numpy as np


class SaemScheduler:
    def __init__(
        self,
        nb_iter_burnin: int,
        nb_iter_learning: int,
        nb_iter_smoothing: int,
        init_step_adaptation: float,
        learning_rate_power: float,
        patience: int,
    ):
        """Scheduler class for SAEM iterations and variable tuning parameters (learning rates)."""
        self.nb_iter_burnin = nb_iter_burnin
        self.nb_iter_learning = nb_iter_learning
        self.nb_iter_smoothing = nb_iter_smoothing
        self.init_step_adaptation = init_step_adaptation
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
            return self.init_step_adaptation
        elif self.phase == "learning":
            return self.init_step_adaptation / (
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
