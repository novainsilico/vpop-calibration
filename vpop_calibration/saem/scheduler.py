from typing import Literal, Any
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
        nb_iter_fim: int = 0,
    ):
        """Scheduler class for SAEM iterations and variable tuning parameters (learning rates)."""
        self.nb_iter_burnin = nb_iter_burnin
        self.nb_iter_learning = nb_iter_learning
        self.nb_iter_smoothing = nb_iter_smoothing
        self.nb_iter_fim = nb_iter_fim

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
    def phase(self) -> Literal["burnin", "learning", "smoothing", "fim"]:
        if self.iteration < self.nb_iter_burnin:
            return "burnin"
        elif self.iteration < self.nb_iter_burnin + self.nb_iter_learning:
            return "learning"
        elif (
            self.iteration
            < self.nb_iter_burnin + self.nb_iter_learning + self.nb_iter_smoothing
        ):
            return "smoothing"
        else:
            return "fim"

    @property
    def mh_learning_rate(self) -> float:
        if self.phase == "burnin":
            return self.init_step_adaptation
        elif self.phase == "learning":
            return self.init_step_adaptation / (
                np.maximum(1, self.iteration - self.nb_iter_burnin + 1) ** 0.5
            )
        elif self.phase == ["smoothing", "fim"]:
            return 0
        else:
            raise NotImplementedError

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
        elif self.phase == "fim":
            k = self.iteration - (
                self.nb_iter_burnin + self.nb_iter_learning + self.nb_iter_smoothing
            )
            return 1.0 / (k + 1.0)
        else:
            raise NotImplementedError

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "nb_iter_burnin": self.nb_iter_burnin,
            "nb_iter_learning": self.nb_iter_learning,
            "nb_iter_smoothing": self.nb_iter_smoothing,
            "nb_iter_fim": self.nb_iter_fim,
            "init_step_adaptation": self.init_step_adaptation,
            "learning_rate_power": self.learning_rate_power,
            "patience": self.patience,
            "iteration": self.iteration,
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "SaemScheduler":
        instance = cls(
            nb_iter_burnin=state_dict["nb_iter_burnin"],
            nb_iter_learning=state_dict["nb_iter_learning"],
            nb_iter_smoothing=state_dict["nb_iter_smoothing"],
            nb_iter_fim=state_dict["nb_iter_fim"],
            init_step_adaptation=state_dict["init_step_adaptation"],
            learning_rate_power=state_dict["learning_rate_power"],
            patience=state_dict["patience"],
        )
        instance.iteration = state_dict["iteration"]
        return instance
