from typing import Any, NamedTuple
import torch
from vpop_calibration.config import default_dtype, device
from vpop_calibration.utils import stochastic_approximation
import numpy as np


class FimComponents(NamedTuple):
    score: torch.Tensor
    hessian: torch.Tensor
    score_outer_product: torch.Tensor

    @classmethod
    def initialize(cls, nb_params: int) -> "FimComponents":
        return cls(
            score=torch.zeros(nb_params, device=device, dtype=default_dtype),
            hessian=torch.zeros(
                (nb_params, nb_params), device=device, dtype=default_dtype
            ),
            score_outer_product=torch.zeros(
                (nb_params, nb_params), device=device, dtype=default_dtype
            ),
        )

    @property
    def cov_scores(self) -> torch.Tensor:
        return self.score_outer_product - torch.outer(self.score, self.score)

    @property
    def fim(self) -> torch.Tensor:
        fim = -(self.hessian + self.cov_scores)
        return 0.5 * (fim + fim.T)

    def get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v.detach().cpu().numpy().tolist() for k, v in self._asdict().items()
        }
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "FimComponents":
        return cls(
            **{
                k: torch.as_tensor(v, device=device, dtype=default_dtype)
                for k, v in state_dict.items()
            },
        )

    def __eq__(self, other) -> bool:
        for field in ("score", "hessian", "score_outer_product"):
            torch.testing.assert_close(getattr(self, field), getattr(other, field))
        return True


class FimState(NamedTuple):
    running_average: FimComponents
    nb_params: int
    nb_burnin: int
    history_size: int
    fim_diagonal_history: np.ndarray
    learning_rate_decay_exponent: float
    nb_iters: int = 0
    nb_samples: int = 0

    @classmethod
    def initialize(
        cls,
        nb_params: int,
        nb_burnin: int,
        history_size: int,
        learning_rate_decay_exponent: float,
    ) -> "FimState":
        init_components = FimComponents.initialize(nb_params=nb_params)
        init_history = np.zeros((0, nb_params))
        return cls(
            running_average=init_components,
            nb_params=nb_params,
            nb_burnin=nb_burnin,
            history_size=history_size,
            fim_diagonal_history=init_history,
            learning_rate_decay_exponent=learning_rate_decay_exponent,
        )

    @property
    def fim(self) -> torch.Tensor | None:
        if self.nb_iters < self.nb_burnin:
            print("Warning: the FIM burn-in is not over, the FIM is undefined")
            return None
        else:
            return self.running_average.fim

    def accumulate(
        self,
        statistics: FimComponents,
    ) -> "FimState":
        """Given new estimates of the FIM components, accumulate the running average and update the state."""
        nb_iters = self.nb_iters + 1
        if nb_iters > self.nb_burnin:
            nb_samples = self.nb_samples + 1
            learning_rate = 1.0 / (
                (nb_iters - self.nb_burnin) ** self.learning_rate_decay_exponent
            )

            new_components = FimComponents(
                score=stochastic_approximation(
                    previous=self.running_average.score,
                    new=statistics.score,
                    learning_rate=learning_rate,
                ),
                hessian=stochastic_approximation(
                    previous=self.running_average.hessian,
                    new=statistics.hessian,
                    learning_rate=learning_rate,
                ),
                score_outer_product=stochastic_approximation(
                    previous=self.running_average.score_outer_product,
                    new=statistics.score_outer_product,
                    learning_rate=learning_rate,
                ),
            )

            diagonal_fim = torch.diag(new_components.fim)
            new_history = np.vstack((self.fim_diagonal_history, diagonal_fim))
            clamped_history = new_history[-self.history_size :,]
            new_state = FimState(
                running_average=new_components,
                nb_params=self.nb_params,
                nb_burnin=self.nb_burnin,
                history_size=self.history_size,
                fim_diagonal_history=clamped_history,
                nb_iters=nb_iters,
                nb_samples=nb_samples,
                learning_rate_decay_exponent=self.learning_rate_decay_exponent,
            )
        else:
            new_state = self._replace(running_average=statistics, nb_iters=nb_iters)
        return new_state

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "running_average": self.running_average.get_state_dict(),
            "nb_samples": self.nb_samples,
            "nb_burnin": self.nb_burnin,
            "nb_iters": self.nb_iters,
            "nb_params": self.nb_params,
            "history_size": self.history_size,
            "learning_rate_decay_exponent": self.learning_rate_decay_exponent,
            "fim_diagonal_history": self.fim_diagonal_history.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "FimState":
        running_average = FimComponents.from_state_dict(state_dict["running_average"])
        variance_history = np.asarray(state_dict["fim_diagonal_history"])
        return cls(
            running_average=running_average,
            nb_params=state_dict["nb_params"],
            nb_burnin=state_dict["nb_burnin"],
            nb_samples=state_dict["nb_samples"],
            nb_iters=state_dict["nb_iters"],
            history_size=state_dict["history_size"],
            learning_rate_decay_exponent=state_dict["learning_rate_decay_exponent"],
            fim_diagonal_history=variance_history,
        )
