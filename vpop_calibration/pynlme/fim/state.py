from typing import Any, NamedTuple
import torch
from vpop_calibration.config import default_dtype, device
from vpop_calibration.pynlme.fim.standard_error import invert_fim


class FimComponents(NamedTuple):
    score: torch.Tensor
    hessian: torch.Tensor
    score_outer_product: torch.Tensor


class FimState(NamedTuple):
    score: torch.Tensor
    hessian: torch.Tensor
    score_outer_product: torch.Tensor
    variance_history: tuple[tuple[float, ...], ...] = ()
    nb_samples: int = 0

    @classmethod
    def initialize(cls, nb_params: int) -> "FimState":
        return cls(
            score=torch.zeros(nb_params, device=device, dtype=default_dtype),
            hessian=torch.zeros(
                (nb_params, nb_params), device=device, dtype=default_dtype
            ),
            score_outer_product=torch.zeros(
                (nb_params, nb_params), device=device, dtype=default_dtype
            ),
            nb_samples=0,
            variance_history=(),
        )

    @property
    def fim(self) -> torch.Tensor:
        fim = -(
            self.hessian
            + self.score_outer_product
            - torch.outer(self.score, self.score)
        )
        return 0.5 * (fim + fim.transpose(-1, -2))

    def accumulate(
        self,
        statistics: FimComponents,
        nb_new: int = 1,
        max_history: int | None = None,
        alpha: float = 0.7,
    ) -> "FimState":
        total = self.nb_samples + nb_new

        iteration_actuelle = total / nb_new
        weight = 1.0 / (iteration_actuelle**alpha)

        def running_mean(previous: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            return previous + weight * (new - previous)

        updated = self._replace(
            score=running_mean(self.score, statistics.score),
            hessian=running_mean(self.hessian, statistics.hessian),
            score_outer_product=running_mean(
                self.score_outer_product, statistics.score_outer_product
            ),
            nb_samples=total,
        )

        inv_fim = invert_fim(updated.fim)
        variances = torch.diagonal(inv_fim).tolist()

        new_history = self.variance_history + (tuple(variances),)
        if max_history is not None:
            new_history = new_history[-max_history:]

        return updated._replace(variance_history=new_history)

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "score": self.score.detach().cpu().numpy().tolist(),
            "hessian": self.hessian.detach().cpu().numpy().tolist(),
            "score_outer_product": self.score_outer_product.detach()
            .cpu()
            .numpy()
            .tolist(),
            "nb_samples": self.nb_samples,
            "variance_history": [list(x) for x in self.variance_history],
        }

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "FimState":
        def as_tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(state_dict[key], device=device, dtype=default_dtype)

        var_history = tuple(tuple(x) for x in state_dict.get("variance_history", []))
        return cls(
            score=as_tensor("score"),
            hessian=as_tensor("hessian"),
            score_outer_product=as_tensor("score_outer_product"),
            nb_samples=state_dict.get("nb_samples", len(var_history)),
            variance_history=var_history,
        )

    def __eq__(self, other) -> bool:
        for field in ("score", "hessian", "score_outer_product"):
            torch.testing.assert_close(getattr(self, field), getattr(other, field))
        return self.nb_samples == other.nb_samples
