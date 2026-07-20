import torch
import torch.distributions as dist
from typing import Any

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.conditional_distribution import ConditionalDistribSamples
from vpop_calibration.config import smoke_test, device, default_dtype


class ImportanceSampler:
    def __init__(self, model: StatisticalModel, df: float = 5.0):
        self.model = model
        self.dist: dist.StudentT | None = None
        self.df = df

    def get_state_dict(self) -> dict[str, Any]:
        if self.dist is None:
            loc = None
            scale = None
        else:
            loc = self.dist.loc.detach().cpu().numpy().tolist()
            scale = self.dist.scale.detach().cpu().numpy().tolist()
        if hasattr(self, "log_lik"):
            ll = self.log_lik
        else:
            ll = None
        state = {"df": self.df, "loc": loc, "scale": scale, "log_lik": ll}
        return state

    @classmethod
    def from_state_dict(
        cls, model: StatisticalModel, state_dict: dict[str, Any]
    ) -> "ImportanceSampler":
        instance = cls(model=model, df=state_dict["df"])
        if state_dict["loc"]:
            instance.dist = dist.StudentT(
                df=torch.tensor(instance.df),
                loc=torch.as_tensor(
                    state_dict["loc"], device=device, dtype=default_dtype
                ),
                scale=torch.as_tensor(
                    state_dict["scale"], device=device, dtype=default_dtype
                ),
            )
        if state_dict["log_lik"]:
            instance.log_lik = state_dict["log_lik"]

        return instance

    def fit_sudent_t_proposal(
        self, conditional_samples: ConditionalDistribSamples
    ) -> None:
        etas = conditional_samples.eta_samples

        nb_samples, nb_patients, nb_pdu = etas.shape
        assert (
            nb_samples > 1
        ), "Need more than one sample to estimate the student distribution."

        mu = torch.mean(etas, dim=0)
        sigma = torch.clamp(torch.var(etas, 0), 1e-6)

        self.dist = dist.StudentT(df=torch.tensor(self.df), loc=mu, scale=sigma)

    def _student_t_log_prob(self, student_samples: torch.Tensor) -> torch.Tensor:
        # Computes the log-density of a multivariate Student's t-distribution:
        assert (
            self.dist is not None
        ), "Invalid call of `_student_t_log_prob` before `fit_student_t_proposal`"
        log_prob = self.dist.log_prob(student_samples)
        return log_prob.sum(dim=-1)

    def _generate_student_samples(self, nb_samples: int) -> torch.Tensor:
        assert (
            self.dist is not None
        ), "Invalid call of `_generate_student_samples` before `fit_student_t_proposal`"

        samples = self.dist.rsample((nb_samples,))
        return samples

    def compute_likelihood(self, nb_samples: int = 100) -> None:

        if smoke_test:
            nb_samples = 2

        student_samples = self._generate_student_samples(nb_samples=nb_samples)

        log_q = self._student_t_log_prob(student_samples=student_samples)

        predictions = self.model.log_posterior_etas_all_patients(student_samples)
        log_posterior = predictions.log_posterior

        N_tensor = torch.tensor(nb_samples)

        marginal_log_lik = torch.logsumexp(log_posterior - log_q, dim=0) - torch.log(
            N_tensor
        )
        log_lik = marginal_log_lik.sum()
        self.log_lik = log_lik.item()
