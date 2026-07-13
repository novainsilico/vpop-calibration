import torch
import torch.distributions as dist
import numpy as np

from vpop_calibration.pynlme.diagnostics import ModelDiagnostics


class ImportanceSampler:
    def __init__(self, diagnostics: ModelDiagnostics):
        self.model_diag = diagnostics
        self.mu: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.df: float | None = None
        self.dist: dist.StudentT | None = None

    def fit_sudent_t_proposal(self, df: float = 5.0, jitter: float = 1e-5) -> None:
        # samples shape: (N,P,D) - N samples, P patients, D parameters

        if not hasattr(self.model_diag.sampler, "ebe"):
            self.model_diag.sample_conditional_distribution()
        samples = self.model_diag.sampler.total_samples.eta_samples

        N, P, D = samples.shape

        self.mu = torch.mean(samples, dim=0)  # (P,D)
        centered_samples = samples - self.mu.unsqueeze(0)
        samples_X = centered_samples.permute(1, 2, 0)  # (P,D,N)
        samples_T = centered_samples.permute(1, 0, 2)  # (P,N,D)
        sigma = torch.bmm(samples_X, samples_T) / (N - 1)  # (P,D,D)
        eye = torch.eye(D).unsqueeze(0).expand(P, -1, -1)
        sigma = sigma + eye * jitter

        self.scale = sigma

        self.df = df

        self.dist = dist.StudentT(df=torch.tensor(self.df), loc=self.mu, scale=sigma)

    def _student_t_log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        # Computes the log-density of a multivariate Student's t-distribution:
        if self.dist is None:
            self.fit_sudent_t_proposal()
        log_prob = self.dist.log_prob(samples)
        return log_prob.sum(dim=-1)

    def _generate_student_samples(self, nb_samples: int) -> torch.Tensor:

        if self.dist is None:
            self.fit_sudent_t_proposal()

        samples = self.dist.rsample((nb_samples,))
        return samples

    def compute_likelihood(self, nb_samples: int = 100) -> float:

        samples = self._generate_student_samples(nb_samples=nb_samples)

        log_q = self._student_t_log_prob(samples=samples)

        predictions = self.model_diag.model.log_posterior_etas_all_patients(samples)
        log_posterior = predictions.log_posterior

        N_tensor = torch.tensor(nb_samples)

        marginal_log_lik = torch.logsumexp(log_posterior - log_q, dim=0) - torch.log(
            N_tensor
        )
        log_lik = marginal_log_lik.sum()

        return log_lik.item()
