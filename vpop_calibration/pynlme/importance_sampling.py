import torch
import torch.distributions as dist
import numpy as np

from vpop_calibration.pynlme.diagnostics import ModelDiagnostics


class ImportanceSampler:
    def __init__(self, diagnostics: ModelDiagnostics):
        self.model_diag = diagnostics
        self.mu: torch.Tensor | None = None
        self.sigma: torch.Tensor | None = None
        self.df: float | None = None

    def fit_sudent_t_proposal(self, df: float = 5.0, jitter: float = 1e-5) -> None:
        # samples shape: (N,P,D) - N samples, P patients, D parameters

        if self.model_diag.conditional_distribution_samples is None:
            self.model_diag.sample_conditional_distribution()
        assert self.model_diag.conditional_distribution_samples is not None

        samples = self.model_diag.conditional_distribution_samples.samples

        N, P, D = samples.shape

        self.mu = torch.mean(samples, dim=0)  # (P,D)

        centered_samples = samples - self.mu.unsqueeze(0)

        samples_X = centered_samples.permute(1, 2, 0)  # (P,D,N)
        samples_T = centered_samples.permute(1, 0, 2)  # (P,N,D)

        self.sigma = torch.bmm(samples_X, samples_T) / (N - 1)  # (P,D,D)
        eye = torch.eye(D, device=samples.device).unsqueeze(0).expand(P, -1, -1)
        self.sigma = self.sigma + eye * jitter

        self.df = df

    def _multivariate_student_t_log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        # Computes the log-density of a multivariate Student's t-distribution:
        # log p = log Gamma((df+D)/2) - log Gamma(df/2) - D/2 log(df * pi) - 1/2 log(|sigma|) - (df+D)/2 log (1 + 1/df * centered_samples^T sigma^-1 centered_samples)
        if self.mu is None or self.sigma is None:
            self.fit_sudent_t_proposal()
        assert self.mu is not None and self.sigma is not None
        df = self.df
        N, P, D = samples.shape
        device = samples.device

        log_gamma_term = torch.lgamma(
            torch.tensor((df + D) / 2.0, device=device)
        ) - torch.lgamma(torch.tensor(df / 2.0, device=device))
        const_term = -(D / 2.0) * np.log(df * np.pi)

        L = torch.linalg.cholesky(self.sigma)  # shape: (P,D,D)

        log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # shape: (P,)
        det_term = -0.5 * log_det.unsqueeze(0)

        centered_samples = samples - self.mu.unsqueeze(0)
        centered_samples_T = centered_samples.permute(1, 2, 0)  # shape: (P,D,N)
        y = torch.linalg.solve_triangular(
            L, centered_samples_T, upper=False
        )  # shape: (P,D,N)
        normalized_sq_dist = (y**2).sum(dim=1)  # shape: (P,N)
        normalized_sq_dist = normalized_sq_dist.T  # shape: (N,P)

        dist_term = -((df + D) / 2.0) * torch.log(
            1.0 + (1.0 / df) * normalized_sq_dist
        )  # shape: (N,P)

        log_prob = log_gamma_term + const_term + det_term + dist_term

        return log_prob

    def _generate_multivariate_student_samples(self, nb_samples: int) -> torch.Tensor:

        # samples = mu + z/sqrt(u) avec z ~ Normale Multivariée(0, Sigma) et u ~ Gamma(df/2,df/2)

        if self.mu is None or self.sigma is None:
            self.fit_sudent_t_proposal()

        device = self.mu.device
        P, D = self.mu.shape
        N = nb_samples
        df = self.df

        gamma_dist = dist.Gamma(
            concentration=torch.tensor(df / 2.0, device=device),
            rate=torch.tensor(df / 2.0, device=device),
        )
        u = gamma_dist.sample((N, P)).unsqueeze(-1)  # shape: (N,P,1)

        mvn_dist = dist.MultivariateNormal(
            loc=torch.zeros_like(self.mu), covariance_matrix=self.sigma
        )
        z = mvn_dist.sample((N,))  # shape: (N, P, D)

        samples = self.mu.unsqueeze(0) + z * torch.rsqrt(u)
        return samples

    def compute_likelihood(self, nb_samples: int = 10000) -> float:

        samples = self._generate_multivariate_student_samples(nb_samples=nb_samples)

        log_q = self._multivariate_student_t_log_prob(samples=samples)

        predictions = self.model_diag.model.log_posterior_etas_all_patients(samples)
        log_posterior = predictions.log_posterior

        N_tensor = torch.tensor(nb_samples, dtype=torch.float32, device=samples.device)

        marginal_log_lik = torch.logsumexp(log_posterior - log_q, dim=0) - torch.log(
            N_tensor
        )
        log_lik = marginal_log_lik.sum()

        return log_lik.item()
