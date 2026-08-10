import torch
import pandas as pd
import numpy as np
from IPython.display import display
from tqdm.notebook import tqdm

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import (
    log_likelihood_observation,
    ResidualErrorEstimates,
)
from vpop_calibration.saem.utils import stochastic_approximation
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step
from typing import Any
from vpop_calibration.config import device, default_dtype, smoke_test


class Fim:
    def __init__(self, model: StatisticalModel):
        self.model = model
        self.tril_idx = torch.tril_indices(model.nb_pdu, model.nb_pdu, device=device)
        self.sigma_mask = torch.cat(
            (
                model.residual_var.additive_output,
                model.residual_var.proportional_output,
            )
        )
        self.parameter_names = self._build_parameter_names()
        nb_params = len(self.parameter_names)

        # The three stochastic approximations of Louis' formula
        self.score = torch.zeros(nb_params, device=device, dtype=default_dtype)
        self.hessian = torch.zeros(
            (nb_params, nb_params), device=device, dtype=default_dtype
        )
        self.score_outer_product = torch.zeros(
            (nb_params, nb_params), device=device, dtype=default_dtype
        )

    # --- Parameter vector handling
    def _build_parameter_names(self) -> list[str]:
        model = self.model
        names = list(model.beta_names)
        names += [
            f"omega_{model.pdu_names[i]}_{model.pdu_names[j]}"
            for i, j in zip(*self.tril_idx.tolist())
        ]
        names += list(model.mi_names)
        sigma_names = [
            f"{component}_{output}"
            for component in ("sigma_add", "sigma_prop")
            for output in model.output_names
        ]
        names += [
            name
            for name, active in zip(sigma_names, self.sigma_mask.tolist())
            if active
        ]
        return names

    def flatten_parameters(self) -> torch.Tensor:
        """Current population parameters, as a flat vector (no gradient attached)."""
        model = self.model
        blocks = [
            model.population_betas,
            model.omega_pop[self.tril_idx[0], self.tril_idx[1]],
            model.log_mi,
            torch.cat((model.residual_var.sigma_add, model.residual_var.sigma_prop))[
                self.sigma_mask
            ],
        ]
        return torch.cat(
            [block.detach().flatten().to(default_dtype) for block in blocks]
        )

    def _unflatten(
        self, flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ResidualErrorEstimates]:
        """Rebuild the model parameters from the flat vector, keeping the autograd graph."""
        model = self.model
        cursor = 0

        beta = flat[cursor : cursor + model.nb_betas]
        cursor += model.nb_betas

        nb_omega = self.tril_idx.shape[1]
        lower = torch.zeros(
            (model.nb_pdu, model.nb_pdu), device=device, dtype=flat.dtype
        ).index_put(
            (self.tril_idx[0], self.tril_idx[1]), flat[cursor : cursor + nb_omega]
        )
        omega = lower + lower.tril(-1).transpose(-1, -2)
        cursor += nb_omega

        if model.nb_mi == 0:
            log_mi = torch.empty(0, device=device, dtype=flat.dtype)
        else:
            log_mi = flat[cursor : cursor + model.nb_mi]
        cursor += model.nb_mi

        idx = self.sigma_mask.nonzero(as_tuple=True)[0]
        full_sigma = torch.zeros(
            2 * model.nb_outputs, device=device, dtype=flat.dtype
        ).index_put((idx,), flat[cursor:])
        sigma_add, sigma_prop = full_sigma.chunk(2)

        res_var = model.residual_var._replace(
            sigma_add=sigma_add, sigma_prop=sigma_prop
        )

        return beta, omega, log_mi, res_var

    def _complete_log_likelihood(
        self, flat: torch.Tensor, gaussian_params: torch.Tensor
    ) -> torch.Tensor:
        """`log p(y, psi ; theta)` at fixed individual parameters, averaged over the chains.
        Args:
            flat (torch.Tensor): the population parameters, flattened.
            gaussian_params (torch.Tensor): individual parameters sampled in the E-step.
                Size (nb_chains, nb_patients, nb_pdu).
        """
        model = self.model
        nb_chains = gaussian_params.shape[0]
        beta, omega, log_mi, res_var = self._unflatten(flat)

        # Observation term: log p(y | psi, sigma, mi)
        physical_params = model.convert_gaussian_to_physical(gaussian_params, log_mi)
        thetas = model.convert_physical_to_thetas_all_patients(physical_params)
        inputs = model.convert_thetas_to_model_parameters_all_patients(thetas)
        predictions, _ = model.predict_all_patients(inputs)
        log_lik_obs = log_likelihood_observation(
            observations=model.data.full_obs,
            predictions=predictions,
            residual_error=res_var,
            min_variance=model.config.residual_min_variance,
        )

        # Random effects term: psi_i ~ N(X_i @ beta, Omega)
        mu = model.full_design_matrix @ beta
        log_lik_psi = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=omega
        ).log_prob(gaussian_params)

        return (log_lik_obs + log_lik_psi).sum() / nb_chains

    # --- Stochastic approximation
    def update(self, gaussian_params: torch.Tensor, learning_rate: float) -> None:
        """Update the three Louis statistics with a new draw of the individual parameters."""
        nb_chains = gaussian_params.shape[0]
        flat = self.flatten_parameters()

        # Score of every chain, size (nb_chains, nb_params)
        theta = flat.clone().requires_grad_(True)
        scores = torch.stack(
            [
                torch.autograd.grad(
                    self._complete_log_likelihood(
                        theta, gaussian_params[chain : chain + 1]
                    ),
                    theta,
                )[0]
                for chain in range(nb_chains)
            ]
        )
        # Hessian of the chain-averaged complete log-likelihood, size (nb_params, nb_params)
        hessian = torch.autograd.functional.hessian(
            lambda t: self._complete_log_likelihood(t, gaussian_params), flat
        )

        self.score = stochastic_approximation(
            previous=self.score, new=scores.mean(dim=0), learning_rate=learning_rate
        )
        self.hessian = stochastic_approximation(
            previous=self.hessian, new=hessian, learning_rate=learning_rate
        )
        self.score_outer_product = stochastic_approximation(
            previous=self.score_outer_product,
            new=scores.transpose(0, 1) @ scores / nb_chains,
            learning_rate=learning_rate,
        )

    # --- Results
    @property
    def fim(self) -> torch.Tensor:
        """Observed Fisher Information Matrix, as given by Louis' formula."""
        fim = -(
            self.hessian
            + self.score_outer_product
            - torch.outer(self.score, self.score)
        )
        return 0.5 * (fim + fim.transpose(-1, -2))

    def show_fim(self) -> pd.DataFrame:
        """Returns the FIM as a Dataframe"""

        fim_np = self.fim.detach().cpu().numpy()
        names = self.parameter_names

        df_fim = pd.DataFrame(fim_np, index=names, columns=names)

        print("Fisher Information Matrix (FIM) :")
        display(df_fim)

        return df_fim

    @property
    def covariance_matrix(self) -> torch.Tensor:
        return torch.linalg.inv(self.fim)

    def show_covMatrix(self) -> pd.DataFrame:
        """Returns the covariance Matrix as a Dataframe"""

        cov_np = self.covariance_matrix.detach().cpu().numpy()
        names = self.parameter_names

        df_cov = pd.DataFrame(cov_np, index=names, columns=names)

        print("Covariance Matrix :")
        display(df_cov)

        return df_cov

    @property
    def standard_errors(self) -> torch.Tensor:
        return torch.sqrt(torch.diagonal(self.covariance_matrix))

    @property
    def rse(self) -> torch.Tensor:
        """Relative Standard Error (RSE)"""

        estimates = self.flatten_parameters()
        return (self.standard_errors / torch.abs(estimates)) * 100

    def show_RSE(self) -> pd.DataFrame:
        """Returns estimates, standard errors and relative standard errors"""
        import pandas as pd
        from IPython.display import display

        estimates_np = self.flatten_parameters().detach().cpu().numpy()
        se_np = self.standard_errors.detach().cpu().numpy()
        rse_np = self.rse.detach().cpu().numpy()
        names = self.parameter_names

        df_rse = pd.DataFrame(
            {"Estimate": estimates_np, "Standard Error": se_np, "RSE (%)": rse_np},
            index=names,
        )

        print("Standard Error:")
        display(df_rse)

        return df_rse

    def summary(self) -> pd.DataFrame:
        """Returns a summary of all parameters"""

        estimates = self.flatten_parameters().detach().cpu().numpy()
        se = self.standard_errors.detach().cpu().numpy()
        rse = self.rse.detach().cpu().numpy()
        names = self.parameter_names

        data = {
            name: {"Est": est, "SE": s, "RSE": r}
            for name, est, s, r in zip(names, estimates, se, rse)
        }

        summary_rows = []

        fixed_names = self.model.beta_names + self.model.mi_names
        for name in fixed_names:
            if name in data:
                summary_rows.append(
                    {
                        "Parameter": name,
                        "Estimate": data[name]["Est"],
                        "SE": data[name]["SE"],
                        "RSE (%)": data[name]["RSE"],
                        "BSV (CV%)": np.nan,
                    }
                )

        for pdu in self.model.pdu_names:
            omega_name = f"omega_{pdu}_{pdu}"
            if omega_name in data:
                var_val = data[omega_name]["Est"]
                cv_pct = np.sqrt(var_val) * 100 if var_val > 0 else np.nan

                found = False
                for row in summary_rows:
                    if row["Parameter"] == pdu:
                        row["BSV (CV%)"] = cv_pct
                        found = True
                        break

                if not found:
                    summary_rows.append(
                        {
                            "Parameter": f"BSV_{pdu}",
                            "Estimate": np.nan,
                            "SE": np.nan,
                            "RSE (%)": np.nan,
                            "BSV (CV%)": cv_pct,
                        }
                    )

        sigma_names = [n for n in names if n.startswith("sigma_")]
        for name in sigma_names:
            if name in data:
                summary_rows.append(
                    {
                        "Parameter": name,
                        "Estimate": data[name]["Est"],
                        "SE": data[name]["SE"],
                        "RSE (%)": data[name]["RSE"],
                        "BSV (CV%)": np.nan,
                    }
                )

        df_summary = pd.DataFrame(summary_rows)
        df_summary.set_index("Parameter", inplace=True)

        print("Summary of Population Parameters")
        display(df_summary)

        return df_summary

    def get_state_dict(self) -> dict[str, Any]:
        """Sauvegarde l'état actuel de l'approximation stochastique."""
        return {
            "score": self.score.detach().cpu().numpy().tolist(),
            "hessian": self.hessian.detach().cpu().numpy().tolist(),
            "score_outer_product": self.score_outer_product.detach()
            .cpu()
            .numpy()
            .tolist(),
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], model: StatisticalModel
    ) -> "Fim":
        """Recharge l'état de l'approximation stochastique."""
        instance = cls(model)
        instance.score = torch.as_tensor(
            state_dict["score"], device=device, dtype=default_dtype
        )
        instance.hessian = torch.as_tensor(
            state_dict["hessian"], device=device, dtype=default_dtype
        )
        instance.score_outer_product = torch.as_tensor(
            state_dict["score_outer_product"], device=device, dtype=default_dtype
        )
        return instance


def run_fim_sa_phase(
    model: StatisticalModel,
    fim: Fim,
    mh_state: MetropolisHastingsState,
    nb_iter: int = 50,
) -> MetropolisHastingsState:
    """Extra MCMC iterations at frozen population parameters (`covMethod = "sa"`)."""
    if smoke_test:
        nb_iter = 2
    for k in tqdm(range(nb_iter), disable=not model.config.progress_bar):
        mh_state = mh_step(nlme_model=model, previous_state=mh_state, learning_rate=0.0)
        fim.update(mh_state.gaussian_params, learning_rate=1.0 / (k + 1))
    return mh_state
