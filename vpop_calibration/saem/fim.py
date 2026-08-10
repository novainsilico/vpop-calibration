import torch
import pandas as pd
import numpy as np
import warnings
from IPython.display import display

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import (
    log_likelihood_observation,
    ResidualErrorEstimates,
)
from vpop_calibration.saem.utils import stochastic_approximation
from typing import Any
from vpop_calibration.config import device, default_dtype


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
        self.fim_norm_history = []

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

    @property
    def _mi_slice(self) -> slice:
        """Location of the model-intrinsic parameters in the flat vector."""
        start = self.model.nb_betas + self.tril_idx.shape[1]
        return slice(start, start + self.model.nb_mi)

    def _predict_detached(
        self, log_mi: torch.Tensor, gaussian_params: torch.Tensor
    ) -> torch.Tensor:
        model = self.model
        physical = model.convert_gaussian_to_physical(gaussian_params, log_mi)
        thetas = model.convert_physical_to_thetas_all_patients(physical)
        inputs = model.convert_thetas_to_model_parameters_all_patients(thetas)
        predictions, _ = model.predict_all_patients(inputs)
        return predictions.detach()

    def _analytic_ll_per_chain(
        self,
        flat: torch.Tensor,
        predictions: torch.Tensor,
        gaussian_params: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        beta, omega, _log_mi, res_var = self._unflatten(flat)

        log_lik_obs = log_likelihood_observation(
            observations=model.data.full_obs,
            predictions=predictions,
            residual_error=res_var,
            min_variance=model.config.residual_min_variance,
        )
        mu = model.full_design_matrix @ beta
        log_lik_psi = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=omega
        ).log_prob(gaussian_params)
        return (log_lik_obs + log_lik_psi).sum(dim=1)

    def _analytic_louis_stats(
        self,
        flat: torch.Tensor,
        predictions: torch.Tensor,
        gaussian_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = flat.clone().requires_grad_(True)
        scores = torch.stack(
            [
                torch.autograd.grad(
                    self._analytic_ll_per_chain(
                        theta, predictions[c : c + 1], gaussian_params[c : c + 1]
                    ).sum(),
                    theta,
                )[0]
                for c in range(gaussian_params.shape[0])
            ]
        )
        hessian = torch.autograd.functional.hessian(
            lambda t: self._analytic_ll_per_chain(
                t, predictions, gaussian_params
            ).mean(),
            flat,
        )
        assert isinstance(hessian, torch.Tensor)
        return scores, hessian

    def _mi_finite_differences(
        self,
        flat: torch.Tensor,
        gaussian_params: torch.Tensor,
        base_predictions: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model
        theta = flat.clone().requires_grad_(True)
        log_mi0 = flat[self._mi_slice]
        h = eps * torch.clamp(log_mi0.abs(), min=1.0)
        e = torch.eye(model.nb_mi, device=device, dtype=flat.dtype) * h  # step vectors

        def ll(step: torch.Tensor) -> torch.Tensor:
            preds = self._predict_detached(log_mi0 + step, gaussian_params)
            return self._analytic_ll_per_chain(flat, preds, gaussian_params)

        def ll_and_grad(step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            preds = self._predict_detached(log_mi0 + step, gaussian_params)
            ll_step = self._analytic_ll_per_chain(flat, preds, gaussian_params)
            grad_step = torch.autograd.grad(
                self._analytic_ll_per_chain(theta, preds, gaussian_params).mean(),
                theta,
            )[0]
            return ll_step, grad_step

        ll_0 = self._analytic_ll_per_chain(flat, base_predictions, gaussian_params)
        plus = [ll_and_grad(e[k]) for k in range(model.nb_mi)]
        minus = [ll_and_grad(-e[k]) for k in range(model.nb_mi)]
        ll_p = torch.stack([p[0] for p in plus])
        ll_m = torch.stack([m[0] for m in minus])

        mi_scores = (ll_p - ll_m) / (2 * h.unsqueeze(1))
        cross = torch.stack(
            [(plus[k][1] - minus[k][1]) / (2 * h[k]) for k in range(model.nb_mi)],
            dim=1,
        )

        h_mm = torch.diag((ll_p - 2 * ll_0 + ll_m).mean(dim=1) / h**2)
        for i, j in zip(*torch.triu_indices(model.nb_mi, model.nb_mi, offset=1)):
            h_mm[i, j] = h_mm[j, i] = (
                ll(e[i] + e[j]) - ll(e[i] - e[j]) - ll(-e[i] + e[j]) + ll(-e[i] - e[j])
            ).mean() / (4 * h[i] * h[j])

        return mi_scores, h_mm, cross

    def _compute_louis_stats_hybrid(
        self, flat: torch.Tensor, gaussian_params: torch.Tensor, eps: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Louis' statistics: autograd on (beta, omega, sigma), finite differences
        on MI. Only `log_mi` flows through the structural model, so Omega is never
        perturbed numerically and every non-MI parameter is differentiated exactly.
        """
        nb_chains = gaussian_params.shape[0]
        predictions = self._predict_detached(flat[self._mi_slice], gaussian_params)
        scores, hessian = self._analytic_louis_stats(flat, predictions, gaussian_params)

        if self.model.nb_mi > 0:
            mi = self._mi_slice
            mi_scores, h_mm, cross = self._mi_finite_differences(
                flat, gaussian_params, predictions, eps
            )
            scores, hessian = scores.clone(), hessian.clone()
            scores[:, mi] = mi_scores.transpose(0, 1)
            hessian[:, mi] = cross
            hessian[mi, :] = cross.transpose(0, 1)
            hessian[mi, mi] = h_mm  # written last: overrides the ~0 cross entries

        return (
            scores.mean(dim=0),
            hessian,
            scores.transpose(0, 1) @ scores / nb_chains,
        )

    # --- Stochastic approximation
    def update(self, gaussian_params: torch.Tensor, learning_rate: float) -> None:
        flat = self.flatten_parameters()
        mean_score, hessian, score_outer_product = self._compute_louis_stats_hybrid(
            flat, gaussian_params
        )

        self.score = stochastic_approximation(
            previous=self.score, new=mean_score, learning_rate=learning_rate
        )
        self.hessian = stochastic_approximation(
            previous=self.hessian, new=hessian, learning_rate=learning_rate
        )
        self.score_outer_product = stochastic_approximation(
            previous=self.score_outer_product,
            new=score_outer_product,
            learning_rate=learning_rate,
        )
        self.fim_norm_history.append(torch.norm(self.fim).item())

    def get_history_df(self) -> pd.DataFrame:

        if not self.fim_norm_history:
            return pd.DataFrame()

        df = pd.DataFrame({"Global norm of FIM": self.fim_norm_history})
        df.insert(0, "iteration", range(1, len(df) + 1))

        return df

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

    def _invert_fim(self) -> torch.Tensor:
        """Invert the observed FIM to get the covariance matrix."""
        fim = self.fim
        # eigvalsh: real, sorted eigenvalues, valid because fim is symmetric
        eigvals = torch.linalg.eigvalsh(fim)
        min_eig = eigvals.min()
        tol = eigvals.abs().max() * fim.shape[-1] * torch.finfo(fim.dtype).eps

        if min_eig > tol:
            chol = torch.linalg.cholesky(fim)
            return torch.cholesky_inverse(chol)

        print(
            f"Observed FIM is not positive definite (smallest eigenvalue "
            f"{min_eig.item():.3e}): falling back to the pseudo-inverse. "
            f"Standard errors of the affected parameters are unreliable."
        )
        return torch.linalg.pinv(fim)

    @property
    def covariance_matrix(self) -> torch.Tensor:
        return self._invert_fim()

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
        variances = torch.diagonal(self.covariance_matrix)
        negative = variances < 0
        if torch.any(negative):
            bad = [
                self.parameter_names[i]
                for i in negative.nonzero(as_tuple=True)[0].tolist()
            ]
            warnings.warn(
                "Negative variance on the covariance diagonal for: "
                f"{', '.join(bad)}. These parameters are not identified by the "
                "data; their standard error is returned as NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
        variances = variances.masked_fill(negative, float("nan"))
        return torch.sqrt(variances)

    @property
    def rse(self) -> torch.Tensor:
        """Relative Standard Error (RSE)"""

        estimates = self.flatten_parameters()
        return (self.standard_errors / torch.abs(estimates)) * 100

    def show_RSE(self) -> pd.DataFrame:
        """Returns estimates, standard errors and relative standard errors"""

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

        def color_rse(val):
            """Applies color coding based on standard pharmacometrics RSE thresholds."""
            if pd.isna(val):
                return ""
            if val < 30:
                return "color: green"
            elif val < 50:
                return "color: orange"
            else:
                return "color: red"

        styled_df = df_summary.style.map(color_rse, subset=["RSE (%)"])

        print("Summary of Population Parameters")
        display(styled_df)

        return df_summary

    def get_state_dict(self) -> dict[str, Any]:

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
