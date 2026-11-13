import numpy as np
import pandas as pd
import torch

from vpop_calibration.config import device, default_dtype
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.params import Constraint
from vpop_calibration.pynlme.residuals import log_likelihood_observation

IDENTITY_LINK = (lambda x: x, lambda y: y)
LOG_LINK = (torch.log, torch.exp)
ATANH_LINK = (torch.atanh, torch.tanh)


def constraint_link(constraint: Constraint):
    shift, scale = constraint.shift, constraint.scale
    if constraint.transform == "log":
        return (lambda x: torch.log(x - shift), lambda y: shift + torch.exp(y))
    return (
        lambda x: torch.logit((x - shift) / scale),
        lambda y: shift + scale * torch.sigmoid(y),
    )


def build_cholesky_factor(
    cholesky_log_diagonal: torch.Tensor, cholesky_off_diagonal: torch.Tensor
) -> torch.Tensor:
    """Build the lower Cholesky factor L of Omega = L L^T.

    Diagonal parametrized by log L_kk, strictly lower part by L_kl.
    """
    nb_pdu = cholesky_log_diagonal.numel()
    rows, cols = torch.tril_indices(nb_pdu, nb_pdu, offset=-1, device=device)
    off_diagonal = torch.zeros(
        nb_pdu, nb_pdu, device=device, dtype=default_dtype
    ).index_put((rows, cols), cholesky_off_diagonal)
    return torch.diag(torch.exp(cholesky_log_diagonal)) + off_diagonal


class FimAccumulator:
    """Monte-Carlo estimate of the observed Fisher information matrix (Louis' formula).

    Accumulates per-patient scores g_i and hessians H_i of log p(y_i, phi_i; Theta~)
    over conditional samples, Theta~ being the unconstrained current estimates.
    """

    def __init__(self, model: StatisticalModel, finite_difference_step: float):
        self.model = model
        self.finite_difference_step = finite_difference_step
        cholesky = model.omega_pop_lower_chol
        rows, cols = torch.tril_indices(
            model.nb_pdu, model.nb_pdu, offset=-1, device=device
        )
        residual_var = model.residual_var
        self.internal_parameters = torch.cat(
            [
                model.population_betas,
                torch.log(torch.diagonal(cholesky)),
                cholesky[rows, cols],
                model.log_mi,
                model.surv_coeffs,
                torch.log(residual_var.additive_variance[residual_var.additive_output]),
                torch.log(
                    residual_var.proportional_variance[residual_var.proportional_output]
                ),
            ]
        ).to(device=device, dtype=default_dtype)
        self.nb_parameters = self.internal_parameters.numel()
        self.nb_pdu_parameters = model.nb_betas + model.nb_pdu * (model.nb_pdu + 1) // 2
        self.nb_fixed_effects = model.nb_mi + model.nb_surv_coeffs
        self.nb_samples = 0
        self.sum_hessian = torch.zeros(
            (self.nb_parameters, self.nb_parameters), device=device, dtype=default_dtype
        )
        self.sum_score_outer_product = torch.zeros_like(self.sum_hessian)
        self.sum_score_per_patient = torch.zeros(
            (model.nb_patients, self.nb_parameters), device=device, dtype=default_dtype
        )
        self.total_score_per_sample: list[torch.Tensor] = []

    def pdu_derivatives(
        self, gaussian_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and hessian of log p(phi_i; beta, Omega) w.r.t. (beta, log L_kk, L_kl).

        Closed-form gaussian density with phi_i fixed: exact derivatives by autograd.
        """
        pdu_parameters = self.internal_parameters[: self.nb_pdu_parameters]
        nb_chains, nb_patients, nb_pdu = gaussian_params.shape
        gaussian_rows = gaussian_params.reshape(nb_chains * nb_patients, nb_pdu)
        design_rows = self.model.full_design_matrix.expand(
            nb_chains, -1, -1, -1
        ).reshape(nb_chains * nb_patients, nb_pdu, self.model.nb_betas)

        nb_betas = self.model.nb_betas

        def log_likelihood_pdu(parameters, gaussian_row, design_row):
            beta = parameters[:nb_betas]
            cholesky_log_diagonal = parameters[nb_betas : nb_betas + nb_pdu]
            cholesky_off_diagonal = parameters[nb_betas + nb_pdu :]
            cholesky = build_cholesky_factor(
                cholesky_log_diagonal, cholesky_off_diagonal
            )
            etas = gaussian_row - design_row @ beta
            standardized_etas = torch.linalg.solve_triangular(
                cholesky, etas.unsqueeze(-1), upper=False
            ).squeeze(-1)
            return (
                -0.5 * nb_pdu * np.log(2 * np.pi)
                - cholesky_log_diagonal.sum()
                - 0.5 * (standardized_etas**2).sum(dim=-1)
            )

        score = torch.func.vmap(
            torch.func.grad(log_likelihood_pdu), in_dims=(None, 0, 0)
        )(pdu_parameters, gaussian_rows, design_rows)
        hessian = torch.func.vmap(
            torch.func.hessian(log_likelihood_pdu), in_dims=(None, 0, 0)
        )(pdu_parameters, gaussian_rows, design_rows)
        return (
            score.detach().reshape(nb_chains, nb_patients, self.nb_pdu_parameters),
            hessian.detach().reshape(
                nb_chains, nb_patients, self.nb_pdu_parameters, self.nb_pdu_parameters
            ),
        )

    def observation_derivatives(
        self, gaussian_params: torch.Tensor, predictions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score and hessian of log p(y_i | phi_i; psi, kappa, sigma^2).

        Central finite differences: psi and kappa act through the structural model,
        which may not be differentiable. Re-simulates only if psi or kappa is perturbed.
        """
        observation_parameters = self.internal_parameters[self.nb_pdu_parameters :]
        nb_obs_parameters = observation_parameters.numel()
        steps = self.finite_difference_step * observation_parameters.abs().clamp(
            min=1.0
        )
        perturbations = torch.diag(steps)

        model = self.model
        residual_var = model.residual_var
        nb_additive = int(residual_var.additive_output.sum().item())

        def log_likelihood_perturbated_observations(
            perturbation: torch.Tensor,
        ) -> torch.Tensor:
            parameters = observation_parameters + perturbation
            current_predictions = predictions
            if (perturbation[: self.nb_fixed_effects] != 0).any():
                physical_params = model.convert_gaussian_to_physical(
                    psi=gaussian_params,
                    log_mi=parameters[: model.nb_mi],
                    surv_coeffs=parameters[model.nb_mi : self.nb_fixed_effects],
                )
                thetas = model.convert_physical_to_thetas_all_patients(physical_params)
                model_inputs = model.convert_thetas_to_model_parameters_all_patients(
                    thetas
                )
                current_predictions, _ = model.predict_all_patients(model_inputs)
            variances = torch.exp(parameters[self.nb_fixed_effects :])
            additive_variance = residual_var.additive_variance.clone()
            additive_variance[residual_var.additive_output] = variances[:nb_additive]
            proportional_variance = residual_var.proportional_variance.clone()
            proportional_variance[residual_var.proportional_output] = variances[
                nb_additive:
            ]
            return log_likelihood_observation(
                model.data.full_obs,
                current_predictions,
                residual_var._replace(
                    additive_variance=additive_variance,
                    proportional_variance=proportional_variance,
                ),
                model.config.residual_min_variance,
            )

        log_lik_base = log_likelihood_perturbated_observations(
            torch.zeros_like(observation_parameters)
        )
        log_lik_plus = [
            log_likelihood_perturbated_observations(perturbations[q])
            for q in range(nb_obs_parameters)
        ]
        log_lik_minus = [
            log_likelihood_perturbated_observations(-perturbations[q])
            for q in range(nb_obs_parameters)
        ]

        nb_chains, nb_patients = log_lik_base.shape
        score = torch.zeros(
            (nb_chains, nb_patients, nb_obs_parameters),
            device=device,
            dtype=default_dtype,
        )
        hessian = torch.zeros(
            (nb_chains, nb_patients, nb_obs_parameters, nb_obs_parameters),
            device=device,
            dtype=default_dtype,
        )
        for q in range(nb_obs_parameters):
            score[:, :, q] = (log_lik_plus[q] - log_lik_minus[q]) / (2 * steps[q])
            hessian[:, :, q, q] = (
                log_lik_plus[q] - 2 * log_lik_base + log_lik_minus[q]
            ) / steps[q] ** 2
            for r in range(q + 1, nb_obs_parameters):
                log_lik_cross = (
                    log_likelihood_perturbated_observations(
                        perturbations[q] + perturbations[r]
                    )
                    - log_likelihood_perturbated_observations(
                        perturbations[q] - perturbations[r]
                    )
                    - log_likelihood_perturbated_observations(
                        -perturbations[q] + perturbations[r]
                    )
                    + log_likelihood_perturbated_observations(
                        -perturbations[q] - perturbations[r]
                    )
                )
                hessian[:, :, q, r] = log_lik_cross / (4 * steps[q] * steps[r])
                hessian[:, :, r, q] = hessian[:, :, q, r]
        return score, hessian

    def accumulate(self, eta_samples: torch.Tensor, predictions: torch.Tensor) -> None:
        """Add a batch of conditional samples to the sums of H_i, g_i g_i^T and g_i.

        H_i is block diagonal: with phi_i fixed, the two likelihood terms share no
        parameter. Also stores sum_i g_i per sample, for the convergence score history.
        """
        gaussian_params = self.model.convert_etas_to_gaussian_all_patients(eta_samples)
        pdu_score, pdu_hessian = self.pdu_derivatives(gaussian_params)
        observation_score, observation_hessian = self.observation_derivatives(
            gaussian_params, predictions
        )
        score = torch.cat([pdu_score, observation_score], dim=-1)
        hessian = torch.zeros(
            score.shape + (self.nb_parameters,), device=device, dtype=default_dtype
        )
        hessian[:, :, : self.nb_pdu_parameters, : self.nb_pdu_parameters] = pdu_hessian
        hessian[:, :, self.nb_pdu_parameters :, self.nb_pdu_parameters :] = (
            observation_hessian
        )

        self.nb_samples += score.shape[0]
        self.sum_hessian += hessian.sum(dim=(0, 1))
        self.sum_score_outer_product += torch.einsum("cpi,cpj->ij", score, score)
        self.sum_score_per_patient += score.sum(dim=0)
        self.total_score_per_sample.append(score.sum(dim=1))

    def observed_fisher_information(self) -> torch.Tensor:
        """Louis' formula: I = -sum_i (E[H_i] + E[g_i g_i^T] - E[g_i] E[g_i]^T).

        Expectations are estimated over the accumulated samples.
        """
        mean_score_per_patient = self.sum_score_per_patient / self.nb_samples
        fim = -(
            self.sum_hessian / self.nb_samples
            + self.sum_score_outer_product / self.nb_samples
            - mean_score_per_patient.T @ mean_score_per_patient
        )
        return 0.5 * (fim + fim.T)


def normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Normalized symmetric matrix M_qr / sqrt(M_qq M_rr), with the same labels.

    Applied to the FIM it gives r = I_qr / sqrt(I_qq I_rr); applied to the covariance of
    the estimates it gives their correlation R. Non-positive diagonal entries give NaN.
    """
    diagonal = np.diag(matrix.to_numpy())
    scale = np.sqrt(np.where(diagonal > 0, diagonal, np.nan))
    return matrix / np.outer(scale, scale)


def get_displayed_parameters(model: StatisticalModel) -> list[tuple]:
    """Describe each reported parameter: (name, link (G, G^-1), map from Theta~).

    The link builds confidence intervals; the element-wise map is None for omega and
    correlations, which are computed jointly from L.
    """
    rows, cols = torch.tril_indices(
        model.nb_pdu, model.nb_pdu, offset=-1, device=device
    )
    residual_var = model.residual_var

    def from_link(link):
        return (link, link[1])

    betas = [
        (name,)
        + from_link(
            constraint_link(model.input_params.pdu[name].constraint)
            if name in model.pdu_names
            else IDENTITY_LINK
        )
        for name in model.beta_names
    ]
    omega_sd = [(f"omega_{pdu}", LOG_LINK, None) for pdu in model.pdu_names]
    omega_corr = [
        (f"corr_{model.pdu_names[row]}_{model.pdu_names[col]}", ATANH_LINK, None)
        for row, col in zip(rows.tolist(), cols.tolist())
    ]
    mis = [
        (name,)
        + from_link(
            constraint_link(model.input_params.model_intrinsic[name].constraint)
        )
        for name in model.mi_names
    ]
    surv_coeffs = [
        (name,) + from_link(IDENTITY_LINK) for name in model.surv_coeff_names
    ]
    residual_sds = [
        (f"sd_{component}_{output}", LOG_LINK, lambda x: torch.exp(0.5 * x))
        for component, mask in [
            ("add", residual_var.additive_output),
            ("prop", residual_var.proportional_output),
        ]
        for output, active in zip(model.output_names, mask.tolist())
        if active
    ]
    return betas + omega_sd + omega_corr + mis + surv_coeffs + residual_sds


def compute_estimates_uncertainty(
    accumulator: FimAccumulator, model: StatisticalModel
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Covariance of the estimates on the reported scale, C = J I^-1 J^T (delta method).

    Also returns the FIM on the reported scale, J^-T I J^-1, a table (estimates, SE, RSE,
    95% CI, convergence score) and the convergence score history (Newton step in SE units).
    """
    displayed_parameters = get_displayed_parameters(model)
    internal_parameters = accumulator.internal_parameters
    fim = accumulator.observed_fisher_information()
    fim_cholesky, info = torch.linalg.cholesky_ex(fim)
    if info.item() == 0:
        internal_covariance = torch.cholesky_inverse(fim_cholesky)
    else:
        internal_covariance = torch.linalg.pinv(fim)
    nb_pdu_parameters = accumulator.nb_pdu_parameters
    rows, cols = torch.tril_indices(
        model.nb_pdu, model.nb_pdu, offset=-1, device=device
    )

    def display_transformation(parameters: torch.Tensor) -> torch.Tensor:
        cholesky = build_cholesky_factor(
            cholesky_log_diagonal=parameters[
                model.nb_betas : model.nb_betas + model.nb_pdu
            ],
            cholesky_off_diagonal=parameters[
                model.nb_betas + model.nb_pdu : nb_pdu_parameters
            ],
        )
        omega = cholesky @ cholesky.T
        omega_sd = torch.sqrt(torch.diagonal(omega))
        omega_corr = omega[rows, cols] / (omega_sd[rows] * omega_sd[cols])
        elementwise = torch.stack(
            [
                value if element_transform is None else element_transform(value)
                for (_, _, element_transform), value in zip(
                    displayed_parameters, parameters
                )
            ]
        )
        return torch.cat(
            [
                elementwise[: model.nb_betas],
                omega_sd,
                omega_corr,
                elementwise[nb_pdu_parameters:],
            ]
        )

    jacobian = torch.func.jacrev(display_transformation)(internal_parameters)
    estimates = display_transformation(internal_parameters).detach()
    covariance = jacobian @ internal_covariance @ jacobian.T
    standard_errors = torch.sqrt(torch.diagonal(covariance))
    total_score_per_sample = torch.cat(accumulator.total_score_per_sample)
    nb_samples_used = torch.arange(
        1, total_score_per_sample.shape[0] + 1, device=device, dtype=default_dtype
    )
    running_mean_score = total_score_per_sample.cumsum(dim=0) / nb_samples_used[:, None]
    convergence_score_history = (
        running_mean_score @ (jacobian @ internal_covariance).T / standard_errors
    )

    ci_low, ci_high = [], []
    for (_, (link, inverse_link), _), estimate, se in zip(
        displayed_parameters, estimates, standard_errors
    ):
        center = link(estimate)
        half_width = 1.96 * se * torch.func.grad(link)(estimate).abs()
        ci_low.append(inverse_link(center - half_width).item())
        ci_high.append(inverse_link(center + half_width).item())

    names = [name for name, _, _ in displayed_parameters]
    covariance_df = pd.DataFrame(
        covariance.detach().cpu().numpy(), index=names, columns=names
    )
    standard_errors_df = pd.DataFrame(
        {
            "estimate": estimates.cpu().numpy(),
            "se": standard_errors.detach().cpu().numpy(),
            "rse": (100 * standard_errors / estimates.abs()).detach().cpu().numpy(),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "convergence_score": convergence_score_history[-1].detach().cpu().numpy(),
        },
        index=names,
    )
    convergence_score_history_df = pd.DataFrame(
        convergence_score_history.detach().cpu().numpy(),
        index=pd.Index(nb_samples_used.long().cpu().numpy(), name="nb_samples"),
        columns=names,
    )
    inverse_jacobian = torch.linalg.pinv(jacobian)
    reported_fim = inverse_jacobian.T @ fim @ inverse_jacobian
    fisher_information_df = pd.DataFrame(
        reported_fim.detach().cpu().numpy(), index=names, columns=names
    )
    return (
        fisher_information_df,
        covariance_df,
        standard_errors_df,
        convergence_score_history_df,
    )
