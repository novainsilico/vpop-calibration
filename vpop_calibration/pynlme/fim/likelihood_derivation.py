import torch
from typing import Literal, overload
from vpop_calibration.config import device
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import log_likelihood_observation
from vpop_calibration.pynlme.fim.parametrization import unflatten
from vpop_calibration.pynlme.fim.state import FimComponents


def predict_detached(
    model: StatisticalModel, log_mi: torch.Tensor, gaussian_params: torch.Tensor
) -> torch.Tensor:
    """Model predictions for the given parameters, detached from the autograd graph."""
    preds_list = []

    for c in range(gaussian_params.shape[0]):
        physical = model.convert_gaussian_to_physical(gaussian_params[c], log_mi, model.surv_coeffs)
        thetas = model.convert_physical_to_thetas_all_patients(physical)
        inputs = model.convert_thetas_to_model_parameters_all_patients(thetas)
        predictions, _ = model.predict_all_patients(inputs)
        preds_list.append(predictions)

    return torch.stack(preds_list, dim=0).detach()


def complete_log_likelihood(
    model: StatisticalModel,
    flat: torch.Tensor,
    predictions: torch.Tensor,
    gaussian_params: torch.Tensor,
) -> torch.Tensor:
    """log-likelihood, summed over patients, per MCMC chain."""
    params = unflatten(flat, model)
    nb_chains = gaussian_params.shape[0]

    log_lik_chains = []
    mu = model.full_design_matrix @ params.beta

    for c in range(nb_chains):
        log_lik_obs = log_likelihood_observation(
            observations=model.data.full_obs,
            predictions=predictions[c],
            residual_error=params.residual_var,
            min_variance=model.config.residual_min_variance,
        )
        log_lik_psi = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=params.omega
        ).log_prob(gaussian_params[c])
        log_lik_chains.append((log_lik_obs + log_lik_psi).sum())
    return torch.stack(log_lik_chains)


def analytic_score_and_hessian(
    model: StatisticalModel,
    flat: torch.Tensor,
    predictions: torch.Tensor,
    gaussian_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chain scores and mean hessian obtained by autograd."""
    theta = flat.clone().requires_grad_(True)
    scores = torch.stack(
        [
            torch.autograd.grad(
                complete_log_likelihood(
                    model,
                    theta,
                    predictions[c : c + 1],
                    gaussian_params[c : c + 1],
                ).sum(),
                theta,
            )[0]
            for c in range(gaussian_params.shape[0])
        ]
    )
    hessian = torch.autograd.functional.hessian(
        lambda t: complete_log_likelihood(
            model, t, predictions, gaussian_params
        ).mean(),
        flat,
    )
    assert isinstance(hessian, torch.Tensor)
    return scores, hessian


def model_intrinsic_finite_differences(
    model: StatisticalModel,
    flat: torch.Tensor,
    gaussian_params: torch.Tensor,
    base_predictions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = flat.clone().requires_grad_(True)
    log_mi0 = flat[model.mi_location]
    h = eps * torch.clamp(log_mi0.abs(), min=1.0)
    e = torch.eye(model.nb_mi, device=device, dtype=flat.dtype) * h

    @overload
    def perturb_log_likelihood(
        step: torch.Tensor, compute_grad: Literal[False] = False
    ) -> torch.Tensor: ...
    @overload
    def perturb_log_likelihood(
        step: torch.Tensor, compute_grad: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def perturb_log_likelihood(step: torch.Tensor, compute_grad: bool = False):
        preds = predict_detached(model, log_mi0 + step, gaussian_params)
        ll_step = complete_log_likelihood(model, flat, preds, gaussian_params)

        if not compute_grad:
            return ll_step

        grad_step = torch.autograd.grad(
            complete_log_likelihood(model, theta, preds, gaussian_params).mean(),
            theta,
        )[0]
        return ll_step, grad_step

    ll_0 = complete_log_likelihood(model, flat, base_predictions, gaussian_params)
    plus = [perturb_log_likelihood(e[k], compute_grad=True) for k in range(model.nb_mi)]
    minus = [
        perturb_log_likelihood(-e[k], compute_grad=True) for k in range(model.nb_mi)
    ]
    ll_p = torch.stack([p[0] for p in plus])
    ll_m = torch.stack([m[0] for m in minus])

    mi_scores = (ll_p - ll_m) / (2 * h.unsqueeze(1))
    cross = torch.stack(
        [(plus[k][1] - minus[k][1]) / (2 * h[k]) for k in range(model.nb_mi)],
        dim=1,
    )

    mi_hessian = torch.diag((ll_p - 2 * ll_0 + ll_m).mean(dim=1) / h**2)
    for i, j in zip(*torch.triu_indices(model.nb_mi, model.nb_mi, offset=1)):
        mi_hessian[i, j] = mi_hessian[j, i] = (
            perturb_log_likelihood(e[i] + e[j])
            - perturb_log_likelihood(e[i] - e[j])
            - perturb_log_likelihood(-e[i] + e[j])
            + perturb_log_likelihood(-e[i] - e[j])
        ).mean() / (4 * h[i] * h[j])

    return mi_scores, mi_hessian, cross


def compute_fim_components(
    model: StatisticalModel,
    flat: torch.Tensor,
    gaussian_params: torch.Tensor,
    eps: float = 1e-3,
) -> FimComponents:
    """autograd on (beta, omega, sigma),finite differences on the model-intrinsic parameters."""
    nb_chains = gaussian_params.shape[0]
    predictions = predict_detached(model, flat[model.mi_location], gaussian_params)
    scores, hessian = analytic_score_and_hessian(
        model, flat, predictions, gaussian_params
    )

    if model.nb_mi > 0:
        mi = model.mi_location
        mi_scores, mi_hessian, cross = model_intrinsic_finite_differences(
            model, flat, gaussian_params, predictions, eps
        )
        scores, hessian = scores.clone(), hessian.clone()
        scores[:, mi] = mi_scores.transpose(0, 1)
        hessian[:, mi] = cross
        hessian[mi, :] = cross.transpose(0, 1)
        hessian[mi, mi] = mi_hessian

    return FimComponents(
        score=scores.mean(dim=0),
        hessian=hessian,
        score_outer_product=scores.transpose(0, 1) @ scores / nb_chains,
    )
