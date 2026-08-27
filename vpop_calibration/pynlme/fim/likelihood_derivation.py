import torch
from vpop_calibration.config import device, default_dtype
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import log_likelihood_observation
from vpop_calibration.pynlme.fim.state import FimComponents


def complete_log_likelihood_function(
    model: StatisticalModel,
    theta_batch: torch.Tensor,
    etas: torch.Tensor,
) -> torch.Tensor:
    """log-likelihood, summed over patients, for each batch of population parameters."""
    assert theta_batch.dim() == 2
    log_lik_stack = []
    for theta_flat in theta_batch:
        params = model.unflatten(theta_flat)
        predictions = model.predict_from_pop_params(params=params, etas=etas)
        nb_pdu = params.omega_lower_chol.shape[0]

        log_lik_obs = log_likelihood_observation(
            observations=model.data.full_obs,
            predictions=predictions,
            residual_error=params.res_var,
            min_variance=model.config.residual_min_variance,
        )

        log_lik_etas = torch.distributions.MultivariateNormal(
            loc=torch.zeros(nb_pdu, device=device, dtype=default_dtype),
            scale_tril=params.omega_lower_chol,
        ).log_prob(etas)
        log_lik_complete = log_lik_obs + log_lik_etas
        log_lik_stack.append(log_lik_complete.sum())

    return torch.stack(log_lik_stack)


def compute_fim_components(
    model: StatisticalModel,
    theta_flat: torch.Tensor,
    etas: torch.Tensor,
    eps: float,
) -> FimComponents:

    assert etas.shape[0] == 1, (
        "Can only compute FIM components for one sample of the latent variables"
    )

    theta_detached = theta_flat.detach()
    h = eps * theta_flat.abs().clamp(min=1.0)
    nb_params = theta_flat.numel()

    # Total batch size: 1 base + 2n single perts + 2n(n-1) cross perts
    nb_cross_pairs = nb_params * (nb_params - 1) // 2
    total_batch_size = 1 + 2 * nb_params + 4 * nb_cross_pairs

    theta_batch = theta_detached.unsqueeze(0).repeat(total_batch_size, 1)

    # Track perturbation indices
    # Start with single perturbations
    plus_indices = torch.zeros(nb_params, dtype=torch.long, device=device)
    minus_indices = torch.zeros(nb_params, dtype=torch.long, device=device)

    cursor = 1
    for i in range(nb_params):
        # f(theta + h_i)
        theta_batch[cursor, i] += h[i]
        plus_indices[i] = cursor
        cursor += 1

        # f(theta - h_i)
        theta_batch[cursor, i] -= h[i]
        minus_indices[i] = cursor
        cursor += 1

    # Add cross-perturbations (for hessian)
    pair_offsets = {}
    for i in range(nb_params):
        for j in range(i + 1, nb_params):
            pair_offsets[(i, j)] = cursor
            # ++, +-, -+, --
            theta_batch[cursor + 0, i] += h[i]
            theta_batch[cursor + 0, j] += h[j]
            theta_batch[cursor + 1, i] += h[i]
            theta_batch[cursor + 1, j] -= h[j]
            theta_batch[cursor + 2, i] -= h[i]
            theta_batch[cursor + 2, j] += h[j]
            theta_batch[cursor + 3, i] -= h[i]
            theta_batch[cursor + 3, j] -= h[j]
            cursor += 4

    # Evaluate the log likelihood over all batched perturbations
    ll = complete_log_likelihood_function(
        model=model, theta_batch=theta_batch, etas=etas
    )

    # Compute the score (gradient) with central differences
    f_0 = ll[0]
    f_plus = ll[plus_indices]
    f_minus = ll[minus_indices]

    score = (f_plus - f_minus) / (2 * h)

    # Assemble the hessian matrix from finite differences

    hessian = torch.zeros((nb_params, nb_params), device=device, dtype=default_dtype)

    # Diagonal elements H_ii
    for i in range(nb_params):
        hessian[i, i] = (f_plus[i] - 2 * f_0 + f_minus[i]) / (h[i] ** 2)

    # Off-diagonal elements H_ij
    for i in range(nb_params):
        for j in range(i + 1, nb_params):
            idx = pair_offsets[(i, j)]
            f_pp, f_pm, f_mp, f_mm = (
                ll[idx],
                ll[idx + 1],
                ll[idx + 2],
                ll[idx + 3],
            )
            h_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * h[i] * h[j])
            hessian[i, j] = h_ij
            hessian[j, i] = h_ij

    return FimComponents(
        score=score,
        hessian=hessian,
        score_outer_product=torch.outer(score, score),
    )
