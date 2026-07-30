import torch

from vpop_calibration.pynlme.indexing import IndexedObservations
from vpop_calibration.pynlme.params import ErrorType
from vpop_calibration.config import device

RESIDUAL_MIN_VARIANCE = 1e-6


# @torch.compile
def calculate_residuals(
    observed_data: IndexedObservations,
    predictions: torch.Tensor,
) -> torch.Tensor:
    """Calculates residuals based on the error model for each output

    Args:
        observed_data: Indexed observations
        predictions: Tensor of batched predictions

    Returns:
        torch.Tensor: a tensor of residual values
    """
    assert predictions.dim() == 2, (
        "Incorrect amount of dimensions in predictions tensor"
    )
    batch_size = predictions.shape[0]
    obs_vals = observed_data.obs_values.expand(batch_size, -1)
    assert predictions.shape == obs_vals.shape, (
        f"Non-matching shapes in `calculate_residuals`: {predictions.shape=}, {obs_vals.shape=}"
    )

    residuals = obs_vals - predictions
    nan_or_inf_mask = torch.logical_not(torch.isfinite(predictions))
    residuals[nan_or_inf_mask] = -torch.inf
    return residuals


# @torch.compile
def compute_error_variance(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:

    nb_samples = predictions.shape[0]
    output_index = observations.obs_index.output_name.index_values
    sigma_add = sigma[:, 0].index_select(0, output_index).expand(nb_samples, -1)
    sigma_prop = sigma[:, 1].index_select(0, output_index).expand(nb_samples, -1)

    nan_or_inf_mask = torch.logical_not(torch.isfinite(predictions))
    sq_predictions = torch.where(
        nan_or_inf_mask, torch.ones_like(predictions), predictions**2
    )

    return (sigma_add + sigma_prop * sq_predictions).clamp_min(RESIDUAL_MIN_VARIANCE)


# @torch.compile
def log_likelihood_observation(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute log-likelihood of predictions given corresponding observations.

    The output contains one total likelihood per patient, per sample.
    """
    nb_samples = predictions.shape[0]
    nb_patients = len(observations.obs_index.id.ref_values)

    residuals = calculate_residuals(
        observed_data=observations,
        predictions=predictions,
    )

    # Log-likelihood of normal distribution
    variance = compute_error_variance(
        observations=observations,
        predictions=predictions,
        sigma=sigma,
    )
    # Normal likelihood function
    log_lik_full = -0.5 * (
        torch.log(2 * torch.pi * variance) + (residuals**2 / variance)
    )
    log_lik_per_patient = torch.zeros(
        (nb_samples, nb_patients), device=device, dtype=predictions.dtype
    )
    patient_index_expanded = observations.obs_index.id.index_values.expand(
        nb_samples, -1
    )
    log_lik_per_patient.scatter_add_(
        1,
        patient_index_expanded,
        log_lik_full,
    )
    return log_lik_per_patient


# @torch.compile
def add_predictive_error(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    out_variance = compute_error_variance(
        observations=observations,
        predictions=predictions,
        sigma=sigma,
    )
    noisy_predictions = torch.distributions.Normal(
        predictions, torch.sqrt(out_variance)
    ).sample()

    return noisy_predictions


def _solve_combined_output(
    sq_residuals: torch.Tensor,
    sq_predictions: torch.Tensor,
    max_iter: int,
    warm_start: torch.Tensor,
    min_variance: float,
) -> torch.Tensor:
    log_variances = (
        warm_start.clamp_min(min_variance).log().detach().clone().requires_grad_(True)
    )
    optimizer = torch.optim.LBFGS(
        [log_variances], max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        a2, b2 = log_variances.exp()
        variance = (a2 + b2 * sq_predictions).clamp_min(min_variance)
        loss = 0.5 * (variance.log() + sq_residuals / variance).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_variances.detach().exp()


# @torch.compile
def estimate_error_params(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    error_model_selector: dict[ErrorType, list[int]],
    sigma: torch.Tensor,
    max_iter: int = 20,
    min_variance: float = RESIDUAL_MIN_VARIANCE,
) -> torch.Tensor:

    nb_outputs = sigma.shape[0]
    error_type_per_output: dict[int, ErrorType] = {
        output: error_type
        for error_type, outputs in error_model_selector.items()
        for output in outputs
    }
    assert sorted(error_type_per_output) == list(range(nb_outputs)), (
        f"`error_model_selector` must assign exactly one error type to each of "
        f"the {nb_outputs} outputs, got {error_model_selector}"
    )

    residuals = calculate_residuals(observed_data=observations, predictions=predictions)
    output_index = observations.obs_index.output_name.index_values
    finite = torch.isfinite(predictions)
    estimates = torch.zeros_like(sigma)

    for output in range(nb_outputs):
        error_type = error_type_per_output[output]

        keep = (output_index == output).unsqueeze(0) & finite
        if error_type == "proportional":
            keep = keep & (predictions != 0)
        sq_residuals = residuals[keep].detach() ** 2
        sq_predictions = predictions[keep].detach() ** 2
        assert sq_residuals.numel() >= 2, (
            f"Output {output} ({error_type}) has too few usable observations "
            f"to estimate its residual variance"
        )

        if error_type == "additive":
            estimates[output, 0] = sq_residuals.mean()
        elif error_type == "proportional":
            estimates[output, 1] = (sq_residuals / sq_predictions).mean()
        elif error_type == "combined":
            estimates[output] = _solve_combined_output(
                sq_residuals=sq_residuals,
                sq_predictions=sq_predictions,
                max_iter=max_iter,
                warm_start=sigma[output],
                min_variance=min_variance,
            )
        else:
            raise NotImplementedError(
                f"No variance estimator implemented for error_type={error_type!r}"
            )

    return estimates
