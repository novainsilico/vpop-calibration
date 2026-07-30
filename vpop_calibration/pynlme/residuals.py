import torch

from vpop_calibration.pynlme.indexing import IndexedObservations
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
