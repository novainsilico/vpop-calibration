import torch
from typing import get_args

from vpop_calibration.pynlme.indexing import IndexedObservations
from vpop_calibration.pynlme.params import ErrorType
from vpop_calibration.config import device


# @torch.compile
def calculate_residuals(
    observed_data: IndexedObservations,
    predictions: torch.Tensor,
    error_model_selector: dict[ErrorType, list[int]],
) -> torch.Tensor:
    """Calculates residuals based on the error model for each output

    Args:
        observed_data: Indexed observations
        predictions: Tensor of batched predictions

    Returns:
        torch.Tensor: a tensor of residual values
    """
    assert (
        predictions.dim() == 2
    ), "Incorrect amount of dimensions in predictions tensor"
    batch_size = predictions.shape[0]
    obs_vals = observed_data.obs_values.expand(batch_size, -1)
    output_indices = observed_data.obs_index.output_name.index_values.expand(
        batch_size, -1
    )
    assert (
        predictions.shape == obs_vals.shape
    ), f"Non-matching shapes in `calculate_residuals`: {predictions.shape=}, {obs_vals.shape=}"

    residuals = torch.zeros_like(predictions, device=device)
    for error_type in get_args(ErrorType):
        mask = torch.as_tensor(
            sum(output_indices == i for i in error_model_selector[error_type]),
            dtype=torch.bool,
            device=device,
        )
        if error_type == "additive":
            residuals[mask] = obs_vals[mask] - predictions[mask]
        elif error_type == "proportional":
            residuals[mask] = (obs_vals[mask] - predictions[mask]) / predictions[mask]
        else:
            raise NotImplemented(f"Unknown error model type {error_type}")
    return residuals


# @torch.compile
def sum_sq_residuals(
    observations: IndexedObservations,
    prediction: torch.Tensor,
    error_model_selector: dict[ErrorType, list[int]],
) -> torch.Tensor:
    """Compute the sum of squared residuals for a given prediction tensor"""

    nb_samples = prediction.shape[0]
    nb_outputs = len(observations.obs_index.output_name.ref_values)
    sq_residuals = torch.square(
        calculate_residuals(
            observed_data=observations,
            predictions=prediction,
            error_model_selector=error_model_selector,
        )
    )
    sum_residuals_per_sample = torch.zeros(
        nb_samples, nb_outputs, device=device, dtype=sq_residuals.dtype
    )
    output_index = observations.obs_index.output_name.index_values
    output_index_expanded = output_index.expand(nb_samples, -1)
    sum_residuals_per_sample.scatter_add_(1, output_index_expanded, sq_residuals)
    return sum_residuals_per_sample


# @torch.compile
def compute_error_variance(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    error_model_selector: dict[ErrorType, list[int]],
    sigma: torch.Tensor,
) -> torch.Tensor:

    nb_samples = predictions.shape[0]
    output_index = observations.obs_index.output_name.index_values
    output_index_expanded = output_index.expand(nb_samples, -1)
    sigma_expanded = sigma.expand(nb_samples, -1).index_select(1, output_index)

    out_variance = torch.zeros_like(predictions, device=device)
    for error_type in get_args(ErrorType):
        mask = torch.as_tensor(
            sum(output_index_expanded == i for i in error_model_selector[error_type]),
            dtype=torch.bool,
            device=device,
        )
        if error_type == "additive":
            out_variance[mask] = sigma_expanded[mask]
        elif error_type == "proportional":
            out_variance[mask] = sigma_expanded[mask] * torch.square(predictions[mask])
        else:
            raise NotImplemented(f"Unknown error model type {error_type}")

    return out_variance


# @torch.compile
def log_likelihood_observation(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    error_model_selector: dict[ErrorType, list[int]],
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
        error_model_selector=error_model_selector,
    )

    # Log-likelihood of normal distribution
    variance = compute_error_variance(
        observations=observations,
        predictions=predictions,
        error_model_selector=error_model_selector,
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
    error_model_selector: dict[ErrorType, list[int]],
    sigma: torch.Tensor,
) -> torch.Tensor:
    out_variance = compute_error_variance(
        observations=observations,
        predictions=predictions,
        error_model_selector=error_model_selector,
        sigma=sigma,
    )
    noisy_predictions = torch.distributions.Normal(
        predictions, torch.sqrt(out_variance)
    ).sample()

    return noisy_predictions
