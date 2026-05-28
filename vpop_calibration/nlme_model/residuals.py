import torch
from typing import get_args

from vpop_calibration.nlme_model.indexing import IndexedObservations
from vpop_calibration.nlme_model.params import ErrorType


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

    residuals = torch.zeros_like(predictions)
    for error_type in get_args(ErrorType):
        mask = torch.as_tensor(
            sum(output_indices == i for i in error_model_selector[error_type]),
            dtype=torch.bool,
        )
        if error_type == "additive":
            residuals[mask] = obs_vals[mask] - predictions[mask]
        elif error_type == "proportional":
            residuals[mask] = (obs_vals[mask] - predictions[mask]) / predictions[mask]
        else:
            raise NotImplemented(f"Unknown error model type {error_type}")
    return residuals


def compute_error_variance(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    error_model_selector: dict[ErrorType, list[int]],
    sigma: torch.Tensor,
) -> torch.Tensor:

    nb_samples = predictions.shape[0]
    output_index = observations.obs_index.output_name.index_values
    sigma_expanded = sigma.expand(nb_samples, -1).index_select(1, output_index)
    output_index_expanded = output_index.expand(nb_samples, -1)

    out_variance = torch.zeros_like(predictions)
    for error_type in get_args(ErrorType):
        mask = torch.as_tensor(
            sum(output_index_expanded == i for i in error_model_selector[error_type]),
            dtype=torch.bool,
        )
        if error_type == "additive":
            out_variance[mask] = sigma_expanded[mask]
        elif error_type == "proportional":
            out_variance[mask] = sigma_expanded[mask] * torch.square(predictions[mask])
        else:
            raise NotImplemented(f"Unknown error model type {error_type}")

    return out_variance
