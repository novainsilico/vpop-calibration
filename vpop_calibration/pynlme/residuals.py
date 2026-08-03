import torch

from typing import Any, NamedTuple

from vpop_calibration.pynlme.indexing import IndexedObservations
from vpop_calibration.pynlme.params import ErrorModel, ErrorType
from vpop_calibration.config import device


class ResidualErrorEstimates(NamedTuple):
    sigma_add: torch.Tensor
    sigma_prop: torch.Tensor
    additive_output: torch.Tensor  # bool, size (nb_outputs,)
    proportional_output: torch.Tensor  # bool, size (nb_outputs,)

    @classmethod
    def from_priors(
        cls,
        error_model_priors: dict[str, ErrorModel],
        output_names: list[str],
    ) -> "ResidualErrorEstimates":
        """Build the initial estimates from the user-specified priors."""
        priors = [error_model_priors[name] for name in output_names]
        variances = [prior.variance_components for prior in priors]
        active = [prior.active_components for prior in priors]
        return cls(
            sigma_add=torch.as_tensor([var[0] for var in variances], device=device),
            sigma_prop=torch.as_tensor([var[1] for var in variances], device=device),
            additive_output=torch.as_tensor([act[0] for act in active], device=device),
            proportional_output=torch.as_tensor(
                [act[1] for act in active], device=device
            ),
        )

    @property
    def nb_outputs(self) -> int:
        return self.sigma_add.shape[0]

    @property
    def error_types(self) -> list[ErrorType]:
        """Error type of each output, recovered from the active components."""
        return [
            "combined"
            if additive and proportional
            else "additive"
            if additive
            else "proportional"
            for additive, proportional in zip(
                self.additive_output.tolist(), self.proportional_output.tolist()
            )
        ]

    def sanitized(self) -> "ResidualErrorEstimates":
        """Force inactive components to zero and active ones to stay non-negative."""
        return self._replace(
            sigma_add=torch.where(
                self.additive_output,
                self.sigma_add.clamp_min(0.0),
                torch.zeros_like(self.sigma_add),
            ),
            sigma_prop=torch.where(
                self.proportional_output,
                self.sigma_prop.clamp_min(0.0),
                torch.zeros_like(self.sigma_prop),
            ),
        )

    def variance(
        self,
        predictions: torch.Tensor,
        output_index: torch.Tensor,
        min_variance: float,
    ) -> torch.Tensor:
        """Residual variance of each prediction."""
        nb_samples = predictions.shape[0]
        sigma_add = self.sigma_add.index_select(0, output_index).expand(nb_samples, -1)
        sigma_prop = self.sigma_prop.index_select(0, output_index).expand(
            nb_samples, -1
        )
        sq_predictions = torch.where(
            torch.isfinite(predictions), predictions**2, torch.ones_like(predictions)
        )
        return (sigma_add + sigma_prop * sq_predictions).clamp_min(min_variance)

    def get_state_dict(self) -> dict[str, Any]:
        return {key: val.detach().cpu().tolist() for key, val in self._asdict().items()}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "ResidualErrorEstimates":
        dtypes = {"additive_output": torch.bool, "proportional_output": torch.bool}
        return cls(
            **{
                key: torch.as_tensor(val, device=device, dtype=dtypes.get(key))
                for key, val in state_dict.items()
            }
        )


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
    residual_error: ResidualErrorEstimates,
    min_variance: float,
) -> torch.Tensor:

    return residual_error.variance(
        predictions=predictions,
        output_index=observations.obs_index.output_name.index_values,
        min_variance=min_variance,
    )


# @torch.compile
def log_likelihood_observation(
    observations: IndexedObservations,
    predictions: torch.Tensor,
    residual_error: ResidualErrorEstimates,
    min_variance: float,
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
        residual_error=residual_error,
        min_variance=min_variance,
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
    residual_error: ResidualErrorEstimates,
    min_variance: float,
) -> torch.Tensor:
    out_variance = compute_error_variance(
        observations=observations,
        predictions=predictions,
        residual_error=residual_error,
        min_variance=min_variance,
    )
    noisy_predictions = torch.distributions.Normal(
        predictions, torch.sqrt(out_variance)
    ).sample()

    return noisy_predictions
