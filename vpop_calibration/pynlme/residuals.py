import torch

from typing import Any, NamedTuple

from vpop_calibration.pynlme.indexing import ObservationsDataSet
from vpop_calibration.pynlme.params import ErrorModel, ErrorType
from vpop_calibration.config import device, default_dtype


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
        empty_error_model = ErrorModel(
            error_type="survival", sigma=None, sigma_add=None, sigma_prop=None
        )
        priors = [
            error_model_priors.get(name, empty_error_model) for name in output_names
        ]
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

    def assert_initialized(self) -> None:
        if bool(((self.sigma_add < 0.0) | (self.sigma_prop < 0.0)).any()):
            raise RuntimeError(
                "The residual error model still holds its uninitialized value. `update_res_var` was never called"
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

    def __eq__(self, other) -> bool:
        compared_attributes = [
            "sigma_add",
            "sigma_prop",
            "additive_output",
            "proportional_output",
        ]

        for elem in compared_attributes:
            torch.testing.assert_close(
                getattr(self, elem), getattr(other, elem), equal_nan=True
            )
        return True


# @torch.compile
def calculate_residuals(
    observed_data: ObservationsDataSet,
    predictions: torch.Tensor,
) -> torch.Tensor:
    """Calculates residuals based on the error model for each continuous output

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
    observations: ObservationsDataSet,
    predictions: torch.Tensor,
    residual_error: ResidualErrorEstimates,
    min_variance: float,
) -> torch.Tensor:

    return residual_error.variance(
        predictions=predictions,
        output_index=observations.obs_index.output_name.index_values,
        min_variance=min_variance,
    )


def compute_survival_likelihood(
    observations: ObservationsDataSet, predictions: torch.Tensor
) -> torch.Tensor:
    nb_samples = predictions.shape[0]
    nb_patients = len(observations.obs_index.id.ref_values)

    if not observations.survival_outputs:
        # No survival data, contribution to LL is 0
        return torch.zeros(
            (nb_samples, nb_patients), device=device, dtype=default_dtype
        )

    log_hz_rows = torch.as_tensor(
        observations.obs_index.output_name.raw_values.values
        == observations.survival_outputs.log_hazard,
        dtype=torch.bool,
        device=device,
    )

    cumulative_hz_rows = torch.as_tensor(
        observations.obs_index.output_name.raw_values.values
        == observations.survival_outputs.cumulative_hazard,
        dtype=torch.bool,
        device=device,
    )
    log_hz_rows_expanded = log_hz_rows.expand(nb_samples, -1)
    cumulative_hz_rows_expanded = cumulative_hz_rows.expand(nb_samples, -1)
    # Gather the rows where the prediction contains the log_hazard
    log_hz_predicted = predictions[log_hz_rows_expanded].view(nb_samples, -1)

    cumulative_hz_predicted = predictions[cumulative_hz_rows_expanded].view(
        nb_samples, -1
    )

    event_status = observations.obs_values[log_hz_rows].expand(nb_samples, -1)

    # Log likelihood is computed as
    # (event is occurred) * log_hz (event_time) - cumulative_hazard(event_time)
    ll_surv = event_status * log_hz_predicted - cumulative_hz_predicted

    # This log-likelihood is only computed on patients who have a survival event observation

    patient_index_expanded = observations.obs_index.id.index_values[log_hz_rows].expand(
        nb_samples, -1
    )

    ll_per_patient = torch.zeros(
        (nb_samples, nb_patients), dtype=default_dtype, device=device
    )
    ll_per_patient.scatter_add_(1, patient_index_expanded, ll_surv)

    return ll_per_patient


def compute_normal_likelihood(
    observations: ObservationsDataSet,
    predictions: torch.Tensor,
    residual_error: ResidualErrorEstimates,
    min_variance: float,
) -> torch.Tensor:
    """Compute log-likelihood of predictions given corresponding observations.

    A nomal distribution is assumed for continuous model outputs.

    The output contains one total likelihood per patient, per sample.
    """

    nb_samples = predictions.shape[0]
    nb_patients = len(observations.obs_index.id.ref_values)

    continuous_outputs_indicator = torch.logical_or(
        residual_error.sigma_add, residual_error.sigma_prop
    )
    obs_output_indices = observations.obs_index.output_name.index_values
    continuous_outputs_mask = torch.index_select(
        continuous_outputs_indicator, 0, obs_output_indices
    ).expand(nb_samples, -1)
    # Compute residuals on all outputs (survival outputs are simply masked out)
    residuals = calculate_residuals(
        observed_data=observations,
        predictions=predictions,
    )

    # Log-likelihood of normal distribution (survival outputs are simply masked out)
    variance = compute_error_variance(
        observations=observations,
        predictions=predictions,
        residual_error=residual_error,
        min_variance=min_variance,
    )

    log_lik_full = torch.zeros_like(residuals, device=device, dtype=residuals.dtype)
    # Normal likelihood function for the continuous outputs
    log_lik_full[continuous_outputs_mask] = -0.5 * (
        torch.log(2 * torch.pi * variance[continuous_outputs_mask])
        + (residuals[continuous_outputs_mask] ** 2 / variance[continuous_outputs_mask])
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
def log_likelihood_observation(
    observations: ObservationsDataSet,
    predictions: torch.Tensor,
    residual_error: ResidualErrorEstimates,
    min_variance: float,
) -> torch.Tensor:
    """Compute the joint log-likelihood of observations, for all outputs.

    The output contains one likelihood per sample and per patient.
    """
    log_lik_continuous = compute_normal_likelihood(
        observations=observations,
        predictions=predictions,
        residual_error=residual_error,
        min_variance=min_variance,
    )

    log_lik_survival = compute_survival_likelihood(
        observations=observations, predictions=predictions
    )

    log_lik_final = log_lik_continuous + log_lik_survival
    return log_lik_final


# @torch.compile
def add_predictive_error(
    observations: ObservationsDataSet,
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
