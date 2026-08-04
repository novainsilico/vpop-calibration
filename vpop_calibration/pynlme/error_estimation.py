import torch
from torch.optim import LBFGS
from vpop_calibration.pynlme.indexing import IndexedObservations
from vpop_calibration.pynlme.residuals import (
    calculate_residuals,
    ResidualErrorEstimates,
)


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
    optimizer = LBFGS([log_variances], max_iter=max_iter, line_search_fn="strong_wolfe")

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
    min_variance: float,
    residual_error: ResidualErrorEstimates,
    max_iter: int = 20,
) -> ResidualErrorEstimates:

    residuals = calculate_residuals(observed_data=observations, predictions=predictions)
    output_index = observations.obs_index.output_name.index_values
    finite = torch.isfinite(predictions)

    # Outputs without usable observations simply keep their current estimate.
    new_sigma_add = residual_error.sigma_add.clone()
    new_sigma_prop = residual_error.sigma_prop.clone()

    for output, error_type in enumerate(residual_error.error_types):
        keep = (output_index == output).unsqueeze(0) & finite
        if error_type == "proportional":
            keep = keep & (predictions != 0)
        sq_residuals = residuals[keep].detach() ** 2
        sq_predictions = predictions[keep].detach() ** 2
        if sq_residuals.numel() == 0:
            continue

        if error_type == "additive":
            new_sigma_add[output] = sq_residuals.mean()
        elif error_type == "proportional":
            new_sigma_prop[output] = (sq_residuals / sq_predictions).mean()
        elif error_type == "combined":
            warm_start = torch.stack(
                (residual_error.sigma_add[output], residual_error.sigma_prop[output])
            )
            new_sigma_add[output], new_sigma_prop[output] = _solve_combined_output(
                sq_residuals=sq_residuals,
                sq_predictions=sq_predictions,
                max_iter=max_iter,
                warm_start=warm_start,
                min_variance=min_variance,
            )
        else:
            raise NotImplementedError(
                f"No variance estimator implemented for error_type={error_type!r}"
            )

    return residual_error._replace(sigma_add=new_sigma_add, sigma_prop=new_sigma_prop)
