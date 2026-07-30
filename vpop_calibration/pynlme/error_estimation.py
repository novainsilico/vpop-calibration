import torch
from torch.optim import LBFGS
from vpop_calibration.pynlme.indexing import IndexedObservations
from vpop_calibration.pynlme.params import ErrorType
from vpop_calibration.pynlme.residuals import (
    RESIDUAL_MIN_VARIANCE,
    calculate_residuals,
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
        assert sq_residuals.numel() > 0, (
            f"Output {output} ({error_type}) has no usable observation"
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
