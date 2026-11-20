import numpy as np
import pandas as pd
import torch

from vpop_calibration.config import device, default_dtype
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.params import transform_param
from vpop_calibration.pynlme.residuals import log_likelihood_observation
from vpop_calibration.pynlme.utils import inverse_transform_param


def to_natural(model: StatisticalModel, name: str, value: float) -> float:
    if name in model.pdu_names:
        return float(
            inverse_transform_param(value, model.input_params.pdu[name].constraint)
        )
    if name in model.mi_names:
        return float(
            inverse_transform_param(
                value, model.input_params.model_intrinsic[name].constraint
            )
        )
    return value


def to_internal(model: StatisticalModel, name: str, value: float) -> float:
    if name in model.pdu_names:
        return float(transform_param(value, model.input_params.pdu[name].constraint))
    if name in model.mi_names:
        return float(
            transform_param(value, model.input_params.model_intrinsic[name].constraint)
        )
    return value


def natural_bounds(model: StatisticalModel, name: str) -> tuple[float, float]:
    """Open interval of admissible values of a displayed parameter."""
    if name in model.pdu_names:
        constraint = model.input_params.pdu[name].constraint
    elif name in model.mi_names:
        constraint = model.input_params.model_intrinsic[name].constraint
    elif name in [f"omega_{pdu}" for pdu in model.pdu_names] or name in [
        residual_name for residual_name, _, _ in residual_components(model)
    ]:
        return 0.0, np.inf
    elif name in [
        correlation_name(model, row, col)
        for row in range(model.nb_pdu)
        for col in range(row)
    ]:
        return -1.0, 1.0
    else:
        return -np.inf, np.inf
    if constraint.transform == "logit":
        return constraint.shift, constraint.shift + constraint.scale
    return constraint.shift, np.inf


# little boring tools to display parameter names
def correlation_name(model: StatisticalModel, row: int, col: int) -> str:
    return f"corr_{model.pdu_names[row]}_{model.pdu_names[col]}"


def residual_components(model: StatisticalModel) -> list:
    """(name, variance type, output index) of each active residual variance."""
    residual_var = model.residual_var
    return [
        (f"sd_{component}_{output}", var_type, output_index)
        for component, var_type, active_outputs in [
            ("add", "additive_variance", residual_var.additive_output),
            ("prop", "proportional_variance", residual_var.proportional_output),
        ]
        for output_index, (output, active) in enumerate(
            zip(model.output_names, active_outputs.tolist())
        )
        if active
    ]


def to_tensor(values):
    return torch.tensor(values, device=device, dtype=default_dtype)


def get_displayed_estimates(model: StatisticalModel) -> pd.Series:
    """Current estimates on the natural scale: typical values and covariate effects,
    omega (standard deviations and correlations), MI, survival coefficients,
    residual standard deviations (sqrt of the additive / proportional variances)."""
    omega = model.omega_pop
    omega_sd = torch.sqrt(torch.diagonal(omega))
    estimates = {}

    for name, beta in zip(model.beta_names, model.population_betas.tolist()):
        estimates[name] = to_natural(model, name, beta)

    for k, pdu in enumerate(model.pdu_names):
        estimates[f"omega_{pdu}"] = omega_sd[k].item()

    for row in range(model.nb_pdu):
        for col in range(row):
            estimates[correlation_name(model, row, col)] = (
                omega[row, col] / (omega_sd[row] * omega_sd[col])
            ).item()

    for name, log_mi in zip(model.mi_names, model.log_mi.tolist()):
        estimates[name] = to_natural(model, name, log_mi)

    for name, value in zip(model.surv_coeff_names, model.surv_coeffs):
        estimates[name] = value.item()

    for name, field, output_index in residual_components(model):
        variance = getattr(model.residual_var, field)[output_index]
        estimates[name] = torch.sqrt(variance).item()

    return pd.Series(estimates)


def displayed_to_model_state(values: pd.Series, model: StatisticalModel):
    """Inverse of get_displayed_estimates: natural-scale values (indexed by name) ->
    NlmeModelState (beta, Omega, log MI, survival coefficients, residual variances).
    Returns None if a value is non-finite once transformed or Omega is not positive
    definite."""

    beta = to_tensor(
        [to_internal(model, name, values[name]) for name in model.beta_names]
    )
    log_mi = to_tensor(
        [to_internal(model, name, values[name]) for name in model.mi_names]
    )
    surv_coeffs = to_tensor([values[name] for name in model.surv_coeff_names])

    omega_sd = to_tensor([values[f"omega_{pdu}"] for pdu in model.pdu_names])
    correlation = torch.eye(model.nb_pdu, device=device, dtype=default_dtype)
    for row in range(model.nb_pdu):
        for col in range(row):
            correlation[row, col] = values[correlation_name(model, row, col)]
            correlation[col, row] = correlation[row, col]
    omega = torch.diag(omega_sd) @ correlation @ torch.diag(omega_sd)

    residual_var = model.residual_var
    variances = {
        "additive_variance": residual_var.additive_variance.clone(),
        "proportional_variance": residual_var.proportional_variance.clone(),
    }
    for name, field, output_index in residual_components(model):
        variances[field][output_index] = values[name] ** 2

    finite = all(
        torch.isfinite(tensor).all()
        for tensor in [beta, omega, log_mi, surv_coeffs, *variances.values()]
    )
    if not finite or torch.linalg.eigvalsh(omega).min() <= 0:
        return None
    return model.current_params._replace(
        beta=beta,
        omega=omega,
        log_mi=log_mi,
        surv_coeffs=surv_coeffs,
        res_var=residual_var._replace(**variances),
    )


# Here starts the juicy part of the code
def complete_log_likelihood(model, state, gaussian_params, predictions):
    """log p(y_i, phi_i; state) per (sample, patient), with phi_i fixed."""
    if not (
        torch.equal(state.log_mi, model.log_mi)
        and torch.equal(state.surv_coeffs, model.surv_coeffs)
    ):
        physical_params = model.convert_gaussian_to_physical(
            psi=gaussian_params, log_mi=state.log_mi, surv_coeffs=state.surv_coeffs
        )
        thetas = model.convert_physical_to_thetas_all_patients(physical_params)
        model_inputs = model.convert_thetas_to_model_parameters_all_patients(thetas)
        predictions, _ = model.predict_all_patients(model_inputs)
    etas = gaussian_params - model.full_design_matrix @ state.beta
    log_prior_etas_term = torch.distributions.MultivariateNormal(
        loc=torch.zeros(model.nb_pdu, device=device, dtype=default_dtype),
        covariance_matrix=state.omega,
    ).log_prob(
        etas
    )  # computes log p(etas ; omega) but we cannot call log_prior_etas() because it's using omega
    # as it is in the StatisticalModel, not the perturbed one
    log_likelihood_observation_term = log_likelihood_observation(
        model.data.full_obs,
        predictions,
        state.res_var,
        model.config.residual_min_variance,
    )
    return log_prior_etas_term + log_likelihood_observation_term
