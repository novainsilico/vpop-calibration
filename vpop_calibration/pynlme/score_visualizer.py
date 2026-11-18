import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from vpop_calibration.config import device, default_dtype, smoke_test
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.params import Constraint
from vpop_calibration.pynlme.residuals import log_likelihood_observation

IDENTITY_LINK = (lambda x: x, lambda y: y)


def constraint_link(constraint: Constraint):
    shift, scale = constraint.shift, constraint.scale
    if constraint.transform == "log":
        return (lambda x: torch.log(x - shift), lambda y: shift + torch.exp(y))
    return (
        lambda x: torch.log((x - shift) / (shift + scale - x)),
        lambda y: shift + scale / (1 + torch.exp(-y)),
    )


def get_beta_links(model: StatisticalModel) -> list:
    return [
        (
            constraint_link(model.input_params.pdu[name].constraint)
            if name in model.pdu_names
            else IDENTITY_LINK
        )
        for name in model.beta_names
    ]


def get_mi_links(model: StatisticalModel) -> list:
    return [
        constraint_link(model.input_params.model_intrinsic[name].constraint)
        for name in model.mi_names
    ]

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

    for name, (_, inverse_link), beta in zip(
        model.beta_names, get_beta_links(model), model.population_betas
    ):
        estimates[name] = inverse_link(beta).item()

    for k, pdu in enumerate(model.pdu_names):
        estimates[f"omega_{pdu}"] = omega_sd[k].item()

    for row in range(model.nb_pdu):
        for col in range(row):
            estimates[correlation_name(model, row, col)] = (
                omega[row, col] / (omega_sd[row] * omega_sd[col])
            ).item()

    for name, (_, inverse_link), log_mi in zip(
        model.mi_names, get_mi_links(model), model.log_mi
    ):
        estimates[name] = inverse_link(log_mi).item()

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

    def to_internal(names, links):
        return to_tensor(
            [
                link(torch.tensor(values[name], dtype=default_dtype)).item()
                for name, (link, _) in zip(names, links)
            ]
        )

    beta = to_internal(model.beta_names, get_beta_links(model))
    log_mi = to_internal(model.mi_names, get_mi_links(model))
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
    """log p(y_i, phi_i; state) per (sample, patient), with phi_i fixed.
    """
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
    ).log_prob(etas) # computes log p(etas ; omega) but we cannot call log_prior_etas() because it's using omega
                    # as it is in the StatisticalModel, not the perturbed one
    log_likelihood_observation_term = log_likelihood_observation(
        model.data.full_obs,
        predictions,
        state.res_var,
        model.config.residual_min_variance,
    )
    return log_prior_etas_term + log_likelihood_observation_term   


def compute_log_likelihood_around_estimates(
    diagnostics,
    nb_points: int = 11,
    relative_width: float = 0.2,
) -> pd.DataFrame:
    """

    Each parameter is moved on a window of +- relative_width *
    max(|value|, 0.1), the others being fixed. At each point, the complete
    log-likelihood is averaged over the conditional samples (phi_i fixed) and summed
    over patients: sum_i mean_m log p(y_i, phi_i^(m); theta)
    """
 
    model = diagnostics.model
    estimates = get_displayed_estimates(model)

    samples = diagnostics.sampler.total_samples
    predictions = samples.predictions
    offsets = np.linspace(-1.0, 1.0, nb_points)
    rows = []
    gaussian_params = model.convert_etas_to_gaussian_all_patients(
        samples.eta_samples
    )
    for name, estimate in estimates.items():
        half_width = relative_width * max(abs(estimate), 0.1) # in case the variable is estimated to 0, manually tuned on score_visualializer notebook
        for offset in offsets:
            values = estimates.copy()
            values[name] += offset * half_width
            state = displayed_to_model_state(values, model)

            if state is None:
                mean_complete_log_likelihood = np.nan
            else:
                mean_complete_log_likelihood = (
                    complete_log_likelihood(model, state, gaussian_params, predictions)
                    .mean(dim=0)
                    .sum()
                    .item()
                )

            rows.append(
                {
                    "parameter": name,
                    "value": values[name],
                    "complete_log_likelihood": mean_complete_log_likelihood,
                }
            )
    return pd.DataFrame(rows)


def compute_score_samples(diagnostics, relative_step: float = 1e-3):
    """Score g_ik(m) = d/d theta_k log p(y_i, phi_i(m); theta) at the estimates,
    by finite differences, per (sample m, patient i, parameter k).
    """
    model = diagnostics.model
    estimates = get_displayed_estimates(model)
    samples = diagnostics.sampler.total_samples
    gaussian_params = model.convert_etas_to_gaussian_all_patients(
        samples.eta_samples
    )
    scores = []
    for name, estimate in estimates.items():
        step = relative_step * max(abs(estimate), 0.1)
        log_likelihoods = []
        for sign in (1, -1):
            values = estimates.copy()
            values[name] += sign * step
            state = displayed_to_model_state(values, model)
            if state is None: #could be an infinite value or Omega not being positive
                log_likelihoods.append(
                    torch.full(
                        gaussian_params.shape[:2],
                        torch.nan,
                        device=device,
                        dtype=default_dtype,
                    )
                )
            else:
                log_likelihoods.append(
                    complete_log_likelihood(
                        model, state, gaussian_params, samples.predictions
                    )
                )
        scores.append((log_likelihoods[0] - log_likelihoods[1]) / (2 * step))
    return torch.stack(scores, dim=-1), list(estimates.index)


def compute_normalized_scores(diagnostics, relative_step: float = 1e-3) -> pd.DataFrame:
    """ sum_i mean_m g_ik(m) normalized by its noise:
    z_k = sum_i mean_m g_ik(m) / sqrt(sum_i var_m(g_ik) / M), which should follow N(0, 1) at the optimum.
    """
    scores, names = compute_score_samples(diagnostics, relative_step)
    nb_samples = scores.shape[0]
    if nb_samples < 2:
        raise ValueError(
            "At least 2 conditional samples per patient are needed to compute the score."
        )

    total_score = scores.mean(dim=0).sum(dim=0) #g_k
    score_sd = torch.sqrt(scores.var(dim=0).sum(dim=0) / nb_samples) #Var(g_k)

    return pd.DataFrame(
        {
            "total_score": total_score.cpu().numpy(),
            "z_score": (total_score / score_sd).cpu().numpy(),
        },
        index=names,
    )


# a boring tool for improving label visualization :
def compact_tick_label(value, _position=None) -> str:
    for precision in (3, 2, 1):
        label = f"{value:.{precision}g}"
        label = label.replace("-0.", "-.") if len(label) > 4 else label
        label = label.replace("0.", ".", 1) if len(label) > 4 else label
        if len(label) <= 4:
            return label
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"



# plotting
def plot_log_likelihood_around_estimates(
    log_likelihood_variations: pd.DataFrame,
    nb_columns: int = 5,
    facet_width: float = 3.5,
    facet_height: float = 2.8,
    normalized_scores: pd.DataFrame | None = None,
) -> None:
    names = list(dict.fromkeys(log_likelihood_variations["parameter"]))
    nb_rows = int(np.ceil(len(names) / nb_columns))
    fig, axes = plt.subplots(
        nb_rows,
        nb_columns,
        figsize=(facet_width * nb_columns, facet_height * nb_rows),
        squeeze=False,
    )
    for ax, name in zip(axes.flat, names):
        data = log_likelihood_variations[log_likelihood_variations["parameter"] == name]
        estimate = data["value"].iloc[len(data) // 2]
        reference = data["complete_log_likelihood"].iloc[len(data) // 2]
        ax.plot(
            data["value"],
            data["complete_log_likelihood"] - reference,
            "o-",
            
        )
        ax.axvline(estimate, color="red", linewidth=1.2, label="SAEM estimate")
        if normalized_scores is not None:
            y_limits = ax.get_ylim()
            slope = normalized_scores.loc[name, "total_score"]
            ax.plot(
                data["value"],
                slope * (data["value"] - estimate),
                "--",
                color="orange",
                label="Tangent from the score (slope = score)",
            )
            ax.set_ylim(y_limits)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick_label))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(compact_tick_label))
        ax.set_title(name)
        ax.grid(True, linestyle="--", alpha=0.3)
    for ax in list(axes.flat)[len(names) :]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Δ log-likelihood")
    fig.suptitle("Log-likelihood around the estimates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    legend_height = 0.45 / fig.get_figheight()

    if not smoke_test:
        plt.tight_layout(rect=(0, legend_height, 1, 1))
        plt.show()
    plt.close(fig)


def plot_normalized_scores(
    diagnostics,
    relative_step: float = 1e-3,
    bounds: tuple = ( -1, 1),
    max_parameters: int = 40,
    bar_height: float = 0.3,
) -> pd.DataFrame:
    normalized_scores = compute_normalized_scores(diagnostics, relative_step)
    z = normalized_scores["z_score"]
    title = "Normalized score at the estimates"
    if len(z) > max_parameters:
        largest = z.abs().sort_values(ascending=False).index[:max_parameters]
        z = z[z.index.isin(largest)]
        title += f"\n({max_parameters} largest |z| out of {len(normalized_scores)})"

    fig, ax = plt.subplots(figsize=(6, max(2.0, 1.0 + bar_height * len(z))))
    ax.barh(z.index, z)
    for bound in bounds:
        ax.axvline(bound, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_ylim(len(z) - 0.5, -0.5)
    ax.set_xlabel("normalized score")
    ax.set_title(title)

    if not smoke_test:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    return normalized_scores
