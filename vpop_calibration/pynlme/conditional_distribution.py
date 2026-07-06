from tqdm import tqdm
import torch
import pandas as pd
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.config import smoke_test
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step


class ConditionalDistribSamples(NamedTuple):
    samples: torch.Tensor
    log_prob: torch.Tensor


class EbeEstimates(NamedTuple):
    individual_ebe_estimates_tensor: torch.Tensor | None = None
    individual_ebe_estimates_df: pd.DataFrame | None = None
    individual_ebe_predictions_df: pd.DataFrame | None = None


def sample_conditional_distribution_nlme(
    nlme_model: StatisticalModel,
    nb_samples: int = 100,
    nb_burn_in: int = 0,
    plot_frequency: int = 5,
) -> ConditionalDistribSamples:
    """
    Sample random effects from the conditional distribution
    """

    if smoke_test:
        nb_samples = 2
        nb_burn_in = 1

    init_etas = nlme_model.sample_etas(1)
    init_predictions = nlme_model.log_posterior_etas_all_patients(init_etas)
    current_state = MetropolisHastingsState(
        etas=init_etas,
        gaussian_params=init_predictions.gaussian_params,
        prediction=init_predictions.predictions,
        log_prob=init_predictions.log_posterior,
        step_size=0.1,
        complete_likelihood=init_predictions.predictions.sum(dim=0),
    )
    sample_list = []
    log_prob_list = []
    best_log_prob = current_state.log_prob.clone()
    num_improved_history = []
    mean_gain_history = []
    handle, fig, axes, ebe_traces = _build_ebe_convergence_plot()
    print(f"Sampling conditional distribution on {nb_samples} samples:")
    for i in tqdm(range(nb_burn_in + nb_samples)):
        current_state = mh_step(
            nlme_model=nlme_model,
            previous_state=current_state,
            learning_rate=0.0,
        )

        if i >= nb_burn_in:
            sample_list.append(current_state.etas)
            log_prob_list.append(current_state.log_prob)

            delta = current_state.log_prob - best_log_prob
            num_improved = (delta > 0).sum().item()
            mean_gain = delta.clamp(min=0).mean().item()

            num_improved_history.append(num_improved)
            mean_gain_history.append(mean_gain)
            best_log_prob = torch.maximum(
                best_log_prob,
                current_state.log_prob,
            )
            if i % plot_frequency == 0:
                _update_ebe_convergence_plot(
                    handle,
                    fig,
                    axes,
                    ebe_traces,
                    num_improved_history,
                    mean_gain_history,
                )

    _update_ebe_convergence_plot(
        handle, fig, axes, ebe_traces, num_improved_history, mean_gain_history
    )
    samples = torch.stack(sample_list).squeeze(1)
    log_probs = torch.stack(log_prob_list).squeeze(1)
    assert samples.shape == (
        nb_samples,
        nlme_model.nb_patients,
        nlme_model.nb_pdu,
    )
    assert log_probs.shape == (
        nb_samples,
        nlme_model.nb_patients,
    ), f"{log_probs.shape},({nb_samples, nlme_model.nb_patients})"

    _, best_sample_id = log_probs.max(
        dim=0,
    )
    range_indexing = torch.arange(nlme_model.nb_patients)
    ebe_etas = samples[best_sample_id, range_indexing, :].unsqueeze(0)
    ebe_pdus = nlme_model.convert_etas_to_gaussian_all_patients(ebe_etas)
    assert ebe_pdus.shape == (
        1,
        nlme_model.nb_patients,
        nlme_model.nb_pdu,
    ), ebe_pdus.shape
    individual_ebe_estimates_tensor = nlme_model.convert_gaussian_to_physical(
        ebe_pdus, nlme_model.log_mi
    )
    # Compute predictions for these estimates, and store in a data frame
    theta = nlme_model.convert_physical_to_thetas_all_patients(
        individual_ebe_estimates_tensor
    )
    individual_ebe_estimates_df = nlme_model.convert_theta_to_dataframe(theta)
    model_inputs = nlme_model.convert_thetas_to_model_parameters_all_patients(theta)
    individual_ebe_pred, _ = nlme_model.predict_all_patients(model_inputs)
    individual_ebe_predictions_df = nlme_model.data.full_obs.to_pandas(
        prediction=individual_ebe_pred
    )
    return (
        ConditionalDistribSamples(samples=samples, log_prob=log_probs),
        EbeEstimates(
            individual_ebe_estimates_df=individual_ebe_estimates_df,
            individual_ebe_estimates_tensor=individual_ebe_estimates_tensor,
            individual_ebe_predictions_df=individual_ebe_predictions_df,
        ),
    )


def _build_ebe_convergence_plot(plot_indiv_figsize=(5.0, 3.0)):
    fig, axes = plt.subplots(2, 1, figsize=plot_indiv_figsize, sharex=True)

    for ax in axes:
        ax.grid(True)

    axes[0].set_title("EBE convergence")

    axes[0].set_ylabel("Patients improved")
    axes[1].set_ylabel("Mean LL gain")
    axes[1].set_xlabel("Iteration")

    (line1_raw,) = axes[0].plot([], color="lightgray", linewidth=1)
    (line1_ma,) = axes[0].plot([], linewidth=2)
    (line2_raw,) = axes[1].plot([], color="lightgray", linewidth=1)
    (line2_ma,) = axes[1].plot([], linewidth=2)

    ebe_traces = {
        "num_improved": line1_raw,
        "num_improved_ma": line1_ma,
        "mean_gain": line2_raw,
        "mean_gain_ma": line2_ma,
    }

    handle = display(fig, display_id=True)

    return handle, fig, axes, ebe_traces


def _update_ebe_convergence_plot(
    handle, fig, axes, ebe_traces, num_improved_history: list, mean_gain_history: list
):

    x = np.arange(len(num_improved_history))

    mean_gain_ma = moving_average(mean_gain_history, window=20)
    num_improved_ma = moving_average(num_improved_history, window=20)

    ebe_traces["num_improved"].set_data(
        x,
        num_improved_history,
    )
    ebe_traces["num_improved_ma"].set_data(
        x,
        num_improved_ma,
    )
    ebe_traces["mean_gain"].set_data(
        x,
        mean_gain_history,
    )
    ebe_traces["mean_gain_ma"].set_data(
        x,
        mean_gain_ma,
    )

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    if handle is not None:
        handle.update(fig)


def moving_average(x: list[float], window: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")
