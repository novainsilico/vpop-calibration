import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_all_solutions(obs_vs_pred: pd.DataFrame) -> None:
    """Plot the overlapped observations and model predictions for all patients, facetted by output and protocol.

    Args:
        obs_vs_pred (pd.DataFrame): Full data frame containing observations and predicitons from the model. Should contain the following columns
        - `id`
        - `output_name`
        - `protocol_arm`
        - `time`
        - `value`
        - `pred_mean`
    """
    outputs = obs_vs_pred["output_name"].unique().tolist()
    nb_outputs = len(outputs)
    protocol_arms = obs_vs_pred["protocol_arm"].unique().tolist()
    nb_protocol_arms = len(protocol_arms)
    patients = obs_vs_pred["id"].unique().tolist()

    n_cols = nb_outputs
    n_rows = nb_protocol_arms
    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(patients)))
    for output_index, output_name in enumerate(outputs):
        for protocol_index, protocol_arm in enumerate(protocol_arms):
            data_to_plot = obs_vs_pred.loc[
                (obs_vs_pred["output_name"] == output_name)
                & (obs_vs_pred["protocol_arm"] == protocol_arm)
            ]
            ax = axes[protocol_index, output_index]
            ax.set_xlabel("Time")
            for patient_num, patient_ind in enumerate(patients):
                patient_data = data_to_plot.loc[data_to_plot["id"] == patient_ind]
                time_vec = patient_data["time"].values
                sorted_indices = np.argsort(time_vec)
                sorted_times = time_vec[sorted_indices]
                obs_vec = patient_data["value"].values[sorted_indices]
                pred_vec = patient_data["pred_mean"].values[sorted_indices]
                ax.plot(
                    sorted_times,
                    obs_vec,
                    "+",
                    color=colors[patient_num],
                    linewidth=2,
                    alpha=0.6,
                )
                ax.plot(
                    sorted_times,
                    pred_vec,
                    "-",
                    color=colors[patient_num],
                    linewidth=2,
                    alpha=0.5,
                )

            title = f"{output_name} in {protocol_arm}"
            ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_individual_solution(obs_vs_pred: pd.DataFrame) -> None:
    """Plot the model prediction (and confidence interval) vs. the input data for a single patient"""
    outputs = obs_vs_pred["output_name"].unique().tolist()
    nb_outputs = len(outputs)
    protocol_arms = obs_vs_pred["protocol_arm"].unique().tolist()
    nb_protocol_arms = len(protocol_arms)
    patients = obs_vs_pred["id"].unique().tolist()
    assert len(patients) == 1
    patient_id = patients[0]
    ncols = nb_outputs
    nrows = nb_protocol_arms
    _, axes = plt.subplots(nrows, ncols, figsize=(9.0 * nb_outputs, 4.0), squeeze=False)

    patient_params = obs_vs_pred.drop(
        columns=[
            "id",
            "output_name",
            "protocol_arm",
            "value",
            "pred_mean",
            "pred_low",
            "pred_high",
        ]
    ).drop_duplicates()

    for output_index, output_name in enumerate(outputs):
        for protocol_index, protocol_arm in enumerate(protocol_arms):
            data_to_plot = obs_vs_pred.loc[
                (obs_vs_pred["output_name"] == output_name)
                & (obs_vs_pred["protocol_arm"] == protocol_arm)
            ]
            time_steps = np.array(data_to_plot["time"].values)
            sorted_indices = np.argsort(time_steps)
            sorted_time_steps = time_steps[sorted_indices]
            ax = axes[protocol_index, output_index]
            ax.set_xlabel("Time")
            # Plot observations
            ax.plot(
                sorted_time_steps,
                data_to_plot["value"].values[sorted_indices],
                ".-",
                color="C0",
                linewidth=2,
                alpha=0.6,
                label=output_name,
            )

            # Plot model prediction
            ax.plot(
                sorted_time_steps,
                data_to_plot["pred_mean"].values[sorted_indices],
                "-",
                color="C3",
                linewidth=2,
                alpha=0.5,
                label="GP prediction for " + output_name + " (mean)",
            )
            # Add confidence interval
            ax.fill_between(
                sorted_time_steps,
                data_to_plot["pred_low"].values[sorted_indices],
                data_to_plot["pred_high"].values[sorted_indices],
                alpha=0.5,
                color="C3",
                label="GP prediction for " + output_name + " (CI)",
            )

            ax.legend(loc="upper right")
            title = f"{output_name} in {protocol_arm} for patient {patient_id}"
            ax.set_title(title)

            param_text = "Parameters:\n"
            for param in patient_params:
                param_text += f"  {param}: {patient_params[param][0]:.3f}\n"  # Format to 4 decimal places

            ax.text(
                1.02,
                0.98,
                param_text,
                transform=ax.transAxes,  # Coordinate system is relative to the axis
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5, ec="k"),
            )

    plt.tight_layout()
    plt.show()


def plot_obs_vs_predicted(
    obs_vs_pred: pd.DataFrame, logScale: Optional[list[bool]] = None
) -> None:
    """Plots the observed vs. predicted values on the training or validation data set, or on a new data set."""

    outputs = obs_vs_pred["output_name"].unique().tolist()
    nb_outputs = len(outputs)
    protocol_arms = obs_vs_pred["protocol_arm"].unique().tolist()
    nb_protocol_arms = len(protocol_arms)
    patients = obs_vs_pred["id"].unique().tolist()

    n_cols = nb_outputs
    n_rows = nb_protocol_arms
    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    if not logScale:
        logScale = [True] * nb_outputs

    for output_index, output_name in enumerate(outputs):
        for protocol_index, protocol_arm in enumerate(protocol_arms):
            log_viz = logScale[output_index]
            ax = axes[protocol_index, output_index]
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            data_to_plot = obs_vs_pred.loc[
                (obs_vs_pred["protocol_arm"] == protocol_arm)
                & (obs_vs_pred["output_name"] == output_name)
            ]
            for ind in patients:
                patient_data = data_to_plot.loc[data_to_plot["id"] == ind]
                obs_vec = patient_data["value"]
                pred_vec = patient_data["pred_mean"]
                ax.plot(
                    obs_vec,
                    pred_vec,
                    "o",
                    linewidth=1,
                    alpha=0.6,
                )

            min_val = data_to_plot["value"].min().min()
            max_val = data_to_plot["value"].max().max()
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "-",
                linewidth=1,
                alpha=0.5,
                color="black",
            )
            ax.fill_between(
                [min_val, max_val],
                [min_val / 2, max_val / 2],
                [min_val * 2, max_val * 2],
                linewidth=1,
                alpha=0.25,
                color="black",
            )
            title = f"{output_name} in {protocol_arm}"  # More descriptive title
            ax.set_title(title)
            if log_viz:
                ax.set_xscale("log")
                ax.set_yscale("log")
    plt.tight_layout()
    plt.show()


def plot_loss(iterations: np.ndarray, losses: np.ndarray) -> None:
    # plot the loss over iterations
    plt.plot(iterations, losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.show()
