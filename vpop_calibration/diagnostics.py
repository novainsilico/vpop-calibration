import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random as rand
from .nlme import NlmeModel
from .saem import PySaem
from .model.gp import GP
from .structural_model import StructuralGp
from .utils import smoke_test


def check_surrogate_validity_gp(
    nlme_model: NlmeModel, scaling_indiv_plot: int = 3, scaling_2by2_plot: int = 2
) -> tuple[dict, dict]:
    pdus = nlme_model.descriptors
    gp_model_struct = nlme_model.structural_model
    assert isinstance(
        gp_model_struct, StructuralGp
    ), "Posterior surrogate validity check only implemented for GP structural model."

    gp_model: GP = gp_model_struct.gp_model
    train_data = gp_model.data.full_df_raw[pdus].drop_duplicates()

    map_data = nlme_model.map_estimates_descriptors()
    patients = nlme_model.patients

    n_plots = len(pdus)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    _, axes1 = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        figsize=[scaling_indiv_plot * n_cols, scaling_indiv_plot * n_rows],
    )
    diagnostics = {}
    recommended_ranges = {}
    for k, param in enumerate(pdus):
        i, j = k // n_cols, k % n_cols
        train_samples = np.log(train_data[param])
        train_min, train_max = train_samples.min(axis=0), train_samples.max(axis=0)

        map_samples = np.log(map_data[param])
        flag_high = np.where(map_samples > train_max)[0]
        flag_low = np.where(map_samples < train_min)[0]
        recommend_low, recommend_high = train_min, train_max
        param_diagnostic = {}
        if flag_high.shape[0] > 0:
            param_diagnostic.update({"above": [patients[p] for p in flag_high]})
            recommend_high = map_samples.max()
        else:
            param_diagnostic.update({"above": None})
        if flag_low.shape[0] > 0:
            param_diagnostic.update({"below": [patients[p] for p in flag_low]})
            recommend_low = map_samples.min()
        else:
            param_diagnostic.update({"below": None})
        diagnostics.update({param: param_diagnostic})
        recommended_ranges.update(
            {
                param: {
                    "low": f"{recommend_low:.2f}",
                    "high": f"{recommend_high:.2f}",
                    "log": True,
                }
            }
        )

        ax = axes1[i, j]
        ax.hist([train_samples, map_samples], density=True)
        ax.axvline(train_min, linestyle="dashed", color="black")
        ax.axvline(train_max, linestyle="dashed", color="black")
        ax.set_title(f"{param}")

    _, axes2 = plt.subplots(
        n_plots,
        n_plots,
        squeeze=False,
        figsize=[scaling_2by2_plot * n_plots, scaling_2by2_plot * n_plots],
        sharex="col",
        sharey="row",
    )
    for k1, param1 in enumerate(pdus):
        train_samples_1 = np.log(train_data[param1])
        map_samples_1 = np.log(map_data[param1])
        for k2, param2 in enumerate(pdus):
            train_samples_2 = np.log(train_data[param2])
            map_samples_2 = np.log(map_data[param2])
            ax = axes2[k1, k2]
            if k1 != k2:
                # param 1 is the row -> y axis
                # param 2 is the column -> x axis
                ax.scatter(train_samples_2, train_samples_1, alpha=0.5, s=1.0)
                ax.scatter(map_samples_2, map_samples_1, s=5)
            if k2 == 0:
                ax.set_ylabel(param1)
            if k1 == len(pdus) - 1:
                ax.set_xlabel(param2)

    if not smoke_test:
        plt.tight_layout()
        plt.show()
    return diagnostics, recommended_ranges


def plot_map_estimates(
    nlme_model: NlmeModel, facet_width: int = 5, facet_height: int = 4
) -> None:
    observed = nlme_model.observations_df
    simulated_df = nlme_model.map_estimates_predictions()

    n_cols = nlme_model.nb_outputs
    n_rows = nlme_model.structural_model.nb_protocols
    _, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(facet_width * n_cols, facet_height * n_rows),
        squeeze=False,
    )

    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, nlme_model.nb_patients))
    for output_index, output_name in enumerate(nlme_model.outputs_names):
        for protocol_index, protocol_arm in enumerate(
            nlme_model.structural_model.protocols
        ):
            obs_loop = observed.loc[
                (observed["output_name"] == output_name)
                & (observed["protocol_arm"] == protocol_arm)
            ]
            pred_loop = simulated_df.loc[
                (simulated_df["output_name"] == output_name)
                & (simulated_df["protocol_arm"] == protocol_arm)
            ]
            ax = axes[protocol_index, output_index]
            ax.set_xlabel("Time")
            patients_protocol = obs_loop["id"].drop_duplicates().to_list()
            for patient_ind in patients_protocol:
                patient_num = nlme_model.patients.index(patient_ind)
                patient_obs = obs_loop.loc[obs_loop["id"] == patient_ind]
                patient_pred = pred_loop.loc[pred_loop["id"] == patient_ind]
                time_vec = patient_obs["time"].values
                sorted_indices = np.argsort(time_vec)
                sorted_times = time_vec[sorted_indices]
                obs_vec = patient_obs["value"].values[sorted_indices]
                ax.plot(
                    sorted_times,
                    obs_vec,
                    "+",
                    color=colors[patient_num],
                    linewidth=2,
                    alpha=0.6,
                )
                if patient_pred.shape[0] > 0:
                    pred_vec = patient_pred["predicted_value"].values[sorted_indices]
                    ax.plot(
                        sorted_times,
                        pred_vec,
                        "-",
                        color=colors[patient_num],
                        linewidth=2,
                        alpha=0.5,
                    )

            title = f"{output_name} in {protocol_arm}"  # More descriptive title
            ax.set_title(title)

    if not smoke_test:
        plt.tight_layout()
        plt.show()


def plot_individual_map_estimates(
    nlme_model: NlmeModel,
    patient_num: int | None = None,
    facet_width: int = 5,
    facet_height: int = 4,
    verbose: bool = False,
) -> None:

    # Plot a random patient as default
    if patient_num is None:
        total_patient_num = len(nlme_model.patients)
        patient_num = rand.randrange(0, total_patient_num)

    # Filter datasets for the selected patient
    observed_df = nlme_model.observations_df
    simulated_df = nlme_model.map_estimates_predictions()
    patient_ind = nlme_model.patients[patient_num]
    patient_obs = observed_df.loc[(observed_df["id"] == patient_ind)]
    patient_pred = simulated_df.loc[(simulated_df["id"] == patient_ind)]

    # Print patient parameters if verbose selected
    if verbose:
        patient_params = nlme_model.map_estimates_descriptors()
        print(patient_params.loc[patient_params["id"] == patient_ind])

    # Initialize subplots
    n_cols = nlme_model.nb_outputs
    n_rows = 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(facet_width * n_cols, facet_height * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"Outputs for patient {patient_num}")

    # Initialize colormap according to outputs
    cmap = plt.get_cmap("brg")
    colors = cmap(np.linspace(0, 1, len(nlme_model.outputs_names)))

    for output_index, output_name in enumerate(nlme_model.outputs_names):
        # Filter dataset on current output
        patient_obs_output = patient_obs.loc[
            (patient_obs["output_name"] == output_name)
        ]
        patient_pred_output = patient_pred.loc[
            (patient_pred["output_name"] == output_name)
        ]

        # Sort dataset w.r.t time
        time_vec = patient_obs_output["time"].to_numpy()
        sorted_indices = np.argsort(time_vec)
        sorted_times = time_vec[sorted_indices]

        ax = axes[0, output_index]
        ax.set_xlabel("Time")

        # Plot observed and predicted vectors
        if patient_obs_output.shape[0] > 0:
            obs_vec = patient_obs_output["value"].values[sorted_indices]
            ax.plot(
                sorted_times,
                obs_vec,
                "+",
                color=colors[output_index],
                linewidth=2,
                alpha=0.6,
            )

        if patient_pred_output.shape[0] > 0:
            pred_vec = patient_pred_output["predicted_value"].values[sorted_indices]
            ax.plot(
                sorted_times,
                pred_vec,
                "-",
                color=colors[output_index],
                linewidth=2,
                alpha=0.5,
            )

        title = f"{output_name}"
        ax.set_title(title)
        plt.tight_layout()

    if not smoke_test:
        plt.show()

    plt.close(fig)


def plot_all_individual_map_estimates(
    nlme_model: NlmeModel,
    n_rows: int = 1,
    n_cols: int = 5,
    n_patients_to_plot: int | None = None,
    facet_width: int = 5,
    facet_height: int = 4,
    randomize: bool = False,
) -> None:

    observed_df = nlme_model.observations_df
    simulated_df = nlme_model.map_estimates_predictions()

    total_patient_num = len(nlme_model.patients)

    # Plot all patients by default
    if n_patients_to_plot is None or n_patients_to_plot > total_patient_num:
        n_patients_to_plot = total_patient_num

    print(
        f"There are {total_patient_num} patients in total. {n_patients_to_plot} will be plotted."
    )

    # Raise an error if too many patients for the grid
    if n_patients_to_plot > n_rows * n_cols:
        raise ValueError(
            f"{n_patients_to_plot} patients cannot be plotted in a {n_rows}x{n_cols} grid. Enter a n_patients_to_plot value under {n_rows*n_cols} or use a larger grid."
        )

    if randomize:
        ind_to_plot = rand.sample(range(total_patient_num), n_patients_to_plot)
    else:
        ind_to_plot = np.linspace(0, n_patients_to_plot, n_patients_to_plot)

    cmap = plt.get_cmap("brg")
    colors = cmap(np.linspace(0, 1, len(nlme_model.outputs_names)))

    # One plot for each output, containing all individual patients subplots for this output
    for output_index, output_name in enumerate(nlme_model.outputs_names):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(facet_width * n_cols, facet_height * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"Output: {output_name}")

        patient_obs_output = observed_df.loc[
            (observed_df["output_name"] == output_name)
        ]
        patient_pred_output = simulated_df.loc[
            (simulated_df["output_name"] == output_name)
        ]

        for k in range(0, n_patients_to_plot):
            # Change indexing from 1d to 2d
            i = k // n_cols
            j = k % n_cols
            ax = axes[i, j]
            ax.set_xlabel("Time")

            # Filter dataset for current patient
            patient_ind = nlme_model.patients[ind_to_plot[k]]
            patient_obs = patient_obs_output.loc[
                patient_obs_output["id"] == patient_ind
            ]
            patient_pred = patient_pred_output.loc[
                patient_pred_output["id"] == patient_ind
            ]

            time_vec = patient_obs["time"].to_numpy()
            sorted_indices = np.argsort(time_vec)
            sorted_times = time_vec[sorted_indices]

            if patient_obs.shape[0] > 0:
                obs_vec = patient_obs["value"].values[sorted_indices]
                ax.plot(
                    sorted_times,
                    obs_vec,
                    "+",
                    color=colors[output_index],
                    linewidth=2,
                    alpha=0.6,
                )
            if patient_pred.shape[0] > 0:
                pred_vec = patient_pred["predicted_value"].values[sorted_indices]
                ax.plot(
                    sorted_times,
                    pred_vec,
                    "-",
                    color=colors[output_index],
                    linewidth=2,
                    alpha=0.5,
                )

            title = f"patient {ind_to_plot[k]}"
            ax.set_title(title)
            plt.tight_layout()

    if not smoke_test:
        plt.show()

    plt.close(fig)


def plot_map_estimates_gof(
    nlme_model: NlmeModel, facet_width: int = 8, facet_height: int = 8
) -> None:

    observed_df = nlme_model.observations_df
    simulated_df = nlme_model.map_estimates_predictions()
    sim_vs_obs_df = simulated_df.merge(observed_df, on=["time", "id", "output_name"])

    unique_outputs = sim_vs_obs_df["output_name"].unique()
    num_plots = len(unique_outputs)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(facet_width * num_plots, facet_height), squeeze=False
    )

    fig.suptitle(f"Observed vs. simulated plot")

    for output_index, output_name in enumerate(nlme_model.outputs_names):

        ax = axes[0, output_index]
        gof_df = sim_vs_obs_df.loc[(sim_vs_obs_df["output_name"] == output_name)]

        # Compute R² and RMSE
        r2 = r2_score(gof_df["value"], gof_df["predicted_value"])
        rmse = np.sqrt(np.mean((gof_df["value"] - gof_df["predicted_value"]) ** 2))
        metrics_text = f"$R^2 = {r2:.3f}$\n$RMSE= {rmse:.3f}$"

        # Plot (obs,pred) points
        ax.scatter(
            x=gof_df["value"],
            y=gof_df["predicted_value"],
            alpha=0.7,
            s=50,
            edgecolors="w",
        )

        # Plot 2x interval
        all_vals = gof_df[["value", "predicted_value"]]
        min_val = all_vals.min().min()
        max_val = all_vals.max().max()
        margin = (max_val - min_val) * 0.05
        range_val = [min_val - margin, max_val + margin]
        ax.plot(range_val, range_val, color="red", linestyle="-", linewidth=1.5)
        ax.plot(
            range_val,
            [i * 2 for i in range_val],
            color="red",
            linestyle="--",
            linewidth=1.5,
        )
        ax.plot(
            range_val,
            [i / 2 for i in range_val],
            color="red",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlim(range_val)
        ax.set_ylim(range_val)
        ax.grid(True, linestyle=":", alpha=0.6)

        ax.set_xlabel("observed", fontsize=12)
        ax.set_ylabel("simulated", fontsize=12)

        ax.text(
            0.95,
            0.05,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.7, edgecolor="lightgray"
            ),
            fontsize=11,
        )

        title = f"Output: {output_name}"
        ax.set_title(title)
        plt.tight_layout()

    if not smoke_test:
        plt.show()
        print("ok")

    plt.close(fig)
