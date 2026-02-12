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
import scipy.stats as stats


def check_surrogate_validity_gp(
    nlme_model: NlmeModel,
    scaling_indiv_plot: float = 3.0,
    scaling_2by2_plot: float = 2.0,
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
    nlme_model: NlmeModel, facet_width: float = 5.0, facet_height: float = 4.0
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
    facet_width: float = 5.0,
    facet_height: float = 4.0,
    verbose: bool = False,
) -> None:

    # Plot a random patient as default
    if patient_num is None:
        total_patient_num = len(nlme_model.patients)
        patient_num = rand.randrange(total_patient_num)

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
    facet_width: float = 5.0,
    facet_height: float = 4.0,
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
        ind_to_plot = list(range(n_patients_to_plot))

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
    nlme_model: NlmeModel,
    facet_width: float = 8.0,
    facet_height: float = 8.0,
    tolerance_ribbon: str = "mean",
    tolerance_pct: int = 50,
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

        # Plot tolerance interval
        all_vals = gof_df[["value", "predicted_value"]]
        min_val = all_vals.min().min()
        max_val = all_vals.max().max()

        margin = (max_val - min_val) * 0.05
        range_val = [min_val - margin, max_val + margin]

        match tolerance_ribbon:
            case "relative":
                tol = [i * tolerance_pct / 100 for i in range_val]
            case "median":
                tol = (
                    all_vals["value"].median()
                    * tolerance_pct
                    / 100
                    * np.ones_like(range_val)
                )
            case "mean":
                tol = (
                    all_vals["value"].mean()
                    * tolerance_pct
                    / 100
                    * np.ones_like(range_val)
                )
            case _:
                tol = np.zeros_like(range_val)
        lower_bound = [val - tolerance for val, tolerance in zip(range_val, tol)]
        upper_bound = [val + tolerance for val, tolerance in zip(range_val, tol)]

        ax.plot(range_val, range_val, color="red", linestyle="-", linewidth=1.5)
        ax.fill_between(
            range_val,
            lower_bound,
            upper_bound,
            color="grey",
            linestyle="--",
            linewidth=1.5,
            alpha=0.15,
            label=f"CI: {tolerance_pct} % {tolerance_ribbon}",
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
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()

    if not smoke_test:
        plt.show()

    plt.close(fig)


def plot_weighted_residuals(
    nlme_model: NlmeModel,
    res_type: str,
    num_samples: int = 100,
    facet_width: int = 10,
    facet_height: int = 10,
) -> None:

    match res_type:
        case "pwres":
            if smoke_test:
                num_samples = 2
            wres_results = nlme_model.compute_pwres(num_samples)
            compare_to_pop_pred = True
        case "iwres":
            wres_results = nlme_model.compute_iwres()
            compare_to_pop_pred = False
        case "npde":
            if smoke_test:
                num_samples = 2
            wres_results = nlme_model.compute_npde(num_samples)
            compare_to_pop_pred = True
        case _:
            raise ValueError(f"Not implemented residual type: {res_type}")

    all_wres = np.concatenate([p[res_type] for p in wres_results.values()]).flatten()
    all_times = np.concatenate([p["time"] for p in wres_results.values()]).flatten()

    fig, ax = plt.subplots(2, 2, figsize=(facet_width, facet_height))

    ## Histogram plot
    ax[0, 0].hist(
        all_wres, bins=30, density=True, alpha=0.6, color="skyblue", edgecolor="black"
    )
    mu, std = 0, 1
    x = np.linspace(min(all_wres), max(all_wres), 100)
    p = stats.norm.pdf(x, mu, std)
    ax[0, 0].plot(x, p, "r", linewidth=2, label=r"$\mathcal{N}(0,1)$")
    ax[0, 0].set_title(f"{res_type.upper()} distribution")
    ax[0, 0].set_xlabel("Residual values")
    ax[0, 0].set_ylabel("Density")
    ax[0, 0].legend()

    ## Q-Q plot
    stats.probplot(all_wres, dist="norm", plot=ax[1, 0])
    ax[1, 0].set_title(f"{res_type.upper()} Q-Q Plot")

    ## Plot vs. time
    ax[0, 1].grid(True, linestyle="--", alpha=0.6, which="both")
    ax[0, 1].set_facecolor("#fdfdfd")
    ax[0, 1].scatter(
        all_times,
        all_wres,
        alpha=0.5,
        color="#2c3e50",
        edgecolors="white",
        s=45,
        zorder=3,
    )
    ax[0, 1].axhline(y=0, color="black", linestyle="-", linewidth=1.5, zorder=4)
    ax[0, 1].axhline(
        y=1.96,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.3,
        label=r"95% CI Limit ($\pm 1.96$)",
    )
    ax[0, 1].axhline(y=-1.96, color="#e74c3c", linestyle="--", linewidth=1.3)
    ax[0, 1].set_xlabel("Time", fontsize=12)
    ax[0, 1].set_ylabel("Weighted Residual (Standard Deviations)", fontsize=12)
    ax[0, 1].set_ylim(-1.1 * max(abs(all_wres)), 1.1 * max(abs(all_wres)))
    ax[0, 1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax[0, 1].set_title(f"{res_type.upper()} vs. Time")

    ## Plot vs. predictions

    if compare_to_pop_pred:
        all_predictions = nlme_model.map_predictions_eta_zero()
    else:
        all_predictions = nlme_model.map_estimates_predictions()

    # Transform WRES dict into a dataframe
    rows = []
    for patient_id, content in wres_results.items():
        for wres, time in zip(content[res_type], content["time"]):
            rows.append({"id": patient_id, res_type: wres, "time": time})

    wres_df = pd.DataFrame(rows)

    # Merge WRES with predictions, matching patientID and time
    vs_pred_plot_df = pd.merge(
        wres_df, all_predictions[["id", "time", "predicted_value"]], on=["id", "time"]
    )

    wres_to_plot = vs_pred_plot_df[res_type]
    pred_to_plot = vs_pred_plot_df["predicted_value"]
    ax[1, 1].set_facecolor("#fdfdfd")
    ax[1, 1].scatter(
        pred_to_plot,
        wres_to_plot,
        alpha=0.5,
        color="#2c3e50",
        edgecolors="white",
        s=45,
        zorder=3,
    )
    ax[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=1.5, zorder=4)
    ax[1, 1].axhline(
        y=1.96,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.3,
        label=r"95% CI Limit ($\pm 1.96$)",
    )
    ax[1, 1].axhline(y=-1.96, color="#e74c3c", linestyle="--", linewidth=1.3)

    if compare_to_pop_pred:
        ax[1, 1].set_xlabel("Population Predictions")
    else:
        ax[1, 1].set_xlabel("Individual Predictions")

    ax[1, 1].set_ylabel("Weighted Residual (Standard Deviations)")
    ax[1, 1].set_ylim(-1.1 * max(abs(wres_to_plot)), 1.1 * max(abs(wres_to_plot)))
    ax[1, 1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax[1, 1].set_title(f"{res_type.upper()} vs. Predictions")

    plt.tight_layout()

    if not smoke_test:
        plt.show()

    plt.close(fig)


def plot_map_vs_posterior(
    nlme_model: NlmeModel,
    nb_samples: int,
    n_patients_to_plot: int,
):

    # Sample new etas, compute physical
    sample_etas = nlme_model.sample_conditional_distribution(nb_samples)
    sample_gaussian = nlme_model.etas_to_gaussian_params(sample_etas)
    sample_physical = nlme_model.gaussian_to_physical_params(
        sample_gaussian, nlme_model.log_MI
    )

    total_patient_num = len(nlme_model.patients)
    ind_to_plot = rand.sample(range(total_patient_num), n_patients_to_plot)

    # Get EBE estimates for descriptors
    ebe_theta = nlme_model.ebe_estimates

    # Get false MAP for comparison
    map_theta = nlme_model.map_estimates_descriptors()

    nb_samples, nb_patients, nb_params = sample_physical.shape

    for k in range(n_patients_to_plot):
        patient_id = nlme_model.patients[ind_to_plot[k]]
        patient_samples = sample_physical[:, ind_to_plot[k], :].detach().cpu().numpy()

        # Adapt rows to columns
        n_cols = 3
        n_rows = (nb_params + n_cols - 1) // n_cols
        _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        # Plot distribution and MAP for each PDU
        for i in range(len(axes)):
            ax = axes[i]
            if i < nb_params:
                param_data = patient_samples[:, i]

                if np.unique(param_data).size > 1:
                    kde = stats.gaussian_kde(param_data)
                    x_range = np.linspace(param_data.min(), param_data.max(), 200)
                    ax.plot(
                        x_range, kde(x_range), color="blue", lw=1.5, label="PDF (KDE)"
                    )

                map_val = ebe_theta[0][ind_to_plot[k]][i]
                map_theta_val = map_theta.iloc[ind_to_plot[k], i + nlme_model.nb_PDK]

                ax.axvline(
                    map_val,
                    color="red",
                    linewidth=1.5,
                    linestyle="dashed",
                    label=f"MAP estimate: {map_val:.2f}",
                )

                ax.axvline(
                    param_data.mean(),
                    color="blue",
                    linewidth=1.5,
                    linestyle="dashed",
                    label=f"Conditional mean: {param_data.mean():.2f}",
                )

                ax.axvline(
                    map_theta_val,
                    color="yellow",
                    linewidth=1.5,
                    linestyle="dashed",
                    label=f"Previous MAP estimate: {map_theta_val:.2f}",
                )

                ci_low, ci_high = np.percentile(param_data, [2.5, 97.5])
                ax.axvspan(
                    ci_low,
                    ci_high,
                    color="gray",
                    alpha=0.2,
                    label=f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]",
                )

                ax.set_title(f"Patient {ind_to_plot[k]} - {nlme_model.PDU_names[i]}")
                ax.legend(fontsize="small")

            else:
                # Hide plot if empty
                ax.set_visible(False)

        plt.tight_layout()
        plt.show()
