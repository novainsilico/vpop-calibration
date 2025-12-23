import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .nlme import NlmeModel
from .saem import PySaem
from .model.gp import GP
from .structural_model import StructuralGp
from .utils import smoke_test


def check_surrogate_validity_gp(nlme_model: NlmeModel) -> tuple[dict, dict]:
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

    scaling_indiv_plots = 3
    _, axes1 = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        figsize=[scaling_indiv_plots * n_cols, scaling_indiv_plots * n_rows],
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

    scaling_2by2_plots = 2
    _, axes2 = plt.subplots(
        n_plots,
        n_plots,
        squeeze=False,
        figsize=[scaling_2by2_plots * n_plots, scaling_2by2_plots * n_plots],
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


def plot_map_estimates(nlme_model: NlmeModel) -> None:
    observed = nlme_model.observations_df
    simulated_df = nlme_model.map_estimates_predictions()

    n_cols = nlme_model.nb_outputs
    n_rows = nlme_model.structural_model.nb_protocols
    _, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
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
