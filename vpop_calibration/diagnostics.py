import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .nlme import NlmeModel
from .saem import PySaem
from .model.gp import GP
from .structural_model import StructuralGp


def check_surrogate_validity_gp(optimizer: PySaem):
    nlme_model: NlmeModel = optimizer.model
    pdus = nlme_model.descriptors
    gp_model_struct = nlme_model.structural_model
    assert isinstance(
        gp_model_struct, StructuralGp
    ), "Posterior surrogate validity check only implemented for GP structural model."

    gp_model: GP = gp_model_struct.gp_model
    train_data = gp_model.data.full_df_raw[pdus].drop_duplicates()

    map_estimates = optimizer.current_thetas
    map_data = pd.DataFrame(data=map_estimates.numpy(), columns=pdus)
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
    return diagnostics, recommended_ranges
