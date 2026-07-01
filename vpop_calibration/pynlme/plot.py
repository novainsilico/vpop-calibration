import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random as rand
import scipy.stats as stats


from vpop_calibration.pynlme.diagnostics import (
    ModelDiagnostics,
    ModelResiduals,
    ResidualType,
)
from vpop_calibration.model.gp import GP
from vpop_calibration.structural_model.gp import StructuralGp
from vpop_calibration.config import smoke_test


class PlottingUtility:
    def __init__(self, diagnostics: ModelDiagnostics):
        self.model_diag = diagnostics

    def check_surrogate_validity_gp(
        self,
        scaling_indiv_plot: float = 3.0,
        scaling_2by2_plot: float = 2.0,
        n_columns: int = 3,
    ) -> tuple[dict, dict]:
        pdus = self.model_diag.model.descriptors
        gp_model_struct = self.model_diag.model.structural_model
        assert isinstance(
            gp_model_struct, StructuralGp
        ), "Posterior surrogate validity check only implemented for GP structural model."

        if self.model_diag.individual_ebe_estimates_df is None:
            self.model_diag.compute_ebe()
        assert self.model_diag.individual_ebe_estimates_df is not None
        gp_model: GP = gp_model_struct.gp_model
        train_data = gp_model.data.full_df_raw[pdus].drop_duplicates()

        map_data = self.model_diag.individual_ebe_estimates_df
        patients = self.model_diag.model.patients

        n_plots = len(pdus)
        n_cols = n_columns
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

    def map_estimates(
        self,
        facet_width: float = 5.0,
        facet_height: float = 4.0,
        max_iter: int = 100,
    ) -> None:
        if self.model_diag.individual_ebe_predictions_df is None:
            self.model_diag.compute_ebe(max_iter)
        assert self.model_diag.individual_ebe_predictions_df is not None
        obs_vs_simulated = self.model_diag.individual_ebe_predictions_df

        n_cols = self.model_diag.model.nb_outputs
        n_rows = self.model_diag.model.nb_protocols
        _, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(facet_width * n_cols, facet_height * n_rows),
            squeeze=False,
        )

        cmap = plt.get_cmap("Spectral")
        colors = cmap(np.linspace(0, 1, self.model_diag.model.nb_patients))
        for output_index, output_name in enumerate(self.model_diag.model.output_names):
            for protocol_index, protocol_arm in enumerate(
                self.model_diag.model.protocol_arms
            ):
                data_loop = obs_vs_simulated.loc[
                    (obs_vs_simulated["output_name"] == output_name)
                    & (obs_vs_simulated["protocol_arm"] == protocol_arm)
                ]
                if data_loop.shape[0] == 0:
                    pass
                ax = axes[protocol_index, output_index]
                ax.set_xlabel("Time")
                patients_protocol = data_loop["id"].drop_duplicates().to_list()
                for patient_ind in patients_protocol:
                    patient_num = self.model_diag.model.patients.index(patient_ind)
                    patient_data = data_loop.loc[data_loop["id"] == patient_ind]
                    time_vec = patient_data["time"].values
                    sorted_indices = np.argsort(time_vec)
                    sorted_times = time_vec[sorted_indices]
                    obs_vec = patient_data["value"].values[sorted_indices]
                    pred_vec = patient_data["predicted_value"].values[sorted_indices]
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

        if not smoke_test:
            plt.tight_layout()
            plt.show()

    def individual_map_estimates(
        self,
        patient_num: int | None = None,
        facet_width: float = 5.0,
        facet_height: float = 4.0,
        verbose: bool = False,
    ) -> None:

        # Plot a random patient as default
        if patient_num is None:
            total_patient_num = self.model_diag.model.nb_patients
            patient_num = rand.randrange(total_patient_num)

        if self.model_diag.individual_ebe_predictions_df is None:
            self.model_diag.compute_ebe()
        assert self.model_diag.individual_ebe_predictions_df is not None
        assert self.model_diag.individual_ebe_estimates_df is not None
        # Filter datasets for the selected patient
        obs_vs_simulated = self.model_diag.individual_ebe_predictions_df

        patient_ind = self.model_diag.model.patients[patient_num]
        patient_data = obs_vs_simulated.loc[obs_vs_simulated["id"] == patient_ind]

        # Print patient parameters if verbose selected
        if verbose:
            patient_params = self.model_diag.individual_ebe_estimates_df
            print(patient_params.loc[patient_params["id"] == patient_ind])

        # Initialize subplots
        n_cols = self.model_diag.model.nb_outputs
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
        colors = cmap(np.linspace(0, 1, len(self.model_diag.model.output_names)))

        for output_index, output_name in enumerate(self.model_diag.model.output_names):
            # Filter dataset on current output
            data_output = patient_data.loc[patient_data["output_name"] == output_name]
            if data_output.shape[0] == 0:
                pass

            # Sort dataset w.r.t time
            time_vec = data_output["time"].to_numpy()
            sorted_indices = np.argsort(time_vec)
            sorted_times = time_vec[sorted_indices]

            ax = axes[0, output_index]
            ax.set_xlabel("Time")

            obs_vec = data_output["value"].values[sorted_indices]
            ax.plot(
                sorted_times,
                obs_vec,
                "+",
                color=colors[output_index],
                linewidth=2,
                alpha=0.6,
            )

            pred_vec = data_output["predicted_value"].values[sorted_indices]
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

    def all_individual_map_estimates(
        self,
        n_rows: int = 1,
        n_cols: int = 5,
        n_patients_to_plot: int | None = None,
        facet_width: float = 5.0,
        facet_height: float = 4.0,
        randomize: bool = False,
    ) -> None:

        if self.model_diag.individual_ebe_predictions_df is None:
            self.model_diag.compute_ebe()
        assert self.model_diag.individual_ebe_predictions_df is not None
        obs_vs_simulated = self.model_diag.individual_ebe_predictions_df

        # Plot all patients by default
        if (
            n_patients_to_plot is None
            or n_patients_to_plot > self.model_diag.model.nb_patients
        ):
            n_patients_to_plot = self.model_diag.model.nb_patients

        print(
            f"There are {self.model_diag.model.nb_patients} patients in total. {n_patients_to_plot} will be plotted."
        )

        # Raise an error if too many patients for the grid
        if n_patients_to_plot > n_rows * n_cols:
            raise ValueError(
                f"{n_patients_to_plot} patients cannot be plotted in a {n_rows}x{n_cols} grid. Enter a n_patients_to_plot value under {n_rows*n_cols} or use a larger grid."
            )

        if randomize:
            ind_to_plot = rand.sample(
                range(self.model_diag.model.nb_patients), n_patients_to_plot
            )
        else:
            ind_to_plot = list(range(n_patients_to_plot))

        cmap = plt.get_cmap("brg")
        colors = cmap(np.linspace(0, 1, self.model_diag.model.nb_outputs))

        # One plot for each output, containing all individual patients subplots for this output
        for output_index, output_name in enumerate(self.model_diag.model.output_names):
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(facet_width * n_cols, facet_height * n_rows),
                squeeze=False,
            )
            fig.suptitle(f"Output: {output_name}")

            data_output = obs_vs_simulated.loc[
                obs_vs_simulated["output_name"] == output_name
            ]

            for k in range(0, n_patients_to_plot):
                # Change indexing from 1d to 2d
                i = k // n_cols
                j = k % n_cols
                ax = axes[i, j]
                ax.set_xlabel("Time")

                # Filter dataset for current patient
                patient_ind = self.model_diag.model.patients[ind_to_plot[k]]
                patient_data = data_output.loc[data_output["id"] == patient_ind]
                if patient_data.shape[0] == 0:
                    pass

                time_vec = patient_data["time"].to_numpy()
                sorted_indices = np.argsort(time_vec)
                sorted_times = time_vec[sorted_indices]

                obs_vec = patient_data["value"].values[sorted_indices]
                ax.plot(
                    sorted_times,
                    obs_vec,
                    "+",
                    color=colors[output_index],
                    linewidth=2,
                    alpha=0.6,
                )
                pred_vec = patient_data["predicted_value"].values[sorted_indices]
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

    def map_estimates_gof(
        self,
        facet_width: float = 8.0,
        facet_height: float = 8.0,
        tolerance_ribbon: str = "mean",
        tolerance_pct: int = 50,
    ) -> None:

        if self.model_diag.individual_ebe_predictions_df is None:
            self.model_diag.compute_ebe()
        assert self.model_diag.individual_ebe_predictions_df is not None
        obs_vs_simulated = self.model_diag.individual_ebe_predictions_df

        num_plots = self.model_diag.model.nb_outputs
        fig, axes = plt.subplots(
            1, num_plots, figsize=(facet_width * num_plots, facet_height), squeeze=False
        )

        fig.suptitle(f"Observed vs. simulated plot")

        for output_index, output_name in enumerate(self.model_diag.model.output_names):

            ax = axes[0, output_index]
            gof_df = obs_vs_simulated.loc[
                (obs_vs_simulated["output_name"] == output_name)
            ]
            # gof_df = gof_df.loc[gof_df["time"] >= 14400]

            # Compute R² and RMSE
            r2 = r2_score(gof_df["value"], gof_df["predicted_value"])
            rmse = np.sqrt(np.mean((gof_df["value"] - gof_df["predicted_value"]) ** 2))
            metrics_text = f"$R^2 = {r2:.3f}$\n$RMSE= {rmse:.3f}$"

            # Plot (obs,pred) points
            colors = ["red" if t < 14400 else "blue" for t in gof_df["time"]]
            ax.scatter(
                x=gof_df["value"],
                y=gof_df["predicted_value"],
                alpha=0.7,
                s=50,
                c=colors,
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
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="lightgray",
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

    def weighted_residuals(
        self,
        res_type: ResidualType,
        facet_width: int = 10,
        facet_height: int = 10,
    ) -> None:

        match res_type:
            case "pwres":
                if self.model_diag.pwres is None:
                    self.model_diag.compute_pwres()
                assert self.model_diag.pwres is not None
                wres_results = self.model_diag.pwres
                compare_to_pop_pred = True
            case "iwres":
                if self.model_diag.iwres is None:
                    self.model_diag.compute_iwres()
                assert self.model_diag.iwres is not None
                wres_results = self.model_diag.iwres
                compare_to_pop_pred = False
            case "npde":
                if self.model_diag.npde is None:
                    self.model_diag.compute_npde()
                assert self.model_diag.npde is not None
                wres_results = self.model_diag.npde
                compare_to_pop_pred = True
            case _:
                raise ValueError(f"Not implemented residual type: {res_type}")
        if compare_to_pop_pred:
            if self.model_diag.population_parameters_predictions_df is None:
                self.model_diag.zero_random_effect_predictions()
            assert self.model_diag.population_parameters_predictions_df is not None
            comparison_df = self.model_diag.population_parameters_predictions_df
        else:
            if self.model_diag.individual_ebe_predictions_df is None:
                self.model_diag.compute_ebe()
            assert self.model_diag.individual_ebe_predictions_df is not None
            comparison_df = self.model_diag.individual_ebe_predictions_df
        self.residual_values(
            res=wres_results,
            comparison=comparison_df,
            res_type=res_type,
            facet_height=facet_height,
            facet_width=facet_width,
        )

    def residual_values(
        self,
        res: ModelResiduals,
        comparison: pd.DataFrame,
        res_type: str,
        facet_width: int = 10,
        facet_height: int = 10,
    ) -> None:
        all_wres = np.concatenate([p.res for p in res.values()]).flatten()
        all_times = np.concatenate([p.time for p in res.values()]).flatten()

        fig, ax = plt.subplots(2, 2, figsize=(facet_width, facet_height))

        ## Histogram plot
        ax[0, 0].hist(
            all_wres,
            bins=30,
            density=True,
            alpha=0.6,
            color="skyblue",
            edgecolor="black",
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
        ax[0, 1].legend(
            loc="upper right", frameon=True, facecolor="white", framealpha=0.9
        )
        ax[0, 1].set_title(f"{res_type.upper()} vs. Time")

        ## Plot vs. predictions

        # Transform WRES dict into a dataframe
        rows = []
        for patient_id, content in res.items():
            rows.append(
                pd.DataFrame(
                    {"id": patient_id, res_type: content.res, "time": content.time}
                )
            )

        wres_df = pd.concat(rows)

        # Merge WRES with predictions, matching patientID and time
        vs_pred_plot_df = pd.merge(
            wres_df, comparison[["id", "time", "predicted_value"]], on=["id", "time"]
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

        ax[1, 1].set_xlabel("Predictions")

        ax[1, 1].set_ylabel("Weighted Residual (Standard Deviations)")
        ax[1, 1].set_ylim(-1.1 * max(abs(wres_to_plot)), 1.1 * max(abs(wres_to_plot)))
        ax[1, 1].legend(
            loc="upper right", frameon=True, facecolor="white", framealpha=0.9
        )
        ax[1, 1].set_title(f"{res_type.upper()} vs. Predictions")

        plt.tight_layout()

        if not smoke_test:
            plt.show()

        plt.close(fig)

    def map_vs_posterior(
        self,
        n_patients_to_plot: int = 3,
    ):
        if self.model_diag.conditional_distribution_samples is None:
            self.model_diag.sample_conditional_distribution()
        assert self.model_diag.conditional_distribution_samples is not None
        sample_etas = self.model_diag.conditional_distribution_samples.samples

        sample_gaussian = self.model_diag.model.convert_etas_to_gaussian_all_patients(
            sample_etas
        )
        sample_physical = self.model_diag.model.convert_gaussian_to_physical(
            psi=sample_gaussian, log_mi=self.model_diag.model.log_mi
        )

        if n_patients_to_plot > self.model_diag.model.nb_patients:
            n_patients_to_plot = self.model_diag.model.nb_patients

        ind_to_plot = rand.sample(
            range(self.model_diag.model.nb_patients), n_patients_to_plot
        )

        # Get EBE estimates for descriptors
        if self.model_diag.individual_ebe_estimates_tensor is None:
            self.model_diag.compute_ebe()
        ebe_theta = self.model_diag.individual_ebe_estimates_tensor
        assert ebe_theta is not None

        for k in range(n_patients_to_plot):
            patient_samples = (
                sample_physical[:, ind_to_plot[k], :].detach().cpu().numpy()
            )

            # Adapt rows to columns
            n_cols = 3
            n_rows = (self.model_diag.model.nb_pdu + n_cols - 1) // n_cols
            _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = np.atleast_1d(axes).flatten()

            # Plot distribution and MAP for each PDU
            for i in range(len(axes)):
                ax = axes[i]
                if i < self.model_diag.model.nb_pdu:
                    param_data = patient_samples[:, i]

                    if np.unique(param_data).size > 1:
                        kde = stats.gaussian_kde(param_data)
                        x_range = np.linspace(param_data.min(), param_data.max(), 200)
                        ax.plot(
                            x_range,
                            kde(x_range),
                            color="blue",
                            lw=1.5,
                            label="PDF (KDE)",
                        )

                    map_val = ebe_theta[0][ind_to_plot[k]][i]

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

                    ci_low, ci_high = np.percentile(param_data, [2.5, 97.5])
                    ax.axvspan(
                        ci_low,
                        ci_high,
                        color="gray",
                        alpha=0.2,
                        label=f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]",
                    )

                    ax.set_title(
                        f"Patient {ind_to_plot[k]} - {self.model_diag.model.pdu_names[i]}"
                    )
                    ax.legend(fontsize="small")

                else:
                    # Hide plot if empty
                    ax.set_visible(False)

            plt.tight_layout()
            plt.show()
