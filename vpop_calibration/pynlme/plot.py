import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score
import random as rand
import scipy.stats as stats
import pandera.pandas as pa


from vpop_calibration.pynlme.diagnostics import (
    ModelDiagnostics,
    WeightedResidualsSchema,
    ResidualType,
)
from vpop_calibration.pynlme.initial_estimates import theoretical_pdf
from vpop_calibration.pynlme.params import Constraint
from vpop_calibration.pynlme.utils import inverse_transform_param
from vpop_calibration.model.gp import GP
from vpop_calibration.structural_model.gp import StructuralGp
from vpop_calibration.config import smoke_test
from vpop_calibration.utils import time_scale_and_label


def _population_marginal_log10(
    means: np.ndarray,
    std: float,
    const: Constraint,
    data_range: tuple[float, float],
    nb_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Density of log10(phi) under the population model, for one PDU.

    With covariates, each patient has its own mean in the gaussian space, so the population marginal
    is the mixture of the individual distributions (averaged over the patients of the dataset).

    Args:
        means: Population mean of the gaussian parameter, per patient. Size (nb_patients,)
        std: Standard deviation of the random effect (sqrt of the Omega diagonal term)
        const: Constraint (transform, shift, scale) of the PDU
        data_range: (min, max) of the plotted samples, in log10 space, to be included in the grid

    Returns:
        The log10 grid and the density evaluated on it
    """
    # Cover +/- 3.5 SD of the population distribution, as well as the plotted samples
    psi_bounds = np.array([means.min() - 3.5 * std, means.max() + 3.5 * std])
    phys_bounds = inverse_transform_param(psi_bounds, const)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_bounds = np.log10(phys_bounds)
    low = np.nanmin([log10_bounds[0], data_range[0]])
    high = np.nanmax([log10_bounds[1], data_range[1]])
    log10_grid = np.linspace(low, high, nb_points)
    x = 10**log10_grid

    unique_means, counts = np.unique(means, return_counts=True)
    density_x = sum(
        count * theoretical_pdf(x, mu=mu, prior_std=std, const=const)
        for mu, count in zip(unique_means, counts)
    ) / len(means)
    # Change of variable x -> log10(x): dx/du = x * ln(10)
    density_log10 = density_x * x * np.log(10)
    return log10_grid, density_log10


def _log10_to_gaussian(
    log10_grid: np.ndarray, const: Constraint
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map log10(phi) values to the gaussian space psi.

    Returns:
        psi, the Jacobian d(psi)/d(log10 phi), and a validity mask (phi inside the support of the transform)
    """
    phi = 10**log10_grid
    if const.transform == "log":
        valid = phi > const.shift
        shifted = np.where(valid, phi - const.shift, 1.0)
        psi = np.log(shifted)
        dpsi_dphi = 1.0 / shifted
    elif const.transform == "logit":
        s = (phi - const.shift) / const.scale
        valid = (s > 0) & (s < 1)
        s = np.where(valid, s, 0.5)
        psi = np.log(s / (1 - s))
        dpsi_dphi = 1.0 / (const.scale * s * (1 - s))
    else:
        raise NotImplementedError(f"Unsupported transform: {const.transform}")
    jacobian = dpsi_dphi * phi * np.log(10)
    return psi, jacobian, valid


def _population_joint_log10(
    means: np.ndarray,
    cov: np.ndarray,
    consts: tuple[Constraint, Constraint],
    grids: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Joint density of (log10(phi_x), log10(phi_y)) under the population model, for a pair of PDUs.

    In the gaussian space, (psi_x, psi_y) ~ N(X_i @ beta, Omega_xy). For log-transformed parameters without
    lower bound, this is the same bivariate normal (rescaled by 1/ln(10)) in log10 space. Other transforms are
    handled through the change of variables. With covariates, the density is averaged over patients.

    Args:
        means: Population means in the gaussian space, per patient. Size (nb_patients, 2)
        cov: Covariance of the two random effects. Size (2, 2)
        consts: Constraints of the x and y parameters
        grids: 1D log10 grids for x and y

    Returns:
        The density on the meshgrid. Size (len(grids[1]), len(grids[0]))
    """
    psi_x, jac_x, valid_x = _log10_to_gaussian(grids[0], consts[0])
    psi_y, jac_y, valid_y = _log10_to_gaussian(grids[1], consts[1])
    psi_xx, psi_yy = np.meshgrid(psi_x, psi_y)
    points = np.stack([psi_xx, psi_yy], axis=-1)

    unique_means, counts = np.unique(means, axis=0, return_counts=True)
    density_psi = sum(
        count * stats.multivariate_normal.pdf(points, mean=mu, cov=cov)
        for mu, count in zip(unique_means, counts)
    ) / len(means)
    density = density_psi * np.outer(jac_y, jac_x)
    return np.where(np.outer(valid_y, valid_x), density, 0.0)


def _highest_density_levels(density: np.ndarray, masses: list[float]) -> np.ndarray:
    """Density levels whose contours enclose the given probability masses (on a uniform grid)."""
    sorted_density = np.sort(density.ravel())[::-1]
    cumulative_mass = np.cumsum(sorted_density) / sorted_density.sum()
    levels = [
        sorted_density[
            min(np.searchsorted(cumulative_mass, m), len(sorted_density) - 1)
        ]
        for m in masses
    ]
    return np.unique(levels)


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

        if not hasattr(self.model_diag.sampler, "map"):
            self.model_diag.sample_conditional_distribution()
        gp_model: GP = gp_model_struct.gp_model
        train_data = gp_model.data.full_df_raw[pdus].drop_duplicates()

        map_data = self.model_diag.sampler.map_parameters_df
        patients = self.model_diag.model.patients

        n_plots = len(pdus)
        n_cols = n_columns
        n_rows = int(np.ceil(n_plots / n_cols))

        fig1, axes1 = plt.subplots(
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

        fig2, axes2 = plt.subplots(
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
        plt.close(fig1)
        plt.close(fig2)
        return diagnostics, recommended_ranges

    def map_estimates(
        self,
        facet_width: float = 5.0,
        facet_height: float = 4.0,
        time_unit: str | None = None,
    ) -> None:
        if not hasattr(self.model_diag.sampler, "map"):
            self.model_diag.sample_conditional_distribution()
        obs_vs_simulated = self.model_diag.sampler.map_predictions_df

        n_cols = self.model_diag.model.input_params.nb_continuous_outputs
        n_rows = self.model_diag.model.nb_protocols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(facet_width * n_cols, facet_height * n_rows),
            squeeze=False,
        )
        xlabel, time_scale = time_scale_and_label(time_unit)
        cmap = plt.get_cmap("Spectral")
        colors = cmap(np.linspace(0, 1, self.model_diag.model.nb_patients))
        for output_index, output_name in enumerate(
            self.model_diag.model.input_params.continuous_output_names
        ):
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
                ax.set_xlabel(xlabel)
                patients_protocol = data_loop["id"].drop_duplicates().to_list()
                for patient_ind in patients_protocol:
                    patient_num = self.model_diag.model.patients.index(patient_ind)
                    patient_data = data_loop.loc[data_loop["id"] == patient_ind]
                    time_vec = patient_data["time"].values / time_scale
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
        plt.close(fig)

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

        if not hasattr(self.model_diag.sampler, "map"):
            self.model_diag.sample_conditional_distribution()
        # Filter datasets for the selected patient
        obs_vs_simulated = self.model_diag.sampler.map_predictions_df

        patient_ind = self.model_diag.model.patients[patient_num]
        patient_data = obs_vs_simulated.loc[obs_vs_simulated["id"] == patient_ind]

        # Print patient parameters if verbose selected
        if verbose:
            patient_params = self.model_diag.sampler.map_parameters_df
            print(patient_params.loc[patient_params["id"] == patient_ind])

        # Initialize subplots
        n_cols = self.model_diag.model.input_params.nb_continuous_outputs
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
        colors = cmap(
            np.linspace(0, 1, self.model_diag.model.input_params.nb_continuous_outputs)
        )

        for output_index, output_name in enumerate(
            self.model_diag.model.input_params.continuous_output_names
        ):
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

        if not smoke_test:
            plt.show()
            plt.tight_layout()

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

        if not hasattr(self.model_diag.sampler, "map"):
            self.model_diag.sample_conditional_distribution()
        obs_vs_simulated = self.model_diag.sampler.map_predictions_df

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
                f"{n_patients_to_plot} patients cannot be plotted in a {n_rows}x{n_cols} grid. Enter a n_patients_to_plot value under {n_rows * n_cols} or use a larger grid."
            )

        if randomize:
            ind_to_plot = rand.sample(
                range(self.model_diag.model.nb_patients), n_patients_to_plot
            )
        else:
            ind_to_plot = list(range(n_patients_to_plot))

        cmap = plt.get_cmap("brg")
        colors = cmap(
            np.linspace(0, 1, self.model_diag.model.input_params.nb_continuous_outputs)
        )

        # One plot for each output, containing all individual patients subplots for this output
        for output_index, output_name in enumerate(
            self.model_diag.model.input_params.continuous_output_names
        ):
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
            if not smoke_test:
                plt.show()
                plt.tight_layout()

            plt.close(fig)

    def map_estimates_gof(
        self,
        facet_width: float = 8.0,
        facet_height: float = 8.0,
        tolerance_ribbon: str = "mean",
        tolerance_pct: int = 50,
    ) -> None:

        if not hasattr(self.model_diag.sampler, "map"):
            self.model_diag.sample_conditional_distribution()
        obs_vs_simulated = self.model_diag.sampler.map_predictions_df

        num_plots = self.model_diag.model.input_params.nb_continuous_outputs
        fig, axes = plt.subplots(
            1, num_plots, figsize=(facet_width * num_plots, facet_height), squeeze=False
        )

        fig.suptitle("Observed vs. simulated plot")

        for output_index, output_name in enumerate(
            self.model_diag.model.input_params.continuous_output_names
        ):
            ax = axes[0, output_index]
            gof_df = obs_vs_simulated.loc[
                (obs_vs_simulated["output_name"] == output_name)
            ]

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
        time_unit: str | None = None,
    ) -> None:

        match res_type:
            case "pwres" | "npde":
                if self.model_diag.population_residuals is None:
                    print(
                        "No population residuals in cache. Calling compute_population_residuals() with default "
                        + "nb_samples=100. Call compute_population_residuals() directly to use another number of samples"
                    )
                    self.model_diag.compute_population_residuals()
                assert self.model_diag.population_residuals is not None
                wres_results = self.model_diag.population_residuals.loc[
                    self.model_diag.population_residuals["residual_type"] == res_type
                ]
                compare_to_pop_pred = True
            case "iwres":
                if self.model_diag.iwres is None:
                    self.model_diag.compute_iwres()
                assert self.model_diag.iwres is not None
                wres_results = self.model_diag.iwres
                compare_to_pop_pred = False
            case _:
                raise ValueError(f"Not implemented residual type: {res_type}")
        if compare_to_pop_pred:
            if self.model_diag.population_parameters_predictions_df is None:
                self.model_diag.zero_random_effect_predictions()
            assert self.model_diag.population_parameters_predictions_df is not None
            comparison_df = self.model_diag.population_parameters_predictions_df
        else:
            if not hasattr(self.model_diag.sampler, "map"):
                self.model_diag.sample_conditional_distribution()
            comparison_df = self.model_diag.sampler.map_predictions_df
        self.residual_values(
            res_df=wres_results,
            comparison=comparison_df,
            res_type=res_type,
            facet_height=facet_height,
            facet_width=facet_width,
            time_unit=time_unit,
        )

    def residual_values(
        self,
        res_df: pa.typing.DataFrame[WeightedResidualsSchema],
        comparison: pd.DataFrame,
        res_type: str,
        facet_width: int = 10,
        facet_height: int = 10,
        time_unit: str | None = None,
    ) -> None:

        fig, ax = plt.subplots(2, 2, figsize=(facet_width, facet_height))
        xlabel, time_scale = time_scale_and_label(time_unit)
        ## Histogram plot
        ax[0, 0].hist(
            res_df["residual_value"],
            bins=30,
            density=True,
            alpha=0.6,
            color="skyblue",
            edgecolor="black",
        )
        mu, std = 0, 1
        x = np.linspace(
            min(res_df["residual_value"]), max(res_df["residual_value"]), 100
        )
        p = stats.norm.pdf(x, mu, std)
        ax[0, 0].plot(x, p, "r", linewidth=2, label=r"$\mathcal{N}(0,1)$")
        ax[0, 0].set_title(f"{res_type.upper()} distribution")
        ax[0, 0].set_xlabel("Residual values")
        ax[0, 0].set_ylabel("Density")
        ax[0, 0].legend()

        ## Q-Q plot
        stats.probplot(res_df["residual_value"], dist="norm", plot=ax[1, 0])
        ax[1, 0].set_title(f"{res_type.upper()} Q-Q Plot")

        ## Plot vs. time
        ax[0, 1].grid(True, linestyle="--", alpha=0.6, which="both")
        ax[0, 1].set_facecolor("#fdfdfd")
        ax[0, 1].scatter(
            res_df["time"] / time_scale,
            res_df["residual_value"],
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
        ax[0, 1].set_xlabel(xlabel, fontsize=12)
        ax[0, 1].set_ylabel("Weighted Residual (Standard Deviations)", fontsize=12)
        ax[0, 1].set_ylim(
            -1.1 * max(abs(res_df["residual_value"])),
            1.1 * max(abs(res_df["residual_value"])),
        )
        ax[0, 1].legend(
            loc="upper right", frameon=True, facecolor="white", framealpha=0.9
        )
        ax[0, 1].set_title(f"{res_type.upper()} vs. Time")

        ## Plot vs. predictions

        # Merge WRES with predictions, matching patientID, output and time
        vs_pred_plot_df = pd.merge(
            res_df,
            comparison[["id", "output_name", "time", "predicted_value"]],
            on=["id", "output_name", "time"],
        )

        wres_to_plot = vs_pred_plot_df["residual_value"]
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
        if not hasattr(self.model_diag.sampler, "map"):
            self.model_diag.sample_conditional_distribution()
        sample_physical = self.model_diag.sampler.total_samples.physical_params_samples
        if n_patients_to_plot > self.model_diag.model.nb_patients:
            n_patients_to_plot = self.model_diag.model.nb_patients

        ind_to_plot = rand.sample(
            range(self.model_diag.model.nb_patients), n_patients_to_plot
        )

        # Get MAP estimates for descriptors
        map_theta = self.model_diag.model.convert_physical_to_thetas_all_patients(
            self.model_diag.sampler.map.physical_params_samples
        )
        assert map_theta is not None

        for k in range(n_patients_to_plot):
            patient_samples = (
                sample_physical[:, ind_to_plot[k], :].detach().cpu().numpy()
            )

            # Adapt rows to columns
            n_cols = 3
            n_rows = (self.model_diag.model.nb_pdu + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
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

                    map_val = map_theta[0][ind_to_plot[k]][i]

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

            if not smoke_test:
                plt.tight_layout()
                plt.show()
            plt.close(fig)

    def vpc(
        self,
        facet_width: int = 10,
        facet_height: int = 6,
        time_unit: str | None = None,
    ):

        if self.model_diag.vpc is None:
            print(
                "No VPC in cache. Calling compute_vpc() with default quantiles, precision and nb_bins. "
                + "Call compute_vpc() directly to use other settings"
            )
            self.model_diag.compute_vpc()

        vpc_df = self.model_diag.vpc
        assert vpc_df is not None

        median_color = "#D389C9"
        outer_color = "#5b9bd5"
        outside_color = "#e60000"

        output_names = vpc_df["output_name"].unique()
        nb_outputs = len(output_names)
        fig, axes = plt.subplots(
            1, nb_outputs, figsize=(facet_width, facet_height), squeeze=False
        )

        xlabel, time_scale = time_scale_and_label(time_unit)

        for i, output_name in enumerate(output_names):
            ax = axes[0, i]
            df_output = vpc_df[vpc_df["output_name"] == output_name]

            ax.grid(True, linestyle="--", alpha=0.3, which="both")
            ax.set_facecolor("#fdfdfd")

            for q, df_q in df_output.groupby("quantile"):
                df_q = df_q.sort_values("bin_center")

                x = df_q["bin_center"].to_numpy() / time_scale
                q_obs = df_q["q_obs"].to_numpy()
                pred_median = df_q["pred_median"].to_numpy()
                pred_lower = df_q["pred_lower"].to_numpy()
                pred_upper = df_q["pred_upper"].to_numpy()

                is_median = np.isclose(q, 0.5)
                color = median_color if is_median else outer_color

                ax.plot(x, q_obs, color="#033a0d", linestyle="--", linewidth=1)
                ax.plot(x, pred_median, color=color, linestyle="-", linewidth=1)

                ax.fill_between(x, pred_lower, pred_upper, color=color, alpha=0.15)

                above = q_obs > pred_upper
                below = q_obs < pred_lower
                ax.fill_between(
                    x,
                    pred_upper,
                    q_obs,
                    where=above,
                    color=outside_color,
                    interpolate=True,
                    alpha=0.5,
                )
                ax.fill_between(
                    x,
                    pred_lower,
                    q_obs,
                    where=below,
                    color=outside_color,
                    interpolate=True,
                    alpha=0.5,
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Observation")
            ax.set_title(f"VPC: {output_name}")
            if i == 0:
                legend_handles = [
                    Line2D([0], [0], color=median_color, lw=1.5, label="Median"),
                    Line2D(
                        [0], [0], color=outer_color, lw=1.5, label="Other quantiles"
                    ),
                    Line2D([0], [0], color="gray", lw=1.2, ls="--", label="Empirical"),
                    Line2D([0], [0], color="gray", lw=1.2, ls="-", label="Prediction"),
                    Patch(
                        facecolor=median_color, alpha=0.15, label="Prediction interval"
                    ),
                    Patch(
                        facecolor=outside_color,
                        alpha=0.5,
                        label="Observed quantile out of CI",
                    ),
                ]
                ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        if not smoke_test:
            plt.tight_layout()
            plt.show()
        plt.close(fig)

    def conditional_codistributions(
        self,
        scaling_indiv_plot: float = 3.0,
        scaling_2by2_plot: float = 2.5,
        n_columns: int = 3,
        contour_masses: list[float] = [0.5, 0.9, 0.99],
    ) -> None:
        def _value_formatter_log(v, pos):
            return f"{10**v:.3g}"

        format_log_values = ticker.FuncFormatter(_value_formatter_log)
        pdus = self.model_diag.model.pdu_names
        sampler = self.model_diag.sampler

        if not hasattr(sampler, "map"):
            self.model_diag.sample_conditional_distribution()

        map_data = self.model_diag.sampler.map_parameters_df
        cond_data = self.model_diag.sampler.total_samples_parameters_df

        # Population model: psi_i = X_i @ beta + eta_i, eta_i ~ N(0, Omega)
        model = self.model_diag.model
        pop_means = (
            (model.full_design_matrix @ model.population_betas).detach().cpu().numpy()
        )  # (nb_patients, nb_pdu)
        pop_cov = model.omega_pop.detach().cpu().numpy()
        pop_stds = np.sqrt(np.diag(pop_cov))
        pdu_constraints = [model.input_params.pdu[p].constraint for p in pdus]
        # Plotting range of each parameter (log10 space), reused for the 2D contours
        log10_ranges = {}

        n_plots = len(pdus)
        n_cols = n_columns
        n_rows = int(np.ceil(n_plots / n_cols))

        fig1, axes1 = plt.subplots(
            n_rows,
            n_cols,
            squeeze=False,
            figsize=[scaling_indiv_plot * n_cols, scaling_indiv_plot * n_rows],
        )

        for k, param in enumerate(pdus):
            i, j = k // n_cols, k % n_cols
            cond_samples = np.log10(cond_data[param])
            map_samples = np.log10(map_data[param])

            ax = axes1[i, j]
            ax.hist(
                [cond_samples, map_samples],
                density=True,
                label=["Conditional samples", "MAP"],
                zorder=2,
            )
            log10_grid, pop_density = _population_marginal_log10(
                means=pop_means[:, k],
                std=pop_stds[k],
                const=pdu_constraints[k],
                data_range=(
                    min(cond_samples.min(), map_samples.min()),
                    max(cond_samples.max(), map_samples.max()),
                ),
            )
            log10_ranges[param] = (log10_grid[0], log10_grid[-1])
            ax.plot(
                log10_grid,
                pop_density,
                color="black",
                linewidth=1.5,
                label="Population model ~ N(β, Ω)",
                zorder=1,
            )
            ax.set_title(f"{param}")
            ax.xaxis.set_major_formatter(format_log_values)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

        handles, labels = axes1[0, 0].get_legend_handles_labels()
        fig1.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        # Keep ~0.4 inch at the top of the figure for the legend
        fig1.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig1.get_figheight()))

        fig2, axes2 = plt.subplots(
            n_plots,
            n_plots,
            squeeze=False,
            figsize=[scaling_2by2_plot * n_plots, scaling_2by2_plot * n_plots],
            sharex="col",
            sharey="row",
        )

        # Plot in log10 space on linear axes, so that a few readable ticks can be placed
        # even when a parameter spans less than a decade
        for k1, param1 in enumerate(pdus):
            cond_samples_1 = np.log10(cond_data[param1])
            map_samples_1 = np.log10(map_data[param1])
            for k2, param2 in enumerate(pdus):
                cond_samples_2 = np.log10(cond_data[param2])
                map_samples_2 = np.log10(map_data[param2])
                ax = axes2[k1, k2]
                if k1 != k2:
                    # param 1 is the row -> y axis
                    # param 2 is the column -> x axis
                    ax.scatter(
                        cond_samples_2,
                        cond_samples_1,
                        alpha=0.5,
                        s=1.0,
                        label="Conditional samples",
                    )
                    ax.scatter(map_samples_2, map_samples_1, s=5, label="MAP")
                    grids = (
                        np.linspace(*log10_ranges[param2], 100),
                        np.linspace(*log10_ranges[param1], 100),
                    )
                    pop_density_2d = _population_joint_log10(
                        means=pop_means[:, [k2, k1]],
                        cov=pop_cov[np.ix_([k2, k1], [k2, k1])],
                        consts=(pdu_constraints[k2], pdu_constraints[k1]),
                        grids=grids,
                    )
                    ax.contour(
                        *grids,
                        pop_density_2d,
                        levels=_highest_density_levels(pop_density_2d, contour_masses),
                        colors="black",
                        linewidths=0.8,
                    )
                if k2 == 0:
                    ax.set_ylabel(param1)
                if k1 == len(pdus) - 1:
                    ax.set_xlabel(param2)

        for ax in fig2.axes:
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(format_log_values)
                axis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.tick_params(axis="x", labelrotation=45)

        # Wrap the legend on two rows when the figure is too narrow for a single one
        legend_ncol = 3
        legend_height = 0.4
        if n_plots > 1:
            # The diagonal is empty, the first off-diagonal subplot carries the labels
            handles, labels = axes2[0, 1].get_legend_handles_labels()
            masses_str = "/".join(f"{m:.0%}" for m in contour_masses)
            handles.append(Line2D([], [], color="black", linewidth=0.8))
            labels.append(f"Population model ({masses_str})")
            fig2.legend(
                handles,
                labels,
                loc="upper center",
                ncol=legend_ncol,
                frameon=False,
                markerscale=3,
            )
        fig2.tight_layout(rect=(0, 0, 1, 1 - legend_height / fig2.get_figheight()))

        if not smoke_test:
            plt.show()

        plt.close(fig1)
        plt.close(fig2)
