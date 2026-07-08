import torch
from typing import NamedTuple, Literal
import numpy as np
import pandas as pd

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import (
    calculate_residuals,
    compute_error_variance,
)
from vpop_calibration.config import smoke_test
from vpop_calibration.pynlme.conditional_distribution import (
    ConditionalDistributionSampler,
)


class PatientResiduals(NamedTuple):
    time: np.ndarray
    res: np.ndarray


ResidualType = Literal["pwres", "iwres", "npde"]
ModelResiduals = dict[str, PatientResiduals]


class VPCResult(NamedTuple):
    bins: np.ndarray
    bin_centers: np.ndarray
    obs_quantiles: Dict[float, np.ndarray]
    pred_quantiles_ci: Dict[float, np.ndarray]


class ModelDiagnostics:
    def __init__(
        self,
        nlme_model: StatisticalModel,
    ):
        self.model = nlme_model
        self.population_parameters_predictions_df: pd.DataFrame | None = None
        self.pwres: ModelResiduals | None = None
        self.iwres: ModelResiduals | None = None
        self.npde: ModelResiduals | None = None
        self.sampler = ConditionalDistributionSampler(nlme_model=self.model)
        self.shrinkage: torch.Tensor | None = None
        self.vpc: VPCResult | None = None

    def sample_conditional_distribution(
        self,
        nb_samples: int = 100,
    ) -> None:
        self.sampler.run_sampler(nb_samples=nb_samples)

    def compute_iwres(self) -> None:
        """Compute Individual Weighted Residuals (IWRES), following the formula :

        IWRES_(ij) = ( y_ij - f(t_ij, psi_i) ) / g(t_ij, psi_i)
        where psi_i are the patients empirical bayesian estimators.

        Returns:
            dict: IWRES with patientId as key, with IWRES and timesteps for each patient
        """
        if not hasattr(self.sampler, "ebe"):
            print("No EBEs available, computing them...")
            self.sample_conditional_distribution()
        assert hasattr(self.sampler, "ebe")

        ebe_physical_params = self.sampler.ebe.physical_params_samples
        assert ebe_physical_params.shape == (
            1,
            self.model.nb_patients,
            self.model.nb_pdu + self.model.nb_mi,
        ), f"{ebe_physical_params.shape}"

        # Assemble the thetas by adding the PDKs
        theta = self.model.convert_physical_to_thetas_all_patients(
            physical_params=ebe_physical_params
        )
        model_inputs = self.model.convert_thetas_to_model_parameters_all_patients(
            theta=theta
        )
        simulated_tensor, _ = self.model.predict_all_patients(inputs=model_inputs)

        # Compute residuals and variance
        residuals = calculate_residuals(
            observed_data=self.model.data.full_obs,
            predictions=simulated_tensor,
            error_model_selector=self.model.error_model_selector,
        )

        variance = compute_error_variance(
            observations=self.model.data.full_obs,
            predictions=simulated_tensor,
            error_model_selector=self.model.error_model_selector,
            sigma=self.model.residual_var,
        )

        iwres_full = residuals / torch.sqrt(variance)
        iwres_full.squeeze_(0)

        self.iwres = {}

        # Separate IWRES per patient in a dict
        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):
            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            this_patient_iwres = iwres_full[this_patient_rows]
            this_patient_time = self.model.data.individual_observations[
                patient_id
            ].obs_index.time.raw_values.to_numpy()
            this_patient_residuals = PatientResiduals(
                time=this_patient_time,
                res=this_patient_iwres.squeeze().cpu().numpy(),
            )
            self.iwres.update({patient_id: this_patient_residuals})

    def compute_pwres(self, nb_samples: int = 100) -> None:
        """Compute Population Weighted Residuals (PWRES), following the formula :

        PWRES_i = V_i^(-1/2) (y_i - E(f(t_ij, psi_i))

        Returns:
            dict: PWRES with patientId as key, with PWRES and timesteps for each patient
        """

        if smoke_test:
            nb_samples = 3
        # Sample new etas, in order to approximate mean E(y_i) and variance V_i
        mc_etas = self.model.sample_etas(nb_samples)
        mc_gaussian = self.model.convert_etas_to_gaussian_all_patients(mc_etas)
        mc_physical = self.model.convert_gaussian_to_physical(
            psi=mc_gaussian, log_mi=self.model.log_mi
        )
        mc_thetas = self.model.convert_physical_to_thetas_all_patients(
            physical_params=mc_physical
        )
        inputs = self.model.convert_thetas_to_model_parameters_all_patients(
            theta=mc_thetas
        )
        # Simulate model
        simulated_tensor, _ = self.model.predict_all_patients(inputs=inputs)

        # Compute PWRES per patient
        self.pwres = {}

        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):
            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            this_patient_data = simulated_tensor[:, this_patient_rows]

            # mean_patient shape: nb_samples * n_obs_patient -> n_obs_patient
            mean_patient = this_patient_data.mean(dim=0)

            # obs_patient shape: n_obs_patient
            obs_patient = self.model.data.individual_observations[patient_id].obs_values
            time_steps_patient = self.model.data.individual_observations[
                patient_id
            ].obs_index.time.raw_values.to_numpy()

            # variance_patient shape: n_obs_patient * n_obs_patient
            variance_patient = torch.cov(obs_patient.T)

            # Transform residual into a column
            residual = (obs_patient - mean_patient).unsqueeze(-1)

            # Compute V^-1/2 with Cholesky factorization, adding a jitter for stability purposes
            if variance_patient.dim() > 1:
                jitter = torch.eye(variance_patient.size(0)) * 1e-6
                L = torch.linalg.cholesky(variance_patient + jitter)
                pwres_patient = torch.linalg.solve_triangular(L, residual, upper=False)
            else:
                jitter = 1e-6
                pwres_patient = variance_patient ** (-1 / 2) * residual

            # Compute patient PWRES and add them to dictionnary
            patient_pwres = PatientResiduals(
                time=time_steps_patient,
                res=pwres_patient.squeeze(-1).cpu().numpy(),
            )
            self.pwres.update({patient_id: patient_pwres})

    def compute_npde(self, nb_samples: int = 100) -> None:
        if smoke_test:
            nb_samples = 3

        # Sample new etas
        mc_etas = self.model.sample_etas(nb_samples)
        mc_gaussian = self.model.convert_etas_to_gaussian_all_patients(mc_etas)
        mc_physical = self.model.convert_gaussian_to_physical(
            psi=mc_gaussian, log_mi=self.model.log_mi
        )
        mc_thetas = self.model.convert_physical_to_thetas_all_patients(mc_physical)
        inputs = self.model.convert_thetas_to_model_parameters_all_patients(mc_thetas)

        # Simulate outputs
        simulated_tensor, _ = self.model.predict_all_patients(inputs)

        # Expand observation tensor to match simulated tensor
        observed_tensor = self.model.data.full_obs.obs_values.expand(nb_samples, -1)

        # Compute indicator function in NPDE formula
        mc_F = simulated_tensor <= observed_tensor
        mc_F = mc_F.to(torch.float)

        # Average on MC samples, avoiding 0 and 1 values
        mean_F = mc_F.mean(dim=0)
        eps = 1.0 / simulated_tensor.shape[0]
        mean_F_clamped = torch.clamp(mean_F, min=eps, max=1.0 - eps)

        # Apply normal inverse CDF to compare NPDE with N(0,1)
        normal_dist = torch.distributions.Normal(0, 1)
        npde = normal_dist.icdf(mean_F_clamped)

        self.npde = {}

        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):

            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            this_patient_data = npde[this_patient_rows]
            this_patient_time = self.model.data.individual_observations[
                patient_id
            ].obs_index.time.raw_values.to_numpy()
            this_patient_npde = PatientResiduals(
                res=this_patient_data.squeeze(-1).cpu().numpy(),
                time=this_patient_time,
            )
            self.npde.update({patient_id: this_patient_npde})

    def zero_random_effect_predictions(self) -> None:
        eta = torch.zeros((1, self.model.nb_patients, self.model.nb_pdu))
        gaussian = self.model.convert_etas_to_gaussian_all_patients(eta)
        physical = self.model.convert_gaussian_to_physical(
            psi=gaussian, log_mi=self.model.log_mi
        )
        theta = self.model.convert_physical_to_thetas_all_patients(
            physical_params=physical
        )
        inputs = self.model.convert_thetas_to_model_parameters_all_patients(theta)
        pred, _ = self.model.predict_all_patients(inputs)
        pred_df = self.model.data.full_obs.to_pandas(prediction=pred)
        self.population_parameters_predictions_df = pred_df

    def compute_shrinkage(self, nb_samples: int = 50) -> None:

        if not hasattr(self.sampler, "ebe"):
            self.sampler.run_sampler(nb_samples=nb_samples)
        assert self.sampler.ebe is not None

        ebe_etas = self.sampler.ebe.eta_samples

        eta_sd = torch.std(ebe_etas, dim=0, unbiased=True)
        omega_sd = torch.sqrt(torch.diag(self.model.omega_pop))

        shrinkage = 1 - eta_sd / omega_sd

        self.shrinkage = shrinkage

    def compute_vpc(
        self,
        output_name: str,
        nb_samples: int = 100,
        nb_bins: int = 30,
        quantiles: list[float] = [0.05, 0.5, 0.95],
    ) -> None:

        if self.conditional_distribution_samples is None:
            self.sample_conditional_distribution(nb_samples=nb_samples)

        assert self.conditional_distribution_samples is not None

        etas_samples = self.conditional_distribution_samples.samples
        gaussian_params = self.model.convert_etas_to_gaussian_all_patients(etas_samples)
        physical_params = self.model.convert_gaussian_to_physical(
            gaussian_params, self.model.log_mi
        )
        theta = self.model.convert_physical_to_thetas_all_patients(physical_params)
        model_inputs = self.model.convert_thetas_to_model_parameters_all_patients(theta)
        all_pred, _ = self.model.predict_all_patients(model_inputs)

        # only keep output_name predicted values
        df = self.model.data.full_obs.to_pandas(prediction=all_pred)
        df_output = df[df["output_name"] == output_name]

        obs_times = df_output["time"].values
        obs_values = df_output["value"].values

        pred = np.vstack(df_output["predicted_value"].values).T

        bins = np.unique(np.quantile(obs_times, np.linspace(0, 1, nb_bins + 1)))
        actual_nb_bins = len(bins) - 1

        obs_bin = np.digitize(obs_times, bins) - 1
        obs_bin = np.clip(obs_bin, 0, actual_nb_bins - 1)

        bin_centers = []
        obs_quantiles_lists = {q: [] for q in quantiles}
        pred_quantiles_ci_lists = {q: [] for q in quantiles}

        for b in range(actual_nb_bins):
            in_bin = obs_bin == b

            vals = obs_values[in_bin]
            b_times = obs_times[in_bin]

            if len(vals) > 0:
                bin_centers.append(np.median(b_times))
                sim_vals_in_bin = pred[:, in_bin]

                for q in quantiles:
                    obs_quantiles_lists[q].append(np.quantile(vals, q))

                    q_per_sim = np.quantile(sim_vals_in_bin, q, axis=1)
                    pred_quantiles_ci_lists[q].append(
                        np.percentile(q_per_sim, [2.5, 97.5])
                    )

            else:
                bin_centers.append(0.5 * (bins[b] + bins[b + 1]))
                for q in quantiles:
                    obs_quantiles_lists[q].append(np.nan)
                    pred_quantiles_ci_lists[q].append((np.nan, np.nan))

        final_obs_quantiles = {
            q: np.array(lst) for q, lst in obs_quantiles_lists.items()
        }
        final_pred_quantiles_ci = {
            q: np.array(lst) for q, lst in pred_quantiles_ci_lists.items()
        }

        self.vpc = VPCResult(
            bins=bins,
            bin_centers=np.array(bin_centers),
            obs_quantiles=final_obs_quantiles,
            pred_quantiles_ci=final_pred_quantiles_ci,
        )
