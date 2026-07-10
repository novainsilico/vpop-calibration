import torch
from typing import Literal
import numpy as np
import pandas as pd
import pandera.pandas as pa

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import (
    calculate_residuals,
    compute_error_variance,
)
from vpop_calibration.config import smoke_test
from vpop_calibration.pynlme.conditional_distribution import (
    ConditionalDistributionSampler,
)

ResidualType = Literal["pwres", "iwres", "npde"]


class WeightedResidualsSchema(pa.DataFrameModel):
    id: str
    time: float
    output_name: str
    residual_value: float = pa.Field(coerce=True)
    residual_type: str


class ModelDiagnostics:
    def __init__(
        self,
        nlme_model: StatisticalModel,
    ):
        self.model = nlme_model
        self.population_parameters_predictions_df: pd.DataFrame | None = None
        self.pwres: pa.typing.DataFrame[WeightedResidualsSchema] | None = None
        self.iwres: pa.typing.DataFrame[WeightedResidualsSchema] | None = None
        self.npde: pa.typing.DataFrame[WeightedResidualsSchema] | None = None
        self.sampler = ConditionalDistributionSampler(nlme_model=self.model)
        self.shrinkage: torch.Tensor | None = None
        self.vpc: pd.DataFrame | None = None

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

        iwres_list = []

        # Separate IWRES per patient in a dict
        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):
            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            this_patient_iwres = iwres_full[this_patient_rows].squeeze().cpu().numpy()
            this_patient_time = self.model.data.individual_observations[
                patient_id
            ].obs_index.time.raw_values
            this_patient_output_name = self.model.data.individual_observations[
                patient_id
            ].obs_index.output_name.raw_values
            this_patient_residuals = pd.DataFrame(
                {
                    "id": patient_id,
                    "time": this_patient_time,
                    "residual_value": this_patient_iwres,
                    "residual_type": "iwres",
                    "output_name": this_patient_output_name,
                }
            )
            iwres_list.append(this_patient_residuals)
        self.iwres = WeightedResidualsSchema.validate(pd.concat(iwres_list))

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
        pwres_list = []

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
            ].obs_index.time.raw_values
            output_names_patient = self.model.data.individual_observations[
                patient_id
            ].obs_index.output_name.raw_values

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
            patient_pwres = pd.DataFrame(
                {
                    "id": patient_id,
                    "time": time_steps_patient,
                    "residual_value": pwres_patient.squeeze(-1).cpu().numpy(),
                    "residual_type": "pwres",
                    "output_name": output_names_patient,
                }
            )
            pwres_list.append(patient_pwres)
        self.pwres = WeightedResidualsSchema.validate(pd.concat(pwres_list))

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

        npde_list = []

        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):

            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            this_patient_data = npde[this_patient_rows]
            this_patient_time = self.model.data.individual_observations[
                patient_id
            ].obs_index.time.raw_values
            this_patient_output_names = self.model.data.individual_observations[
                patient_id
            ].obs_index.output_name.raw_values
            this_patient_npde = pd.DataFrame(
                {
                    "id": patient_id,
                    "residual_value": this_patient_data.squeeze(-1).cpu().numpy(),
                    "residual_type": "npde",
                    "time": this_patient_time,
                    "output_name": this_patient_output_names,
                }
            )
            npde_list.append(this_patient_npde)
        self.npde = WeightedResidualsSchema.validate(pd.concat(npde_list))

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

        ebe_etas = self.sampler.ebe.eta_samples.squeeze(0)

        eta_sd = torch.std(ebe_etas, dim=0, unbiased=True)
        omega_sd = torch.sqrt(torch.diag(self.model.omega_pop))

        shrinkage = 1 - eta_sd / omega_sd

        self.shrinkage = shrinkage

    def compute_vpc(
        self,
        nb_bins: int = 10,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        precision: float = 0.9,
    ) -> None:

        if not hasattr(self.sampler, "samples"):
            self.sampler.run_sampler()

        df = self.sampler.total_samples_predictions_df
        all_vpc_records = []
        quantiles_arr = np.asarray(quantiles)

        for output_name in self.model.output_names:

            df_output = df[df["output_name"] == output_name]
            bin_labels, bin_edges = pd.cut(
                df_output["time"].astype("float"),
                bins=nb_bins,
                include_lowest=True,
                labels=False,
                retbins=True,
            )
            df_output.insert(1, "bin", bin_labels)

            default_centers = pd.Series(
                0.5 * (bin_edges[:-1] + bin_edges[1:]), index=range(nb_bins)
            )
            bin_centers = (
                df_output.loc[df_output["batch_id"] == 0]
                .groupby("bin")["time"]
                .median()
                .reindex(range(nb_bins))
                .fillna(default_centers)
            )

            q_obs = (
                df_output.loc[df_output["batch_id"] == 0]
                .groupby("bin")["value"]
                .quantile(quantiles_arr)
                .rename("q_obs")
            )
            q_obs.index.names = ["bin", "quantile"]

            pred_q_batch = df_output.groupby(["bin", "batch_id"])[
                "predicted_value"
            ].quantile(quantiles_arr)
            pred_q_batch.index.names = ["bin", "batch_id", "quantile"]
            pred_median = (
                pred_q_batch.groupby(["bin", "quantile"])
                .quantile(0.5)
                .rename("pred_median")
            )
            pred_lower = (
                pred_q_batch.groupby(["bin", "quantile"])
                .quantile(1 - precision)
                .rename("pred_lower")
            )
            pred_upper = (
                pred_q_batch.groupby(["bin", "quantile"])
                .quantile(precision)
                .rename("pred_upper")
            )

            df_q = pd.concat(
                [q_obs, pred_median, pred_lower, pred_upper], axis=1
            ).reset_index()
            df_q["bin_center"] = df_q["bin"].map(bin_centers)
            df_q["output_name"] = output_name

            all_vpc_records.append(df_q)

        vpc_df = pd.concat(all_vpc_records, ignore_index=True)
        self.vpc = vpc_df
