import torch
from typing import Literal, Any
import numpy as np
import pandas as pd
import pandera.pandas as pa
from vpop_calibration.compatibility import tqdm
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import (
    add_predictive_error,
    calculate_residuals,
    compute_error_variance,
)
from vpop_calibration.config import smoke_test
from vpop_calibration.pynlme.conditional_distribution import (
    ConditionalDistributionSampler,
)
from vpop_calibration.pynlme.importance_sampling import ImportanceSampler

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
        self.population_residuals: (
            pa.typing.DataFrame[WeightedResidualsSchema] | None
        ) = None
        self.iwres: pa.typing.DataFrame[WeightedResidualsSchema] | None = None
        self.sampler = ConditionalDistributionSampler(nlme_model=self.model)
        self.importance_sampler = ImportanceSampler(
            model=self.model,
            df=self.model.config.importance_sampling_df,
        )
        self.shrinkage: torch.Tensor | None = None
        self.vpc: pd.DataFrame | None = None
        self.cached_population_predictions: torch.Tensor | None = None

    def get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            "sampler": self.sampler.get_state_dict(),
            "importance_sampler": self.importance_sampler.get_state_dict(),
        }

        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], nlme_model: StatisticalModel
    ) -> "ModelDiagnostics":
        instance = cls(nlme_model=nlme_model)
        instance.sampler = ConditionalDistributionSampler.from_state_dict(
            state_dict=state_dict["sampler"], model=nlme_model
        )
        instance.importance_sampler = ImportanceSampler.from_state_dict(
            model=nlme_model, state_dict=state_dict["importance_sampler"]
        )
        return instance

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
        if not hasattr(self.sampler, "map"):
            print("No MAPs available, computing them...")
            self.sample_conditional_distribution()
        assert hasattr(self.sampler, "map")

        map_physical_params = self.sampler.map.physical_params_samples
        assert map_physical_params.shape == (
            1,
            self.model.nb_patients,
            self.model.nb_pdu + self.model.nb_mi,
        ), f"{map_physical_params.shape}"

        # Assemble the thetas by adding the PDKs
        theta = self.model.convert_physical_to_thetas_all_patients(
            physical_params=map_physical_params
        )
        model_inputs = self.model.convert_thetas_to_model_parameters_all_patients(
            theta=theta
        )
        simulated_tensor, _ = self.model.predict_all_patients(inputs=model_inputs)

        # Compute residuals and variance
        residuals = calculate_residuals(
            observed_data=self.model.data.full_obs,
            predictions=simulated_tensor,
        )

        variance = compute_error_variance(
            observations=self.model.data.full_obs,
            predictions=simulated_tensor,
            residual_error=self.model.residual_var,
            min_variance=self.model.config.residual_min_variance,
        )

        iwres_full = residuals / torch.sqrt(variance)
        iwres_full.squeeze_(0)

        iwres_list = []

        # Separate IWRES per patient in a dict
        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):
            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            this_patient_iwres = (
                iwres_full[this_patient_rows].squeeze().detach().cpu().numpy()
            )
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

    def simulate_population_samples(
        self, nb_samples=100, chunk_size=20
    ) -> torch.Tensor:
        """
        Sample from the POPULATION distribution (NOT the conditional one), simulates the model and return
        the predictions tensor.
        The eta samples are split by chunks of size 'chunk_size', mostly such that we can display a progress bar
        """
        if smoke_test:
            nb_samples = 3
        # Sample etas in order to approximate mean E(y_i) and variance V_i
        etas = self.model.sample_etas(nb_samples)
        chunks = torch.split(etas, chunk_size, dim=0)
        predictions = []
        with tqdm(
            total=nb_samples,
            desc="Simulating population",
            disable=not self.model.config.progress_bar,
        ) as pbar:
            for etas_chunk in chunks:
                gaussian = self.model.convert_etas_to_gaussian_all_patients(etas_chunk)
                physical = self.model.convert_gaussian_to_physical(
                    psi=gaussian,
                    log_mi=self.model.log_mi,
                    surv_coeffs=self.model.surv_coeffs,
                )
                thetas = self.model.convert_physical_to_thetas_all_patients(
                    physical_params=physical
                )
                patients_inputs = (
                    self.model.convert_thetas_to_model_parameters_all_patients(
                        theta=thetas
                    )
                )
                # Simulate model
                predictions_chunk, _ = self.model.predict_all_patients(
                    inputs=patients_inputs
                )
                predictions.append(predictions_chunk)
                pbar.update(etas_chunk.shape[0])
        population_predictions = torch.cat(predictions, dim=0)
        self.cached_population_predictions = population_predictions
        return population_predictions

    def get_population_predictions(self, nb_samples: int):
        """
        Sample from the POPULATION distribution (NOT the conditional one), simulates the model and return
        the predictions. Reads from the cache if any
        """
        if (
            self.cached_population_predictions is not None
            and self.cached_population_predictions.shape[0] >= nb_samples
        ):
            return self.cached_population_predictions[:nb_samples]
        else:
            return self.simulate_population_samples(nb_samples)

    def compute_population_diagnostics(
        self,
        nb_samples: int = 100,
        vpc_nb_bins=10,
        vpc_quantiles=[0.05, 0.5, 0.95],
        vpc_precision=0.95,
    ):
        """
        A convenient helper to compute all population diagnostics: PWRES, NPDE and VPC
        """
        self.compute_population_residuals(nb_samples=nb_samples)
        self.compute_vpc(
            nb_samples=nb_samples,
            nb_bins=vpc_nb_bins,
            quantiles=vpc_quantiles,
            precision=vpc_precision,
        )

    def compute_population_residuals(
        self, nb_samples: int = 100
    ) -> pa.typing.DataFrame[WeightedResidualsSchema]:
        """Compute PWRES and NPDE from the same simulated population.

        PWRES_i = L_i^(-1) (y_i - E(f_i)), where L_i L_i^T = V_i
        and V_i is the covariance matrix of the model-predicted observations
        (predictive covariance).

        The covariance of observations is decomposed into two contributions: the
        variation between individual profiles and the residual-error contribution
        (which is diagonal because of the independent noise assumption):

        V_i = Cov(f_i) + diag(E(g_i^2))

        where f_i are the model predictions for patient i: f(t_ij, theta_i)
        and g_i^2 are the residual-error variances.

        Why go through all this trouble? To remove the within-patient correlation between measurements.

        NPDE ranks observed PWRES against noisy simulations transformed with the
        same mean E(f_i) and Cholesky factor L_i, then applies the normal inverse CDF.

        Returns:
            A dataframe with one row per observation and diagnostic, identified
            by residual_type ("pwres" or "npde"). Also stored in population_residuals.
        """
        if nb_samples < 2:
            raise ValueError("Population diagnostics require at least two simulations.")
        if self.model.data.full_obs.survival_outputs is not None:
            raise ValueError(
                "Population diagnostics currently support continuous observations only."
            )
        simulated_tensor = self.get_population_predictions(nb_samples)
        # Compute the error variance given the prescribed error model
        variance = compute_error_variance(
            observations=self.model.data.full_obs,
            predictions=simulated_tensor,
            residual_error=self.model.residual_var,
            min_variance=self.model.config.residual_min_variance,
        )
        # Add noise to the predictions for NPDE
        noisy_predictions = add_predictive_error(
            observations=self.model.data.full_obs,
            predictions=simulated_tensor,
            residual_error=self.model.residual_var,
            min_variance=self.model.config.residual_min_variance,
        )
        normal_dist = torch.distributions.Normal(
            simulated_tensor.new_tensor(0.0), simulated_tensor.new_tensor(1.0)
        )

        residuals_list = []

        for i, patient_id in enumerate(
            self.model.data.full_obs.obs_index.id.ref_values
        ):
            this_patient_rows = self.model.data.full_obs.obs_index.id.index_values == i
            # Discard replicates with non-finite predictions for this patient (e.g. failed simulations or NaN or Inf values)
            valid_replicates = torch.isfinite(
                simulated_tensor[:, this_patient_rows]
            ).all(dim=1)
            nb_valid = int(valid_replicates.sum())
            if nb_valid < 2:
                raise ValueError(
                    f"Less than two finite simulated replicates for {patient_id}."
                )
            this_patient_data = simulated_tensor[valid_replicates][:, this_patient_rows]
            observations = self.model.data.individual_observations[patient_id]
            mean_patient = this_patient_data.mean(dim=0)
            centered = this_patient_data - mean_patient
            variance_patient = centered.T @ centered / (nb_valid - 1)
            variance_patient += torch.diag(
                variance[valid_replicates][:, this_patient_rows].mean(dim=0)
            )
            if not torch.isfinite(variance_patient).all():
                raise ValueError(f"Non-finite predictive covariance for {patient_id}.")

            # Scale jitter to each measurement's predictive variance for stability.
            jitter = torch.diag(variance_patient.diagonal() * 1e-6)
            L = torch.linalg.cholesky(variance_patient + jitter)

            # Center and decorrelate the observations
            pwres_patient = torch.linalg.solve_triangular(
                L,
                (observations.obs_values.unsqueeze(0) - mean_patient).T,
                upper=False,
            ).T
            # Center and decorrelate the simulated noisy predictions
            simulated_pwres = torch.linalg.solve_triangular(
                L,
                (
                    noisy_predictions[valid_replicates][:, this_patient_rows]
                    - mean_patient
                ).T,
                upper=False,
            ).T
            # The indicator function of whether each decorrelated simulated value is
            # at or below the decorrelated observation
            is_below_obs = (simulated_pwres <= pwres_patient).to(simulated_tensor.dtype)
            # By averaging over all replicates, we get the empirical CDF of the simulated values evaluated at the
            # observed values.
            # These should be uniform on [0, 1] across observations if each observation is a realization of
            # the simulated distribution, i.e. if the model is correct (that's the probability integral transform)
            empirical_cdf = is_below_obs.mean(dim=0)
            # Map the CDF values to the N(0, 1) space using the standard normal ICDF
            # Why clamp? Because if an observation is either below or above all simulations, it will produce an
            # infinite ICDF value
            # Keep the ICDF finite for NPDE, including when there are only two draws.
            eps = 0.5 / nb_valid
            npde_patient = normal_dist.icdf(empirical_cdf.clamp(min=eps, max=1.0 - eps))
            for residual_type, values in (
                ("pwres", pwres_patient.squeeze(0)),
                ("npde", npde_patient),
            ):
                residuals_list.append(
                    pd.DataFrame(
                        {
                            "id": patient_id,
                            "time": observations.obs_index.time.raw_values,
                            "output_name": observations.obs_index.output_name.raw_values,
                            "residual_value": values.detach().cpu().numpy(),
                            "residual_type": residual_type,
                        }
                    )
                )
        self.population_residuals = WeightedResidualsSchema.validate(
            pd.concat(residuals_list, ignore_index=True)
        )
        return self.population_residuals

    def zero_random_effect_predictions(self) -> None:
        eta = torch.zeros((1, self.model.nb_patients, self.model.nb_pdu))
        gaussian = self.model.convert_etas_to_gaussian_all_patients(eta)
        physical = self.model.convert_gaussian_to_physical(
            psi=gaussian, log_mi=self.model.log_mi, surv_coeffs=self.model.surv_coeffs
        )
        theta = self.model.convert_physical_to_thetas_all_patients(
            physical_params=physical
        )
        inputs = self.model.convert_thetas_to_model_parameters_all_patients(theta)
        pred, _ = self.model.predict_all_patients(inputs)
        pred_df = self.model.data.full_obs.to_pandas(prediction=pred)
        self.population_parameters_predictions_df = pred_df

    def compute_shrinkage(self, nb_samples: int = 50) -> None:

        if not hasattr(self.sampler, "map"):
            self.sampler.run_sampler(nb_samples=nb_samples)
        assert self.sampler.map is not None

        map_etas = self.sampler.map.eta_samples.squeeze(0)

        eta_sd = torch.std(map_etas, dim=0, unbiased=True)
        omega_sd = torch.sqrt(torch.diag(self.model.omega_pop))

        shrinkage = 1 - eta_sd / omega_sd

        self.shrinkage = shrinkage

    def compute_vpc(
        self,
        nb_samples: int = 100,
        nb_bins: int = 10,
        quantiles: list[float] = [0.05, 0.5, 0.95],
        precision: float = 0.95,
    ) -> None:
        """Population VPC: random effects are drawn from N(0, Omega) for each patient."""
        simulated_tensor = self.get_population_predictions(nb_samples)
        # Add residual noise to the predictions
        noisy_predictions = add_predictive_error(
            observations=self.model.data.full_obs,
            predictions=simulated_tensor,
            residual_error=self.model.residual_var,
            min_variance=self.model.config.residual_min_variance,
        )
        # One copy of the observed data per simulated replicate
        obs_df = self.model.data.full_obs.to_pandas()
        nb_replicates = noisy_predictions.shape[0]
        df = pd.concat([obs_df] * nb_replicates, ignore_index=True)
        df["batch_id"] = np.repeat(np.arange(nb_replicates), len(obs_df))
        df["simulated_value_with_noise"] = (
            noisy_predictions.detach().cpu().numpy().reshape(-1)
        )
        all_vpc_records = []
        quantiles_arr = np.asarray(quantiles)

        for output_name in self.model.input_params.continuous_output_names:
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
                "simulated_value_with_noise"
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

    def compute_log_likelihood_importance_sampling(
        self, nb_proposal_samples: int = 100
    ) -> None:
        if not hasattr(self.sampler, "samples"):
            raise ValueError(
                "The conditional distribution has not yet been sampled from. Use `sample_conditional_distribution` first."
            )
        self.importance_sampler.fit_student_t_proposal(
            conditional_samples=self.sampler.total_samples
        )
        self.importance_sampler.compute_likelihood(nb_samples=nb_proposal_samples)
