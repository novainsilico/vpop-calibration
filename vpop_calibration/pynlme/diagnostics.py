import torch
from typing import NamedTuple, Literal
import numpy as np
import pandas as pd

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.residuals import (
    calculate_residuals,
    compute_error_variance,
)
from vpop_calibration.pynlme.ebe import compute_ebe_nlme
from vpop_calibration.config import smoke_test
from vpop_calibration.pynlme.conditional_distribution import (
    sample_conditional_distribution_nlme,
)


class PatientResiduals(NamedTuple):
    time: np.ndarray
    res: np.ndarray


ResidualType = Literal["pwres", "iwres", "npde"]
ModelResiduals = dict[str, PatientResiduals]


class ModelDiagnostics:
    def __init__(self, nlme_model: StatisticalModel):
        self.model = nlme_model
        self.individual_ebe_estimates_tensor: torch.Tensor | None = None
        self.individual_ebe_estimates_df: pd.DataFrame | None = None
        self.individual_ebe_predictions_df: pd.DataFrame | None = None
        self.population_parameters_predictions_df: pd.DataFrame | None = None
        self.pwres: ModelResiduals | None = None
        self.iwres: ModelResiduals | None = None
        self.npde: ModelResiduals | None = None
        self.conditional_distribution_samples: torch.Tensor | None = None

    def compute_ebe(self, max_iter: int = 50) -> None:
        self.individual_ebe_estimates_tensor = compute_ebe_nlme(
            nlme_model=self.model, max_iter=max_iter
        )
        # Compute predictions for these estimates, and store in a data frame
        theta = self.model.convert_physical_to_thetas_all_patients(
            self.individual_ebe_estimates_tensor
        )
        self.individual_ebe_estimates_df = self.model.convert_theta_to_dataframe(theta)
        model_inputs = self.model.convert_thetas_to_model_parameters(theta)
        individual_ebe_pred, _ = self.model.predict_all_patients(model_inputs)
        self.individual_ebe_predictions_df = self.model.data.full_obs.to_pandas(
            prediction=individual_ebe_pred
        )

    def compute_iwres(self) -> None:
        """Compute Individual Weighted Residuals (IWRES), following the formula :

        IWRES_(ij) = ( y_ij - f(t_ij, psi_i) ) / g(t_ij, psi_i)
        where psi_i are the patients empirical bayesian estimators.

        Returns:
            dict: IWRES with patientId as key, with IWRES and timesteps for each patient
        """
        if self.individual_ebe_estimates_tensor is None:
            print("No EBEs available, computing them...")
            self.compute_ebe()
        assert self.individual_ebe_estimates_tensor is not None

        assert self.individual_ebe_estimates_tensor.shape == (
            1,
            self.model.nb_patients,
            self.model.nb_pdu + self.model.nb_mi,
        )

        # Assemble the thetas by adding the PDKs
        theta = self.model.convert_physical_to_thetas_all_patients(
            physical_params=self.individual_ebe_estimates_tensor
        )
        model_inputs = self.model.convert_thetas_to_model_parameters(theta=theta)
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
        inputs = self.model.convert_thetas_to_model_parameters(theta=mc_thetas)
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
        inputs = self.model.convert_thetas_to_model_parameters(mc_thetas)

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

    def sample_conditional_distribution(
        self,
        nb_samples: int = 1000,
        nb_burn_in: int = 50,
    ) -> None:
        self.conditional_distribution_samples = sample_conditional_distribution_nlme(
            nlme_model=self.model, nb_samples=nb_samples, nb_burn_in=nb_burn_in
        )

    def zero_random_effect_predictions(self) -> None:
        eta = torch.zeros((1, self.model.nb_patients, self.model.nb_pdu))
        gaussian = self.model.convert_etas_to_gaussian_all_patients(eta)
        physical = self.model.convert_gaussian_to_physical(
            psi=gaussian, log_mi=self.model.log_mi
        )
        theta = self.model.convert_physical_to_thetas_all_patients(
            physical_params=physical
        )
        inputs = self.model.convert_thetas_to_model_parameters(theta)
        pred, _ = self.model.predict_all_patients(inputs)
        pred_df = self.model.data.full_obs.to_pandas(prediction=pred)
        self.population_parameters_predictions_df = pred_df
