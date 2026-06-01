import torch
from typing import get_args, NamedTuple, Callable
import pandas as pd

from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.indexing import ObservationIndex
from vpop_calibration.pynlme.params import MixedEffectParameters, ErrorType
from vpop_calibration.pynlme.utils import init_transform_function
from vpop_calibration.pynlme.residuals import log_likelihood_observation
from vpop_calibration.config import device


class LogPosteriorPrediction(NamedTuple):
    log_posterior: torch.Tensor
    gaussian_params: torch.Tensor
    predictions: torch.Tensor


class StatisticalModel:
    def __init__(
        self,
        structural_model: StructuralModel,
        dataset: ObsData,
        prior_params: MixedEffectParameters,
        nb_chains: int = 1,
    ):
        """Non-linear mixed effects model

        Create a NLME model from three main elements: structural model, observed data and population parameter priors. This class is the main user entrypoint for simulating distributed data and computing likelihood.

        Args:
            structural_model (StructuralModel): The deterministic model equations
            dataset (ObsData): The observed longitudinal data.
            prior_params (MixedEffectParameters): The user-defined parameter priors and configuration.
            num_chains (int, optional): Number of parallel MC chains to track. Defaults to 1.
        """
        # -- Class composition
        self.structural_model = structural_model
        self.data = dataset
        self.prior_params = prior_params

        # -- Attributes initialization
        self.mi_names = self.prior_params.mi_names
        self.nb_mi = len(self.mi_names)
        self.pdu_names = self.prior_params.pdu_names
        self.nb_pdu = len(self.pdu_names)
        self.pdk_names = self.prior_params.pdk
        self.nb_pdk = len(self.pdk_names)
        self.beta_names = self.prior_params.beta_names
        self.nb_betas = len(self.beta_names)
        self.descriptors = self.pdk_names + self.pdu_names + self.mi_names
        self.nb_descriptors = len(self.descriptors)
        self.covariate_names = self.prior_params.covariate_names
        self.nb_covariates = len(self.covariate_names)
        self.covariate_coeff_names = self.prior_params.covariate_coeff_names
        self.patients = self.data.patients
        self.nb_patients = len(self.patients)
        self.nb_chains = nb_chains

        # -- Validation
        # Validate observed data against the user-specified parameters
        self.prior_params.validate_data(self.data)
        # Validate the structural model against user-specified parameters
        assert set(self.descriptors) == set(
            self.structural_model.parameter_names
        ), f"Inconsistent parameter set between patient data and structural model:\nIn the data: {set(self.descriptors)}\nIn the structural model: {set(self.structural_model.parameter_names)}"

        # -- Mapping
        # Map structural model inputs with NLME parameters
        self.nlme_descriptor_to_struct_model_input = torch.as_tensor(
            [
                self.descriptors.index(param)
                for param in self.structural_model.parameter_names
            ],
            device=device,
        ).long()

        # Map observation indices to structural model reference values
        # This applies to output names, protocol arms and tasks
        # The ordering coming from the structural model takes precedence.
        self.output_names = self.structural_model.output_names
        self.nb_outputs = len(self.output_names)
        self.protocol_arms = self.structural_model.protocol_arms
        self.nb_protocols = len(self.protocol_arms)
        self.task_names = self.structural_model.task_names
        self.nb_tasks = len(self.task_names)
        self.data.remap_all_indexings(
            new_output_names=self.output_names,
            new_protocol_arms=self.protocol_arms,
            new_tasks=self.task_names,
        )
        # Map the error model type to the output indices
        self.error_model_selector: dict[ErrorType, list[int]] = {}
        for error_type in get_args(ErrorType):
            self.error_model_selector.update({error_type: []})
        for i, output in enumerate(self.output_names):
            self.error_model_selector[
                self.prior_params.error_model[output].error_type
            ].append(i)

        # -- NLME state initialization
        # Initiate the nlme model parameters in torch tensors
        self.init_beta = torch.as_tensor(self.prior_params.beta_init, device=device)
        self.init_omega = torch.diag(
            torch.as_tensor(
                [
                    self.prior_params.pdu[param].prior_omega
                    for param in self.prior_params.pdu_names
                ]
            )
        )
        self.init_mi = torch.as_tensor(
            [
                self.prior_params.model_intrinsic[param].tansformed_prior
                for param in self.mi_names
            ]
        )
        self.init_res_var = torch.as_tensor(
            [self.prior_params.error_model[out].sigma for out in self.output_names]
        )

        self.set_current_parameters(
            omega=self.init_omega,
            beta=self.init_beta,
            log_mi=self.init_mi,
            res_var=self.init_res_var,
        )

        # Sample some etas to initialize the model state
        etas = self.sample_etas(self.nb_chains)
        self.update_eta_samples(etas)

        # Create design matrices
        self.design_matrices, self.full_design_matrix = self.init_all_design_matrices()

        # Assemble patients pdk tensors
        self.data.init_pdk_values(self.pdk_names)

        # Initiate transforms
        self.pdu_transform = init_transform_function(
            self.prior_params.pdu, self.pdu_names
        )

        self.mi_transform = init_transform_function(
            self.prior_params.model_intrinsic, self.mi_names
        )

    def init_design_matrix(self, patient_covariates: dict) -> torch.Tensor:
        """
        Creates the design matrix X_i for a single individual based on the model's covariate map.
        This matrix will be multiplied with population betas so that log(theta_i[PDU]) = X_i @ betas + eta_i.
        """
        design_matrix_X_i = torch.zeros((self.nb_pdu, self.nb_betas), device=device)
        col_idx = 0
        for i, param in enumerate(self.pdu_names):
            design_matrix_X_i[i, col_idx] = 1.0
            col_idx += 1
            param_covariates = self.prior_params.pdu[param].covariates
            if param_covariates is not None:
                for covariate in param_covariates:
                    design_matrix_X_i[i, col_idx] = float(patient_covariates[covariate])
                    col_idx += 1
        return design_matrix_X_i

    def init_all_design_matrices(
        self,
    ) -> tuple[dict[str | int, torch.Tensor], torch.Tensor]:
        """Creates a design matrix for each unique individual based on their covariates."""
        design_matrices = {}
        if self.nb_covariates == 0:
            # No covariates: all design matrices are the identity matrix
            assert (
                self.nb_betas == self.nb_pdu
            ), "No covariates are identified, yet the number of PDUs and the number of betas differ."
            ind_design_matrix = torch.diag(torch.ones((self.nb_pdu), device=device))
            for ind_id in self.patients:
                design_matrices[ind_id] = ind_design_matrix
        else:
            # The NLME model contains covariates
            for ind_id in self.patients:
                individual_covariates = (
                    self.data.patients_df.loc[
                        self.data.patients_df["id"] == ind_id, self.covariate_names
                    ]
                    .drop_duplicates()
                    .iloc[0]
                )
                covariates_dict = individual_covariates.to_dict()
                design_matrices[ind_id] = self.init_design_matrix(covariates_dict)

        full_design_matrix = torch.stack(
            [design_matrices[p] for p in self.patients]
        ).to(device)
        return design_matrices, full_design_matrix

    def update_omega(self, omega: torch.Tensor) -> None:
        """Update the covariance matrix of the NLME model and the distribution of random effects."""

        if hasattr(self, "omega_pop"):
            expected_shape = self.omega_pop.shape
        else:
            expected_shape = (self.nb_pdu, self.nb_pdu)
        assert (
            omega.shape == expected_shape
        ), f"Wrong shape in omega update: {omega.shape}, expected: {expected_shape}"

        self.omega_pop = omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop).to(device)
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_pdu, device=device),
            covariance_matrix=self.omega_pop,
        )

    def update_res_var(self, residual_var: torch.Tensor) -> None:
        """Update the residual variance of the NLME model, while ensuring it remains positive."""

        if hasattr(self, "residual_var"):
            expected_shape = self.residual_var.shape
        else:
            expected_shape = (self.nb_outputs,)
        assert (
            residual_var.shape == expected_shape
        ), f"Wrong shape in residual variance update: {residual_var.shape}, expected: {expected_shape}"

        self.residual_var = residual_var.clamp(min=1e-6)

    def update_betas(self, betas: torch.Tensor) -> None:
        """Update the betas of the NLME model."""

        if hasattr(self, "population_betas"):
            expected_shape = self.population_betas.shape
        else:
            expected_shape = (self.nb_betas,)
        assert (
            betas.shape == expected_shape
        ), f"Wrong shape in Betas update: {betas.shape}, expected: {expected_shape}"

        self.population_betas = betas

    def update_log_mi(self, log_mi: torch.Tensor) -> None:
        """Update the model intrinsic parameter values of the NLME model."""

        if hasattr(self, "log_mi"):
            expected_shape = self.log_mi.shape
        else:
            expected_shape = (self.nb_mi,)
        assert (
            log_mi.shape == expected_shape
        ), f"Wrong shape in model intrinsic parameters update: {log_mi.shape}, expected: {expected_shape}"

        self.log_mi = log_mi

    def update_eta_samples(self, eta: torch.Tensor) -> None:
        """Update the model current individual random effect samples."""

        if hasattr(self, "eta_samples_chains"):
            expected_shape = self.eta_samples_chains.shape
        else:
            expected_shape = (self.nb_chains, self.nb_patients, self.nb_pdu)
        assert (
            eta.shape == expected_shape
        ), f"Wrong shape in eta samples update: {eta.shape}, expected: {expected_shape}"

        self.eta_samples_chains = eta

    def set_current_parameters(
        self,
        omega: torch.Tensor,
        beta: torch.Tensor,
        log_mi: torch.Tensor,
        res_var: torch.Tensor,
    ):
        """Update or initialize the current population parameter values

        Args:
            omega (torch.Tensor): The random effects covariance matrix
            beta (torch.Tensor): The fixed effects vector (means and covariates)
            log_mi (torch.Tensor): The model intrinsic parameters (log-transformed)
            res_var (torch.Tensor): The residual error variance per output
        """
        self.update_omega(omega)
        self.update_betas(beta)
        self.update_log_mi(log_mi)
        self.update_res_var(res_var)

    # @torch.compile
    def sample_etas(self, nb_samples: int) -> torch.Tensor:
        """Sample individual random effects on all from the current estimate of Omega

        Returns:
            torch.Tensor (nb_samples, nb_patients, nb_PDUs) : individual random effects for all patients in the population, one per chain, per patient, and per PDU.
        """
        etas = self.eta_distribution.expand([self.nb_patients]).sample([nb_samples])
        return etas

    def log_prior_etas(self, etas: torch.Tensor) -> torch.Tensor:
        """Compute log-prior of random effect samples (etas)

        Args:
            etas (torch.Tensor): Individual samples, assuming eta_i ~ N(0, Omega)

        Returns:
            torch.Tensor [nb_eta_i x nb_PDU]: Values of log-prior, computed according to:

            P(eta) = (1/sqrt((2pi)^k * |Omega|)) * exp(-0.5 * eta.T * omega.inv * eta)
            log P(eta) = -0.5 * (k * log(2pi) + log|Omega| + eta.T * omega.inv * eta)

        """
        nb_samples = etas.shape[0]
        nb_patients = etas.shape[1]
        assert etas.shape[2] == self.nb_pdu

        log_priors: torch.Tensor = (
            self.eta_distribution.expand([nb_patients]).log_prob(etas).to(device)
        )
        return log_priors

    # --- Parameter transformation methods
    # @torch.compile
    def _etas_to_gaussian(
        self, etas: torch.Tensor, design_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Atomic method to combine random effects and a design matrix into gaussian parameters."""
        nb_samples = etas.shape[0]

        expanded_design_matrix = design_matrix.expand(nb_samples, -1, -1, -1)
        gaussian_params = expanded_design_matrix @ self.population_betas + etas
        return gaussian_params

    # @torch.compile
    def convert_etas_to_gaussian_all_patients(self, etas: torch.Tensor) -> torch.Tensor:
        """Compute individual (gaussian) parameters from random effects chains

        Args:
            individual_etas (torch.Tensor): Individual random effects samples. Size: (nb_chains, nb_patients, nb_pdu)

        Returns:
            torch.Tensor: The individual parameters in gaussian (unconstrained) space. Size: (nb_chains, nb_patients, nb_pdu)
        """
        nb_samples = etas.shape[0]
        assert etas.shape == torch.Size(
            [nb_samples, self.nb_patients, self.nb_pdu]
        ), f"Wrong shape of etas passed to `transform_etas_to_gaussian`: {etas.shape}"

        gaussian_params = self._etas_to_gaussian(
            etas=etas, design_matrix=self.full_design_matrix
        )

        return gaussian_params

    # @torch.compile
    def convert_gaussian_to_physical(
        self, psi: torch.Tensor, log_mi: torch.Tensor
    ) -> torch.Tensor:
        """Transform gaussian parameters to physical parameters (thetas)

        Args:
            psi (torch.Tensor): Tensor of individual unconstrained parameter values, converted from etas with `convert_etas_to_gaussian`. Size: (nb_chains, nb_patients, nb_pdu)
            log_mi (torch.Tensor): Tensor of current estimates for the (transformed) model intrinsic parameters.

        Returns:
            torch.Tensor: Tensor of individual physical parameter values. Both PDUs and MIs are included. Size: (nb_chains, nb_patients, nb_pdu + nb_mi)
        """
        nb_samples = psi.shape[0]
        nb_patients_local = psi.shape[1]
        assert psi.shape[2] == self.nb_pdu

        # Apply the transforms
        pdu = self.pdu_transform(psi)
        mi = self.mi_transform(log_mi.expand(nb_samples, nb_patients_local, self.nb_mi))

        phi = torch.cat((pdu, mi), dim=-1).to(device)
        return phi

    # @torch.compile
    def _combine_physical_pdk(
        self, physical_params: torch.Tensor, pdk: torch.Tensor
    ) -> torch.Tensor:
        """Atomic method to combine physical parameters and pdks"""
        nb_samples = physical_params.shape[0]
        nb_patients_local = physical_params.shape[1]
        assert physical_params.shape[2] == self.nb_pdu + self.nb_mi
        assert pdk.shape == (
            nb_patients_local,
            self.nb_pdk,
        ), f"Inconsistent shapes provided in _combine_physical_pdk:\n{physical_params.shape=}\n{pdk.shape=}"

        pdk_expanded = pdk.expand(nb_samples, -1, -1)
        theta = torch.cat((pdk_expanded, physical_params), dim=-1)

        return theta

    # @torch.compile
    def convert_physical_to_thetas_all_patients(
        self, physical_params: torch.Tensor
    ) -> torch.Tensor:
        """Assemble patient individual parameters

        The patient individual parameters are assembled, always in the following order (pdk, pdu, mi).

        Args:
            physical_params (Tensor): Parameters converted to the physical space with `convert_gaussian_to_physical`

        Returns:
            torch.Tensor: One parameter set for each patient. Size (nb_samples, nb_patients, nb_descriptors)
        """
        nb_samples = physical_params.shape[0]
        assert physical_params.shape == (
            nb_samples,
            self.nb_patients,
            self.nb_pdu + self.nb_mi,
        )
        theta = self._combine_physical_pdk(
            physical_params=physical_params, pdk=self.data.patients_pdk_full
        )
        assert theta.shape == (nb_samples, self.nb_patients, self.nb_descriptors)
        return theta

    # @torch.compile
    def convert_thetas_to_model_parameters(self, theta: torch.Tensor) -> torch.Tensor:
        """Assemble model inputs

        Args:
            thetas (torch.Tensor): Parameter values per patient

        Returns:
            torch.Tensor: the full inputs required to simulate all patients on all time steps. Size (nb_samples, nb_patients, nb_time_steps, nb_descriptors+1).
        """
        nb_samples = theta.shape[0]
        nb_patients_local = theta.shape[1]

        assert theta.shape[2] == self.nb_descriptors

        theta_expanded = (
            theta[:, :, self.nlme_descriptor_to_struct_model_input]
            .unsqueeze(-2)
            .expand((-1, -1, self.data.nb_global_timesteps, -1))
        )
        time_steps_expanded = (
            self.data.global_timesteps.unsqueeze(0)
            .unsqueeze(-1)
            .repeat((nb_patients_local, 1, 1))
        )
        struct_model_inputs = torch.cat(
            (theta_expanded, time_steps_expanded.expand(nb_samples, -1, -1, -1)), dim=-1
        )

        assert struct_model_inputs.shape == (
            nb_samples,
            nb_patients_local,
            self.data.nb_global_timesteps,
            self.nb_descriptors + 1,
        ), f"Unexpected shape {struct_model_inputs.shape}"
        return struct_model_inputs

    def _predict(
        self, inputs: torch.Tensor, pred_index: ObservationIndex
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mean, pred_var = self.structural_model.simulate(
            X=inputs, prediction_index=pred_index
        )
        return pred_mean, pred_var

    def predict_all_patients(self, inputs: torch.Tensor):
        """Return model predictions for all patients

        This function carries the number of chains in the batch dimension (0).

        Args:
            thetas (torch.Tensor): Parameter values per patient. Size (nb_samples, nb_patients, nb_time_steps, nb_descriptors + 1)

        Returns:
            torch.Tensor: predicted mean. Size (nb_samples, nb_patients, nb_time_steps, nb_outputs)
            torch.Tensor: predicted variance (if applicable). Size (nb_samples, nb_patients, nb_time_steps, nb_outputs)
        """
        nb_samples = inputs.shape[0]
        assert inputs.shape == (
            nb_samples,
            self.nb_patients,
            self.data.nb_global_timesteps,
            self.nb_descriptors + 1,
        )
        pred_mean, pred_var = self._predict(
            inputs=inputs, pred_index=self.data.full_obs.obs_index
        )

        assert pred_mean.shape == (
            nb_samples,
            self.data.nb_total_observations,
        )

        return pred_mean, pred_var

    def log_posterior_etas_all_patients(
        self, etas: torch.Tensor
    ) -> LogPosteriorPrediction:
        nb_samples = etas.shape[0]
        assert etas.shape == (nb_samples, self.nb_patients, self.nb_pdu)

        gaussian_params = self.convert_etas_to_gaussian_all_patients(etas)
        physical_params = self.convert_gaussian_to_physical(
            gaussian_params, self.log_mi
        )
        thetas = self.convert_physical_to_thetas_all_patients(physical_params)
        inputs = self.convert_thetas_to_model_parameters(thetas)
        pred, _ = self.predict_all_patients(inputs)

        log_prior = self.log_prior_etas(etas)
        assert log_prior.shape == (nb_samples, self.nb_patients)

        log_likelihood_obs = log_likelihood_observation(
            self.data.full_obs, pred, self.error_model_selector, self.residual_var
        )
        assert log_likelihood_obs.shape == (nb_samples, self.nb_patients)

        log_posterior = log_likelihood_obs + log_prior

        return LogPosteriorPrediction(
            log_posterior=log_posterior,
            gaussian_params=gaussian_params,
            predictions=pred,
        )

    def single_patient_likelihood_factory(
        self, id: str
    ) -> Callable[[torch.Tensor], LogPosteriorPrediction]:
        observations = self.data.individual_observations[id]
        design_matrix = self.design_matrices[id].unsqueeze(0)
        pdk = self.data.patients_pdk[id]

        def log_posterior_etas_single_patient(
            etas: torch.Tensor,
        ) -> LogPosteriorPrediction:
            nb_samples = etas.shape[0]
            assert etas.shape[1] == 1
            assert etas.shape[2] == self.nb_pdu

            gaussian_params = self._etas_to_gaussian(
                etas=etas, design_matrix=design_matrix
            )
            physical_params = self.convert_gaussian_to_physical(
                psi=gaussian_params, log_mi=self.log_mi
            )
            thetas = self._combine_physical_pdk(
                physical_params=physical_params, pdk=pdk
            )
            inputs = self.convert_thetas_to_model_parameters(thetas)
            pred, _ = self._predict(inputs=inputs, pred_index=observations.obs_index)

            log_prior = self.log_prior_etas(etas)
            assert log_prior.shape == (nb_samples, 1)

            log_likelihood_obs = log_likelihood_observation(
                observations=observations,
                predictions=pred,
                error_model_selector=self.error_model_selector,
                sigma=self.residual_var,
            )
            assert log_likelihood_obs.shape == (nb_samples, 1)

            log_posterior = log_likelihood_obs + log_prior

            return LogPosteriorPrediction(
                log_posterior=log_posterior,
                gaussian_params=gaussian_params,
                predictions=pred,
            )

        return log_posterior_etas_single_patient

    def convert_theta_to_dataframe(self, theta: torch.Tensor) -> pd.DataFrame:
        assert theta.shape[0] == 1, "Cannot convert batched parameters to dataframe."
        vpop = pd.DataFrame(theta.squeeze(0).cpu().numpy(), columns=self.descriptors)
        vpop["id"] = self.patients

        return vpop
