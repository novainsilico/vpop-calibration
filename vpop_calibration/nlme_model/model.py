import torch

from vpop_calibration.structural_model import StructuralModel
from vpop_calibration.nlme_model.data import ObsData
from vpop_calibration.nlme_model.params import (
    MixedEffectParameters,
)
from vpop_calibration.nlme_model.utils import init_transform_function
from vpop_calibration.config import device


class NlmeModel:
    def __init__(
        self,
        structural_model: StructuralModel,
        dataset: ObsData,
        prior_params: MixedEffectParameters,
        num_chains: int = 1,
    ):
        """Non-linear mixed effects model

        Create a NLME model from three main elements: structural model, observed data and population parameter priors.

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
        # self.task_map = self.data.task_map
        # Both the structural model and the data come with a task map.
        # The data task map takes precedence in the NLME model
        # The structural model task map is validated against it by checking protocol arms and output names exist in the task map.
        self.task_map.validate_tasks(
            self.structural_model.task_map.protocol_arms,
            self.structural_model.task_map.output_names,
        )

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
        self.patients = self.data.patients
        self.nb_patients = self.data.nb_patients
        self.output_names = self.data.output_names
        self.nb_outputs = len(self.output_names)
        self.num_chains = num_chains

        # -- Validation
        # Validate observed data against the user-specified parameters
        self.prior_params.validate_data(self.data)
        # Validate the structural model against user-specified parameters
        assert set(self.output_names) == set(self.task_map.output_names)
        assert set(self.descriptors) == set(
            self.structural_model.parameter_names
        ), f"Inconsistent parameter set between patient data and structural model:\nIn the data: {set(self.descriptors)}\nIn the structural model: {set(self.structural_model.parameter_names)}"
        assert set(self.structural_model.task_map.protocol_arms) == set(
            self.data.protocol_arms
        ), f"Inconsistent protocol arms between patient data and structural model:\nIn the data: {set(
            self.data.protocol_arms
        )}\nIn the structural model: {set(self.structural_model.task_map.protocol_arms)}"

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
        # Map structural model inputs with NLME parameters
        # We assume that the descriptors will always be provided to the model in the following order: PDK, PDU, MI
        # This is enforced in `convert_gaussian_to_physical` and `convert_physical_to_theta`
        self.struct_model_input_to_descriptor = torch.as_tensor(
            [
                self.descriptors.index(param)
                for param in self.structural_model.parameter_names
            ],
            device=device,
        ).long()

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
            ind_design_matrix = torch.diag(
                torch.ones((self.nb_pdu, self.nb_pdu), device=device)
            )
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
        ).expand([self.nb_patients])

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
            expected_shape = (self.num_chains, self.nb_patients, self.nb_pdu)
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

    def sample_etas(self, nb_samples: int) -> torch.Tensor:
        """Sample individual random effects on all from the current estimate of Omega

        Returns:
            torch.Tensor (nb_samples, nb_patients, nb_PDUs) : individual random effects for all patients in the population, one per chain, per patient, and per PDU.
        """
        etas = self.eta_distribution.sample([nb_samples])
        return etas

    def convert_etas_to_gaussian(self, etas: torch.Tensor) -> torch.Tensor:
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

        expanded_design_matrix = self.full_design_matrix.expand(nb_samples, -1, -1, -1)
        gaussian_params = expanded_design_matrix @ self.population_betas + etas

        return gaussian_params

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
        assert psi.shape == (nb_samples, self.nb_patients, self.nb_pdu)

        # Apply the transforms
        pdu = self.pdu_transform(psi)
        mi = self.mi_transform(log_mi.expand(nb_samples, self.nb_patients, self.nb_mi))

        phi = torch.cat((pdu, mi), dim=-1).to(device)
        return phi

    def convert_physical_to_thetas(self, physical_params: torch.Tensor) -> torch.Tensor:
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
        pdk_tensor = self.data.patients_pdk_full
        theta = torch.cat(
            (pdk_tensor.expand(nb_samples, -1, -1), physical_params), dim=-1
        )

        assert theta.shape == (nb_samples, self.nb_patients, self.nb_descriptors)
        return theta

    def convert_thetas_to_model_parameters(self, theta: torch.Tensor) -> torch.Tensor:
        """Assemble model inputs for all patients

        Args:
            thetas (torch.Tensor): Parameter values per patient

        Returns:
            torch.Tensor: the full inputs required to simulate all patients on all time steps. Size (nb_samples, nb_patients, nb_time_steps, nb_descriptors+1).
        """
        nb_samples = theta.shape[0]
        assert theta.shape == (nb_samples, self.nb_patients, self.nb_descriptors)

        theta_expanded = (
            theta[:, :, self.struct_model_input_to_descriptor]
            .unsqueeze(-2)
            .expand((-1, -1, self.data.nb_global_time_steps, -1))
        )
        time_steps_expanded = (
            self.data.time_steps_tensor.unsqueeze(0)
            .unsqueeze(-1)
            .repeat((self.nb_patients, 1, 1))
        )
        struct_model_inputs = torch.cat(
            (theta_expanded, time_steps_expanded.expand(nb_samples, -1, -1, -1)), dim=-1
        )

        assert struct_model_inputs.shape == (
            nb_samples,
            self.nb_patients,
            self.data.nb_global_time_steps,
            self.nb_descriptors + 1,
        ), f"Unexpected shape {struct_model_inputs.shape}"
        return struct_model_inputs

    def predict(self, inputs: torch.Tensor):
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
            self.data.nb_global_time_steps,
            self.nb_descriptors + 1,
        )
        prediction_index = self.data.prediction_index
        pred_mean, pred_var = self.structural_model.simulate(inputs, prediction_index)

        assert pred_mean.shape == (
            nb_samples,
            self.data.nb_total_observations,
        )

        return pred_mean, pred_var
