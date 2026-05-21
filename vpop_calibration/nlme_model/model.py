import torch

from vpop_calibration.structural_model import StructuralModel
from vpop_calibration.data.observed import ObsData
from vpop_calibration.nlme_model.params import (
    MixedEffectParameters,
    ModelIntrinsicParam,
    PatientDescriptorUnknown,
)
from vpop_calibration.config import device


class NlmeModel:
    def __init__(
        self,
        structural_model: StructuralModel,
        dataset: ObsData,
        prior_params: MixedEffectParameters,
        num_chains: int = 1,
    ):
        # Load input data and initiate attributes
        self.structural_model = structural_model
        self.data = dataset
        self.prior_params = prior_params
        self.mi_names = self.prior_params.mi_names
        self.nb_mi = len(self.mi_names)
        self.pdu_names = self.prior_params.pdu_names
        self.nb_pdu = len(self.pdu_names)
        self.pdk_names = self.prior_params.pdk
        self.beta_names = self.prior_params.beta_names
        self.nb_betas = len(self.beta_names)
        self.covariate_names = self.prior_params.covariate_names
        self.nb_covariates = len(self.covariate_names)
        self.patients = self.data.patients
        self.nb_patients = self.data.nb_patients
        self.output_names = self.data.output_names
        self.nb_outputs = len(self.output_names)
        self.num_chains = num_chains
        # Validate observed data against the user-specified parameters
        self.prior_params.validate_data(self.data)
        # Initiate the nlme model parameters
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

        self.update_current_parameters(
            omega=self.init_omega,
            beta=self.init_beta,
            log_mi=self.init_mi,
            res_var=self.init_res_var,
        )

        # Create individual design matrices
        self.design_matrices = self.init_all_design_matrices()
        # Create full design matrix
        # Size (nb_patients, nb_PDU, nb_betas)
        self.full_design_matrix = torch.stack(
            [self.design_matrices[p] for p in self.patients]
        ).to(device)

        # Initiate transforms
        self.pdu_transforms, self.pdu_shift, self.pdu_scale = (
            self.init_transform_tensors(self.prior_params.pdu, self.pdu_names)
        )

        self.mi_transforms, self.mi_shift, self.mi_scale = self.init_transform_tensors(
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

    def init_all_design_matrices(self) -> dict[str | int, torch.Tensor]:
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
        return design_matrices

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
            expected_shape = (self.num_chains, self.nb_patients, self.nb_pdu)
        assert (
            eta.shape == expected_shape
        ), f"Wrong shape in eta samples update: {eta.shape}, expected: {expected_shape}"

        self.eta_samples_chains = eta

    def update_current_parameters(
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

    def init_transform_tensors(
        self,
        param_dict: (
            dict[str, PatientDescriptorUnknown] | dict[str, ModelIntrinsicParam]
        ),
        param_names: list[str],
    ) -> tuple[dict[str, torch.LongTensor], torch.Tensor, torch.Tensor]:
        """Extract transform functions and parameters (scale and shift) into tensors for efficient gaussian parameters transformation."""

        transforms = {
            "exp": torch.LongTensor(
                torch.tensor(
                    [
                        param_names.index(p_name)
                        for p_name, p_content in param_dict.items()
                        if p_content.constraint.transform == "log"
                    ],
                    device=device,
                    dtype=torch.long,
                )
            ),
            "sigmoid": torch.LongTensor(
                torch.tensor(
                    [
                        param_names.index(p_name)
                        for p_name, p_content in param_dict.items()
                        if p_content.constraint.transform == "logit"
                    ],
                    device=device,
                    dtype=torch.long,
                )
            ),
        }
        scale = torch.Tensor(
            [[[param_dict[param].constraint.scale for param in param_names]]]
        )

        shift = torch.Tensor(
            [[[param_dict[param].constraint.shift for param in param_names]]]
        )
        return transforms, shift, scale
