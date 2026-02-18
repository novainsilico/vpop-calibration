import torch
from typing import Union, Optional, Callable, cast
import pandas as pd
import numpy as np
from .structural_model import StructuralModel
from .utils import device, smoke_test
from tqdm.notebook import tqdm
from scipy.optimize import minimize


class NlmeModel:
    def __init__(
        self,
        structural_model: StructuralModel,
        patients_df: pd.DataFrame,
        init_log_MI: dict[str, float],
        init_PDU: dict[str, dict[str, float]],
        init_res_var: list[float],
        covariate_map: Optional[dict[str, dict[str, dict[str, str | float]]]] = None,
        error_model_type: str = "additive",
        pred_var_threshold: float = 1e-2,
        num_chains: int = 3,
        constraints: Optional[dict[str, dict]] = None,
    ):
        """Create a non-linear mixed effects model

        Using a structural model (simulation model) and a covariate structure, create a non-linear mixed effects model, to be used in PySAEM or another optimizer, or to predict data using a covariance structure.

        Args:
            structural_model (StructuralModel): A simulation model defined via the convenience class StructuralModel
            patients_df (DataFrame): the list of patients to be considered, with potential covariate values listed, and the name of the protocol arm on which the patient was evaluated (optional - if not supplied, `identity` will be used). The `id` column is expected, any additional column will be handled as a covariate
            init_log_MI: for each model intrinsic parameter, provide an initial value (log)
            init_PDU: for each patient descriptor unknown parameter, provide an initial mean and sd of the log
            init_res_var: for each model output, provide an initial residual variance
            covariate_map (optional[dict]): for each PDU, the list of covariates that affect it - each associated with a covariation coefficient (to be calibrated)
            Example
                {"pdu_name":
                    {"covariate_name":
                        {"coef": "coef_name", "value": initial_value}
                    }
                }
            error_model_type (str): either `additive` or `proportional` error model
            pred_var_threshold (float): Threshold of predictive variance that will issue a warning. Default 1e-2.
            num_chains (int): Number of parallel chains to track for each patient
            constraints (dict): Box constraints for each random parameter. Should be a map that associates `pdu_name: {low: value|None, high: value|None}`. Any unspecified constraint will default to `{low: 0, high: Inf}` and result in exponential transformation of the associated gaussian parameter.
        """
        self.structural_model: StructuralModel = structural_model
        self.pred_var_threshold = pred_var_threshold

        self.MI_names: list[str] = list(init_log_MI.keys())
        self.nb_MI: int = len(self.MI_names)
        self.init_log_MI = torch.tensor([val for _, val in init_log_MI.items()]).to(
            device
        )
        self.PDU_names: list[str] = list(init_PDU.keys())
        self.nb_PDU: int = len(self.PDU_names)

        if set(self.MI_names) & set(self.PDU_names):
            raise ValueError(
                f"Overlapping model intrinsic and PDU descriptors:{(set(self.MI_names) & set(self.PDU_names))}"
            )

        self.patients_df: pd.DataFrame = patients_df.drop_duplicates()
        self.patients: list[str | int] = self.patients_df["id"].unique().tolist()
        self.nb_patients: int = len(self.patients)
        covariate_columns = self.patients_df.columns.to_list()
        if "protocol_arm" not in covariate_columns:
            self.patients_df["protocol_arm"] = "identity"

        additional_columns: list[str] = self.patients_df.drop(
            ["id", "protocol_arm"], axis=1
        ).columns.tolist()

        init_betas_list: list = []
        if covariate_map is None:
            print(
                f"No covariate map provided. All additional columns in `patients_df` will be handled as known descriptors: {additional_columns}"
            )
            self.covariate_map = None
            self.covariate_names = []
            self.covariate_coeffs_names = []
            self.nb_covariates = 0
            self.population_betas_names = self.PDU_names
            init_betas_list = [val["mean"] for _, val in init_PDU.items()]
            self.PDK_names = additional_columns
            self.nb_PDK = len(self.PDK_names)
        else:
            self.covariate_map = covariate_map
            self.population_betas_names: list = []
            covariate_set = set()
            covariate_coeffs_set = set()
            pdk_names = set(additional_columns)
            for PDU_name in self.PDU_names:
                self.population_betas_names.append(PDU_name)
                init_betas_list.append(init_PDU[PDU_name]["mean"])
                if PDU_name not in covariate_map:
                    raise ValueError(
                        f"No covariate map listed for {PDU_name}. Add an empty set if it has no covariate."
                    )
                for covariate, coef in self.covariate_map[PDU_name].items():
                    if covariate not in additional_columns:
                        raise ValueError(
                            f"Covariate appears in the map but not in the patient set: {covariate}"
                        )
                    if covariate is not None:
                        covariate_set.add(covariate)
                        if covariate in pdk_names:
                            pdk_names.remove(covariate)
                        coef_name = coef["coef"]
                        covariate_coeffs_set.add(coef_name)
                        coef_val = coef["value"]
                        self.population_betas_names.append(coef_name)
                        init_betas_list.append(coef_val)
            self.covariate_names = list(covariate_set)
            self.covariate_coeffs_names = list(covariate_coeffs_set)
            self.nb_covariates = len(self.covariate_names)
            self.PDK_names = list(pdk_names)
            self.nb_PDK = len(self.PDK_names)

        print(f"Successfully loaded {self.nb_covariates} covariates:")
        print(self.covariate_names)
        if self.nb_PDK > 0:
            self.patients_pdk = {}
            for patient in self.patients:
                row = self.patients_df.loc[
                    self.patients_df["id"] == patient
                ].drop_duplicates()
                self.patients_pdk.update(
                    {
                        patient: torch.as_tensor(
                            row[self.PDK_names].values, device=device
                        )
                    }
                )
            # Store the full pdk tensor on the device
            self.patients_pdk_full = torch.cat(
                [self.patients_pdk[ind] for ind in self.patients]
            ).to(device)
            print(f"Successfully loaded {self.nb_PDK} known descriptors:")
            print(self.PDK_names)
        else:
            # Create an empty pdk tensor
            self.patients_pdk_full = torch.empty((self.nb_patients, 0), device=device)

        if set(self.PDK_names + self.PDU_names + self.MI_names) != set(
            self.structural_model.parameter_names
        ):
            raise ValueError(
                f"Non-matching descriptor set and structural model parameter set:\nPDK, PDU, MI: {set(self.PDK_names + self.PDU_names + self.MI_names)}\nStructural model descriptors: {set(self.structural_model.parameter_names)}"
            )

        self.descriptors: list[str] = self.PDK_names + self.PDU_names + self.MI_names
        self.nb_descriptors: int = len(self.descriptors)
        # Assume that the descriptors will always be provided to the model in the following order:
        #   PDK, PDU, MI
        self.model_input_to_descriptor = torch.as_tensor(
            [
                self.descriptors.index(param)
                for param in self.structural_model.parameter_names
            ],
            device=device,
        ).long()
        self.initial_betas = torch.as_tensor(init_betas_list, device=device)
        self.nb_betas: int = len(self.population_betas_names)
        self.outputs_names: list[str] = self.structural_model.output_names
        self.nb_outputs: int = self.structural_model.nb_outputs
        self.error_model_type: str = error_model_type
        self.init_res_var = torch.as_tensor(init_res_var, device=device)
        # Initialize the covariance matrix with diagonal entries
        self.init_omega = torch.diag(
            torch.as_tensor(
                [float(init_PDU[pdu]["sd"]) for pdu in self.PDU_names], device=device
            )
        )

        # Assemble the list of design matrices from the covariance structure
        self.design_matrices = self._create_all_design_matrices()
        # Store the full design matrix on the device
        # Size (nb_patients, nb_PDU, nb_betas)
        self.full_design_matrix = torch.stack(
            [self.design_matrices[p] for p in self.patients]
        ).to(device)

        # Initiate the NLME parameters
        self.num_chains = num_chains
        self.update_current_parameters(
            omega=self.init_omega,
            beta=self.initial_betas,
            log_mi=self.init_log_MI,
            res_var=self.init_res_var,
        )

        self.pdu_transforms = {"exp": [], "sigmoid": []}
        self.pdu_shift = torch.zeros(1, 1, self.nb_PDU)
        self.pdu_scale = torch.ones(1, 1, self.nb_PDU)

        self.mi_transforms = {"exp": [], "sigmoid": []}
        self.mi_shift = torch.zeros(self.nb_MI)
        self.mi_scale = torch.ones(self.nb_MI)

        if constraints is not None:
            for pdu_index, pdu in enumerate(self.PDU_names):
                if pdu in constraints:
                    # Constraints are specified for this PDU
                    bounds = constraints[pdu]
                    if bounds["low"] is not None:
                        # A lower bound is specified. Include it in the shift
                        self.pdu_shift[0, 0, pdu_index] = bounds["low"]
                        # If no shift is specified, a 0 shift is kept
                    if bounds["high"] is not None:
                        # A high bound is specified: need to include the PDU in the sigmoid transformation group
                        self.pdu_transforms["sigmoid"].append(pdu_index)
                        # Compute the scaling factor
                        self.pdu_scale[0, 0, pdu_index] = (
                            bounds["high"] - self.pdu_shift[0, 0, pdu_index]
                        )
                    else:
                        # No high bound specified -> the exp transformation will be used
                        self.pdu_transforms["exp"].append(pdu_index)
                else:
                    # No constraints specified for this pdu -> the 0 shift and exp transformation will be used
                    self.pdu_transforms["exp"].append(pdu_index)
            print(
                "Successfully loaded the following PDU constraints and transformations:"
            )
            print(self.pdu_transforms)
            print(f"Shift: {self.pdu_shift[0, 0, :]}, scale: {self.pdu_scale[0, 0, :]}")
            for mi_index, mi in enumerate(self.MI_names):
                if mi in constraints:
                    # Constraints are specified for this MI
                    bounds = constraints[mi]
                    if bounds["low"] is not None:
                        # A lower bound is specified. Include it in the shift
                        self.mi_shift[mi_index] = bounds["low"]
                        # If no shift is specified, a 0 shift is kept
                    if bounds["high"] is not None:
                        # A high bound is specified: need to include the MI in the sigmoid transformation group
                        self.mi_transforms["sigmoid"].append(mi_index)
                        # Compute the scaling factor
                        self.mi_scale[mi_index] = (
                            bounds["high"] - self.mi_shift[mi_index]
                        )
                    else:
                        # No high bound specified -> the exp transformation will be used
                        self.mi_transforms["exp"].append(mi_index)

                else:
                    # No constraints specified for this MI -> the 0 shift and exp transformation will be used
                    self.mi_transforms["exp"].append(mi_index)

            print(
                "Successfully loaded the following MI constraints and transformations:"
            )
            print(self.mi_transforms)
            print(f"Shift: {self.mi_shift}, scale: {self.mi_scale}")
        else:
            # No constraints specified at all: 0 shift and exp transformation for all params
            self.pdu_transforms["exp"] = list(np.arange(self.nb_PDU))
            self.mi_transforms["exp"] = list(np.arange(self.nb_MI))

        self.pdu_transforms = {
            "exp": torch.tensor(
                self.pdu_transforms["exp"], dtype=torch.long, device=device
            ),
            "sigmoid": torch.tensor(
                self.pdu_transforms["sigmoid"], dtype=torch.long, device=device
            ),
        }

        self.mi_transforms = {
            "exp": torch.tensor(
                self.mi_transforms["exp"], dtype=torch.long, device=device
            ),
            "sigmoid": torch.tensor(
                self.mi_transforms["sigmoid"], dtype=torch.long, device=device
            ),
        }
        # Initiate the patient parameters
        self.init_etas = self.sample_etas(self.num_chains, self.nb_patients)
        self.update_eta_samples(self.init_etas)
        # Compute a first naive guess for the patient parameters on all chains
        gaussian_params = self.etas_to_gaussian_params(self.eta_samples_chains)
        physical_params = self.gaussian_to_physical_params(gaussian_params, self.log_MI)
        # Average over the chains to estimate the current patient descriptors
        theta = self.assemble_individual_parameters(physical_params).mean(dim=0)

    def _create_design_matrix(self, covariates: dict[str, float]) -> torch.Tensor:
        """
        Creates the design matrix X_i for a single individual based on the model's covariate map.
        This matrix will be multiplied with population betas so that log(theta_i[PDU]) = X_i @ betas + eta_i.
        """
        design_matrix_X_i = torch.zeros((self.nb_PDU, self.nb_betas), device=device)
        col_idx = 0
        for i, PDU_name in enumerate(self.PDU_names):
            design_matrix_X_i[i, col_idx] = 1.0
            col_idx += 1
            if self.covariate_map is not None:
                for covariate in self.covariate_map[PDU_name].keys():
                    design_matrix_X_i[i, col_idx] = float(covariates[covariate])
                    col_idx += 1
        return design_matrix_X_i

    def _create_all_design_matrices(self) -> dict[Union[str, int], torch.Tensor]:
        """Creates a design matrix for each unique individual based on their covariates, given the in the covariates_df."""
        design_matrices = {}
        if self.nb_covariates == 0:
            for ind_id in self.patients:
                design_matrices[ind_id] = self._create_design_matrix({})
        else:
            for ind_id in self.patients:
                individual_covariates = (
                    self.patients_df[self.patients_df["id"] == ind_id]
                    .iloc[0]
                    .drop("id")
                )
                covariates_dict = individual_covariates.to_dict()
                design_matrices[ind_id] = self._create_design_matrix(covariates_dict)
        return design_matrices

    def add_observations(self, observations_df: pd.DataFrame) -> None:
        """Associate the NLME model with a data frame of observations

        Args:
            observations_df (pd.DataFrame): A data frame of observations, with columns
            - `id`: the patient id. Should be consistent with self.patients_df
            - `time`: the observation time
            - `output_name`
            - `value`
        """
        # Store the raw data frame
        self.observations_df = observations_df
        # Data validation
        input_columns = observations_df.columns.tolist()
        unique_outputs = observations_df["output_name"].unique().tolist()
        if "id" not in input_columns:
            raise ValueError(
                "Provided observation data frame should contain `id` column."
            )
        input_patients = observations_df["id"].unique()
        if set(input_patients) != set(self.patients):
            # Note this check might be unnecessary
            raise ValueError(
                f"Missing observations for the following patients: {set(self.patients) - set(input_patients)}"
            )
        if "time" not in input_columns:
            raise ValueError(
                "Provided observation data frame should contain `time` column."
            )
        if not (set(unique_outputs) <= set(self.outputs_names)):
            raise ValueError(
                f"Unknown model output: {set(unique_outputs) - set(self.outputs_names)}"
            )
        if hasattr(self, "observations_tensors"):
            print(
                "Warning: overriding existing observation data frame for the NLME model"
            )
        if "value" not in input_columns:
            raise ValueError(
                "The provided observations data frame does not contain a `value` column."
            )
        processed_df = observations_df[["id", "output_name", "time", "value"]].merge(
            self.patients_df, how="left", on="id"
        )
        processed_df["task"] = processed_df.apply(
            lambda r: r["output_name"] + "_" + r["protocol_arm"], axis=1
        )
        processed_df["task_index"] = processed_df["task"].apply(
            lambda t: self.structural_model.tasks.index(t)
        )
        processed_df["output_index"] = processed_df["output_name"].apply(
            lambda o: self.structural_model.output_names.index(o)
        )
        global_time_steps = (
            processed_df["time"].drop_duplicates().sort_values().to_list()
        )
        processed_df["time_step_index"] = processed_df["time"].apply(
            lambda t: global_time_steps.index(t)
        )
        self.global_time_steps = torch.as_tensor(global_time_steps, device=device)
        self.nb_global_time_steps = self.global_time_steps.shape[0]
        self.global_time_steps_expanded = (
            self.global_time_steps.unsqueeze(0)
            .unsqueeze(-1)
            .repeat((self.nb_patients, 1, 1))
        )
        # Browse the observed data set and store relevant elements
        self.observations_tensors: dict = {}
        self.n_tot_observations_per_output = torch.zeros(self.nb_outputs, device=device)
        for ind, patient in enumerate(self.patients):
            this_patient = processed_df.loc[processed_df["id"] == patient]

            tasks_indices_np = this_patient["task_index"].values
            tasks_indices = torch.as_tensor(tasks_indices_np, device=device).long()

            outputs_indices_np = this_patient["output_index"].values
            outputs_indices = torch.as_tensor(outputs_indices_np, device=device).long()
            # Add counts of observations to the total per output
            self.n_tot_observations_per_output.scatter_add_(
                0,
                outputs_indices,
                torch.ones_like(outputs_indices, device=device, dtype=torch.float64),
            )

            outputs = torch.as_tensor(this_patient["value"].values, device=device)

            time_steps = torch.as_tensor(this_patient["time"].values, device=device)
            time_step_indices = torch.as_tensor(
                this_patient["time_step_index"].values, device=device
            ).long()
            p_index_repeated = torch.full(
                outputs.shape, ind, dtype=torch.int64, device=device
            )

            self.observations_tensors.update(
                {
                    patient: {
                        "observations": outputs,
                        "time_steps": time_steps,
                        "time_step_indices": time_step_indices,
                        "tasks_indices": tasks_indices,
                        "outputs_indices": outputs_indices,
                        "p_index_repeated": p_index_repeated,
                    }
                }
            )
        self.nb_tot_obs = self.n_tot_observations_per_output.sum()

        # Build the full data set tensors
        self.full_obs_data = torch.cat(
            [self.observations_tensors[p]["observations"] for p in self.patients]
        ).to(device)
        assert self.full_obs_data.shape[0] == self.nb_tot_obs

        # Construct the indexing tensors
        self.observation_to_patient_index = (
            torch.cat(
                [
                    self.observations_tensors[p]["p_index_repeated"]
                    for p in self.patients
                ]
            )
            .long()
            .to(device)
        )
        self.observation_to_timestep_index = (
            torch.cat(
                [
                    self.observations_tensors[p]["time_step_indices"]
                    for p in self.patients
                ]
            )
            .long()
            .to(device)
        )
        self.observation_to_task_index = (
            torch.cat(
                [self.observations_tensors[p]["tasks_indices"] for p in self.patients]
            )
            .long()
            .to(device)
        )
        # Construct a tuple allowing to index a 3D tensor of outputs into a 1D tensor of outputs
        self.prediction_index = (
            self.observation_to_patient_index,
            self.observation_to_timestep_index,
            self.observation_to_task_index,
        )

        self.full_output_indices = (
            torch.cat(
                [self.observations_tensors[p]["outputs_indices"] for p in self.patients]
            )
            .long()
            .to(device)
        )
        self.chunk_sizes: list[int] = [
            self.observations_tensors[p]["observations"].cpu().shape[0]
            for p in self.patients
        ]

    def update_omega(self, omega: torch.Tensor) -> None:
        """Update the covariance matrix of the NLME model and the distribution of random effects."""

        if hasattr(self, "omega_pop"):
            expected_shape = self.omega_pop.shape
        else:
            expected_shape = (self.nb_PDU, self.nb_PDU)
        assert (
            omega.shape == expected_shape
        ), f"Wrong shape in omega update: {omega.shape}, expected: {expected_shape}"

        self.omega_pop = omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop).to(device)
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU, device=device),
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

    def update_log_mi(self, log_MI: torch.Tensor) -> None:
        """Update the model intrinsic parameter values of the NLME model."""

        if hasattr(self, "log_MI"):
            expected_shape = self.log_MI.shape
        else:
            expected_shape = (self.nb_MI,)
        assert (
            log_MI.shape == expected_shape
        ), f"Wrong shape in model intrinsic parameters update: {log_MI.shape}, expected: {expected_shape}"

        self.log_MI = log_MI

    def update_eta_samples(self, eta: torch.Tensor) -> None:
        """Update the model current individual random effect samples."""

        if hasattr(self, "eta_samples_chains"):
            expected_shape = self.eta_samples_chains.shape
        else:
            expected_shape = (self.num_chains, self.nb_patients, self.nb_PDU)
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
        """
        self.update_omega(omega)
        self.update_betas(beta)
        self.update_log_mi(log_mi)
        self.update_res_var(res_var)

    def sample_etas(self, nb_samples, nb_patients) -> torch.Tensor:
        """Sample individual random effects on all chains from the current estimate of Omega

        Returns:
            torch.Tensor : individual random effects for all patients in the population, one per chain. Size: (num_chains, nb_patients, nb_PDUs)
        """
        etas = self.eta_distribution.sample([nb_samples, nb_patients])
        return etas

    # @torch.compile
    def etas_to_gaussian_params(
        self,
        individual_etas: torch.Tensor,
        ind_ids_for_etas: Optional[list[str | int]] = None,
    ) -> torch.Tensor:
        """Compute individual (gaussian) parameters from random effects chains

        Args:
            individual_etas (torch.Tensor): Individual random effects samples. Size: (num_chains, nb_patients, nb_PDU)
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas, used to fetch the design matrices

        Returns:
            torch.Tensor: The individual parameters in gaussian (unconstrained) space. Size: (num_chains, nb_patients, nb_PDU)
        """
        if ind_ids_for_etas is None:
            ind_ids_for_etas = self.patients

        (nb_samples, nb_patients_local, nb_PDU) = individual_etas.shape

        assert (nb_patients_local, nb_PDU) == (
            len(ind_ids_for_etas),
            self.nb_PDU,
        ), f"Wrong shape for individual_etas, expected {len(ind_ids_for_etas), self.nb_PDU}, got {nb_patients_local, nb_PDU}"

        if nb_patients_local == self.nb_patients:
            stacked_design_matrices = self.full_design_matrix
        else:
            list_design_matrices = [
                self.design_matrices[ind_id] for ind_id in ind_ids_for_etas
            ]
            stacked_design_matrices = torch.stack(list_design_matrices).to(device)

        gaussian_params = (
            stacked_design_matrices.expand(nb_samples, -1, -1, -1)
            @ self.population_betas
            + individual_etas
        )

        return gaussian_params

    @torch.compile
    def gaussian_to_physical_params(
        self,
        psi: torch.Tensor,
        log_mi: torch.Tensor,
        ind_ids_for_etas: Optional[list[str | int]] = None,
    ) -> torch.Tensor:
        """Transform gaussian parameters to physical parameters (thetas)

        Args:
            psi (torch.Tensor): Tensor of individual unconstrained parameter values. Size: (num_chains, nb_patients, nb_PDU)
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas

        Returns:
            torch.Tensor: Tensor of individual physical parameter values. Both PDUs and MIs are included. Size: (num_chains, nb_patients, nb_PDU + nb_MI)
        """

        if ind_ids_for_etas is None:
            ind_ids_for_etas = self.patients

        (nb_samples, nb_patients_local, nb_PDU) = psi.shape
        assert nb_patients_local == len(ind_ids_for_etas), nb_PDU == self.nb_PDU

        pdu = torch.zeros_like(psi, device=device)
        # Apply exp transform
        pdu[:, :, self.pdu_transforms["exp"]] = torch.exp(
            psi[:, :, self.pdu_transforms["exp"]]
        )
        # Apply sigmoid transform
        pdu[:, :, self.pdu_transforms["sigmoid"]] = torch.sigmoid(
            psi[:, :, self.pdu_transforms["sigmoid"]]
        )
        # Shift the PDU
        pdu = self.pdu_shift + self.pdu_scale * pdu

        mi = torch.zeros_like(log_mi, device=device)
        # Apply the exp transform
        mi[self.mi_transforms["exp"]] = torch.exp(log_mi[self.mi_transforms["exp"]])
        # Apply sigmoid transform
        mi[self.mi_transforms["sigmoid"]] = torch.sigmoid(
            log_mi[self.mi_transforms["sigmoid"]]
        )
        mi = self.mi_shift + self.mi_scale * mi
        mi = mi.expand(nb_samples, nb_patients_local, -1)

        assert mi.shape == (nb_samples, nb_patients_local, self.nb_MI)
        # Todo: allow for different transformations of gaussian parameters (logit)
        phi = torch.cat((pdu, mi), dim=-1).to(device)

        return phi

    @torch.compile
    def assemble_individual_parameters(
        self,
        physical_params: torch.Tensor,
        ind_ids_for_etas: Optional[list[str | int]] = None,
    ) -> torch.Tensor:
        """Compute individual patient parameters

        Transforms log(MI) (Model intrinsic), betas: log(mu)s & coeffs for covariates and individual random effects (etas) into individual parameters (theta_i), for each set of etas of the list and corresponding design matrix.
        Assumes log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.

        Args:
            physical_params (Tensor): output from self.gaussian_to_physical_params. Transformed MI and PDU params
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas

        Returns:
            torch.Tensor [nb_patients x nb_parameters]: One parameter set for each patient. Dim 0 corresponds to the patients, dim 1 is the parameters
        """
        if ind_ids_for_etas is None:
            ind_ids_for_etas = self.patients

        (nb_samples, nb_patients_local, nb_descriptors) = physical_params.shape

        assert len(ind_ids_for_etas) == nb_patients_local

        if nb_patients_local == self.nb_patients:
            pdk = self.patients_pdk_full
        elif nb_patients_local == 1:
            if self.nb_PDK > 0:
                pdk_list = [self.patients_pdk[ind] for ind in ind_ids_for_etas]
                pdk = torch.stack(pdk_list)
            else:
                pdk = torch.empty((nb_patients_local, 0), device=device)
        else:
            raise ValueError(
                "In assemble_individual_parameters, nb_patients_local should either be 1 or the total number of patients."
            )

        # Assemble the thetas by adding the PDKs
        thetas = torch.cat(
            (
                pdk.expand(nb_samples, -1, -1),
                physical_params,
            ),
            dim=-1,
        )

        assert thetas.shape == (nb_samples, nb_patients_local, self.nb_descriptors)
        return thetas

    @torch.compile
    def struc_model_inputs_from_theta(
        self,
        thetas: torch.Tensor,
        ind_ids_for_etas: Optional[list[str | int]] = None,
    ) -> torch.Tensor:
        """Return model inputs for all patients

        Args:
            thetas (torch.Tensor): Parameter values per patient. Theta may contain one parameter per chain
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas

        Returns:
            torch.Tensor: the full inputs required to simulate all patients on all time steps
        """
        if ind_ids_for_etas is None:
            ind_ids_for_etas = self.patients

        # If theta was not provided in batch format, add a dummy dimension
        inputs = thetas

        if inputs.dim() < 3:
            inputs = inputs.unsqueeze(0)
        num_chains_local = inputs.shape[0]
        nb_patients_local = len(ind_ids_for_etas)
        assert inputs.shape == (
            num_chains_local,
            nb_patients_local,
            self.nb_descriptors,
        ), f"Wrong shape in theta to structural model inputs: {inputs.shape}"

        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute patient predictions without an associated observations data frame."
            )

        # Order the columns of theta, and add a repeat dimension to cover time steps
        theta_expanded = (
            inputs[:, :, self.model_input_to_descriptor]
            .unsqueeze(-2)
            .expand((-1, -1, self.nb_global_time_steps, -1))
        )

        # Repeat global time steps as much as there are patients in ind_ids_for_etas
        if nb_patients_local == self.nb_patients:
            time_steps_expanded = self.global_time_steps_expanded
        elif nb_patients_local == 1:
            time_steps_expanded = (
                self.global_time_steps.unsqueeze(0)
                .unsqueeze(-1)
                .repeat((nb_patients_local, 1, 1))
            )
        else:
            raise ValueError(
                "In struc_model_inputs_from_theta, nb_patients_local should either be 1 or the total number of patients."
            )

        full_inputs = torch.cat(
            (
                theta_expanded,
                time_steps_expanded.expand(num_chains_local, -1, -1, -1),
            ),
            dim=-1,
        )

        assert full_inputs.shape == (
            num_chains_local,
            nb_patients_local,
            self.nb_global_time_steps,
            self.nb_descriptors + 1,
        )

        return full_inputs

    def predict_outputs_from_theta(
        self,
        thetas: torch.Tensor,
        ind_ids_for_etas: Optional[list[str | int]] = None,
        model_inputs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model predictions for all patients

        This function carries the number of chains in the batch dimension (0).

        Args:
            thetas (torch.Tensor): Parameter values per patient. Size (num_chains | 1, nb_patients, nb_time_steps, nb_descriptors + 1)
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas

        Returns:
            list[torch.Tensor]: a tensor of predictions for each patient
        """
        if ind_ids_for_etas is None:
            ind_ids_for_etas = self.patients

        nb_patients_local = len(ind_ids_for_etas)

        if model_inputs is None:
            model_inputs = self.struc_model_inputs_from_theta(thetas)

        if nb_patients_local == self.nb_patients:
            prediction_index = self.prediction_index
            chunk_sizes = self.chunk_sizes
        elif nb_patients_local == 1:
            ind = ind_ids_for_etas[0]
            prediction_index = (
                torch.zeros_like(
                    self.observations_tensors[ind]["p_index_repeated"]
                ),  # only one patient, index 0
                self.observations_tensors[ind]["time_step_indices"],
                self.observations_tensors[ind]["tasks_indices"],
            )
            chunk_sizes = [len(prediction_index[0])]
        else:
            raise ValueError(
                "In predict_outputs_from_theta, nb_patients_local should either be 1 or the total number of patients."
            )

        # Shape: (nb_chains | 1, nb_patients, nb_global_time_steps, nb_descriptors + 1)
        pred_mean, pred_var = self.structural_model.simulate(
            model_inputs,
            prediction_index,
            chunk_sizes,
        )

        return pred_mean, pred_var

    def add_residual_error(self, outputs: torch.Tensor) -> torch.Tensor:
        res_var = self.residual_var.index_select(0, self.full_output_indices)
        noise = torch.distributions.Normal(
            torch.zeros(self.full_output_indices.shape[0], device=device), res_var
        ).sample()
        if self.error_model_type == "additive":
            new_out = outputs + noise
        elif self.error_model_type == "proportional":
            new_out = outputs * noise
        else:
            raise ValueError(f"Non-implemented error model {self.error_model_type}")
        return new_out

    def outputs_to_df(self, outputs: torch.Tensor) -> pd.DataFrame:
        """Transform the NLME model outputs to a data frame in order to compare with observed data

        Args:
            outputs (torch.Tensor): Outputs from `self.predict_outputs_from_theta`

        Returns:
            pd.DataFrame: A data frame containing the following columns
            - `id`
            - `output_name`
            - `protocol_arm`
            - `time`
            - `predicted_value`
        """
        assert (
            outputs.shape[0] == 1
        ), "Batched outputs cannot be converted to a dataframe."
        outputs_list = outputs.squeeze(0).cpu().split(self.chunk_sizes)
        df_list = []
        for ind_idx, ind in enumerate(self.patients):
            time_steps = self.observations_tensors[ind]["time_steps"]
            task_list = self.observations_tensors[ind]["tasks_indices"]
            temp_df = pd.DataFrame(
                {
                    "time": time_steps.cpu().numpy(),
                    "id": ind,
                    "task_index": task_list.cpu(),
                    "predicted_value": outputs_list[ind_idx].numpy(),
                }
            )
            temp_df["output_name"] = temp_df["task_index"].apply(
                lambda t: self.outputs_names[
                    self.structural_model.task_idx_to_output_idx[t]
                ]
            )
            temp_df["protocol_arm"] = temp_df["task_index"].apply(
                lambda t: self.structural_model.task_idx_to_protocol[t]
            )
            df_list.append(temp_df)
        out_df = pd.concat(df_list)
        out_df = out_df.drop(columns=["task_index"])
        return out_df

    def _log_prior_etas(self, etas: torch.Tensor) -> torch.Tensor:
        """Compute log-prior of random effect samples (etas)

        Args:
            etas (torch.Tensor): Individual samples, assuming eta_i ~ N(0, Omega)

        Returns:
            torch.Tensor [nb_eta_i x nb_PDU]: Values of log-prior, computed according to:

            P(eta) = (1/sqrt((2pi)^k * |Omega|)) * exp(-0.5 * eta.T * omega.inv * eta)
            log P(eta) = -0.5 * (k * log(2pi) + log|Omega| + eta.T * omega.inv * eta)

        """

        log_priors: torch.Tensor = self.eta_distribution.log_prob(etas).to(device)
        return log_priors

    def log_posterior_etas(
        self,
        etas: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute the log-posterior of a list of random effects

        Args:
            etas (torch.Tensor): Random effects samples

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], DataFrame]:
            - log-posterior likelihood of etas
            - gaussian parameters corresponding to the etas
            - list of simulated values for each patient on each chain

        """
        nb_chains_local = etas.shape[0]
        assert etas.shape == (nb_chains_local, self.nb_patients, self.nb_PDU)
        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute log-posterior without an associated observations data frame."
            )
        gaussian_params_chains = self.etas_to_gaussian_params(etas)
        physical_params_chains = self.gaussian_to_physical_params(
            gaussian_params_chains, self.log_MI
        )
        # Get individual parameters in a tensor
        theta_chains = self.assemble_individual_parameters(physical_params_chains)

        # Run the structural model
        full_pred, full_var = self.predict_outputs_from_theta(theta_chains)
        # var_list = full_var.split(self.chunk_sizes)
        # Validate the variance magnitude
        # warnings = self.variance_level_check(var_list, self.pred_var_threshold)
        # flagged_patients = [individual_params[i, :].tolist() for i in warnings]

        # calculate log-prior of the random samples
        log_priors: torch.Tensor = self._log_prior_etas(etas)
        assert log_priors.shape == (nb_chains_local, self.nb_patients)

        # Compute the log likelihood of observations
        log_likelihood_observations = cast(
            torch.Tensor, self.log_likelihood_observation(full_pred)
        )
        assert log_likelihood_observations.shape == (nb_chains_local, self.nb_patients)

        log_posterior = log_likelihood_observations + log_priors
        assert log_posterior.shape == (nb_chains_local, self.nb_patients)
        return (
            log_posterior,
            gaussian_params_chains,
            full_pred,
        )

    @torch.compile
    def calculate_residuals(
        self,
        observed_data: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates residuals based on the error model

        Args:
            observed_data: Tensor of observations
            predictions: Tensor of predictions

        Returns:
            torch.Tensor: a tensor of residual values
        """
        assert (
            predictions.shape == observed_data.shape
        ), f"Non-matching shapes in `calculate_residuals`: {predictions.shape=}, {observed_data.shape=}"

        if self.error_model_type == "additive":
            return observed_data - predictions
        elif self.error_model_type == "proportional":
            return (observed_data - predictions) / predictions
        else:
            raise ValueError("Unsupported error model type.")

    @torch.compile
    def sum_sq_residuals_chains(self, prediction: torch.Tensor) -> torch.Tensor:
        """Compute the sum of squared residuals for current predictions on all chains

        Args:
            prediction (torch.Tensor): Predictions for all patients and all chains, Size (num_chains, nb_patients, nb_tot_obs)

        Returns:
            torch.Tensor: The sum of squared residuals for each output. Size (nb_outputs)
        """
        assert prediction.shape == (self.num_chains, self.nb_tot_obs)
        sq_residuals = torch.square(
            self.calculate_residuals(
                self.full_obs_data.expand(self.num_chains, -1), prediction
            )
        )
        sum_residuals_per_chain = torch.zeros(
            self.num_chains, self.nb_outputs, device=device
        )
        sum_residuals_per_chain.scatter_add_(
            1, self.full_output_indices.expand(self.num_chains, -1), sq_residuals
        )
        sum_residuals = sum_residuals_per_chain.mean(dim=0)
        return sum_residuals

    @torch.compile
    def compute_error_variance(
        self, predictions: torch.Tensor, output_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if output_indices is None:
            output_indices = self.full_output_indices

        num_chains_local = predictions.shape[0]

        res_error_var = self.residual_var.expand(num_chains_local, -1).index_select(
            1, output_indices
        )

        # Log-likelihood of normal distribution
        if self.error_model_type == "additive":
            variance = res_error_var
        elif self.error_model_type == "proportional":
            variance = res_error_var * torch.square(predictions)
        else:
            raise ValueError("Non supported error type.")
        return variance

    def compute_iwres(
        self,
    ) -> dict:
        """Compute Individual Weighted RESiduals, following the formula :

        IWRES_(ij) = ( y_ij - f(t_ij, psi_i) ) / g(t_ij, psi_i)

        Returns:
            dict: IWRES with patientId as key, with IWRES and timesteps for each patient
        """

        # Avoid computing if already computed residuals
        if hasattr(self, "iwres"):
            return self.iwres

        # Gather observations tensor
        observed_tensor = self.full_obs_data.view(1, -1)

        # Gather predictions tensor
        ebe_physical = self.compute_ebe()
        (nb_samples, nb_patients_local, nb_descriptors) = ebe_physical.shape

        # Assemble the thetas by adding the PDKs
        theta = torch.cat(
            (
                self.patients_pdk_full.expand(nb_samples, -1, -1),
                ebe_physical,
            ),
            dim=-1,
        )
        simulated_tensor, _ = self.predict_outputs_from_theta(theta)

        # Compute residuals and variance
        residuals = self.calculate_residuals(observed_tensor, simulated_tensor)
        variance = self.compute_error_variance(simulated_tensor)

        iwres = residuals / torch.sqrt(variance)

        iwres_list = torch.split(iwres, self.chunk_sizes, dim=1)
        iwres_results = {}

        # Separate IWRES per patient in a dict
        for i, patient_id in enumerate(self.patients):
            iwres_patient = iwres_list[i]
            time_steps_patient = self.observations_tensors[patient_id]["time_steps"]
            iwres_results.update(
                {
                    patient_id: {
                        "iwres": iwres_patient.squeeze().cpu().numpy(),
                        "time": time_steps_patient.cpu().numpy(),
                    }
                }
            )

        self.iwres = iwres_results
        return iwres_results

    def compute_pwres(self, num_samples: int) -> dict:
        """Compute Population Weighted RESiduals, following the formula :

        PWRES_i = V_i^(-1/2) (y_i - E(f(t_ij, psi_i))

        Returns:
            dict: PWRES with patientId as key, with PWRES and timesteps for each patient
        """

        # Avoid computing if already computed residuals
        if hasattr(self, "pwres"):
            return self.pwres

        # Sample new etas, in order to approximate mean E(y_i) and variance V_i
        mc_etas = self.sample_etas(num_samples, self.nb_patients)

        # Compute gaussian parameters
        mc_gaussian = self.etas_to_gaussian_params(mc_etas)

        # Compute physical parameters
        mc_physical = self.gaussian_to_physical_params(mc_gaussian, self.log_MI)

        # Assemble individual parameters
        mc_thetas = self.assemble_individual_parameters(mc_physical)

        # Simulate outputs
        simulated_tensor, _ = self.predict_outputs_from_theta(mc_thetas)

        # Compute PWRES per patient

        # list_sim_per_patient shape: n_patients items of shape nb_samples * n_obs_patient
        list_sim_per_patient = torch.split(simulated_tensor, self.chunk_sizes, dim=1)

        # Initialize a dictionnary gathering PWRES and timesteps for each patient, to plot later
        pwres_results = {}

        for i, patient_id in enumerate(self.patients):

            # patient_data shape: nb_samples * n_obs_patient
            patient_data = list_sim_per_patient[i]

            # mean_patient shape: nb_samples * n_obs_patient -> n_obs_patient
            mean_patient = patient_data.mean(dim=0)

            # variance_patient shape: n_obs_patient * n_obs_patient
            variance_patient = torch.cov(patient_data.T)

            # obs_patient shape: n_obs_patient
            obs_patient = self.observations_tensors[patient_id]["observations"]
            time_steps_patient = self.observations_tensors[patient_id]["time_steps"]

            # Transform residual into a column
            residual = (obs_patient - mean_patient).unsqueeze(-1)

            # Compute V^-1/2 with Cholesky factorization, adding a jitter for stability purposes
            jitter = torch.eye(variance_patient.size(0)) * 1e-6
            L = torch.linalg.cholesky(variance_patient + jitter)

            # Compute patient PWRES and add them to dictionnary
            pwres_patient = L @ residual
            pwres_results.update(
                {
                    patient_id: {
                        "pwres": pwres_patient.squeeze(-1).cpu().numpy(),
                        "time": time_steps_patient.cpu().numpy(),
                    }
                }
            )

        self.pwres = pwres_results
        return pwres_results

    def compute_npde(self, num_samples: int = 100) -> dict:

        # Avoid computing if already computed residuals
        if hasattr(self, "npde"):
            return self.npde

        # Sample new etas
        mc_etas = self.sample_etas(num_samples, self.nb_patients)

        # Compute gaussian parameters
        mc_gaussian = self.etas_to_gaussian_params(mc_etas)

        # Compute physical parameters
        mc_physical = self.gaussian_to_physical_params(mc_gaussian, self.log_MI)

        # Assemble individual parameters
        mc_thetas = self.assemble_individual_parameters(mc_physical)

        # Simulate outputs
        simulated_tensor, _ = self.predict_outputs_from_theta(mc_thetas)

        # Expand observation tensor to match simulated tensor
        observed_tensor = self.full_obs_data.expand(num_samples, -1)

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
        npde_per_patient = torch.split(npde, self.chunk_sizes, dim=0)

        npde_results = {}

        for i, patient_id in enumerate(self.patients):

            npde_patient = npde_per_patient[i]
            time_steps_patient = self.observations_tensors[patient_id]["time_steps"]
            npde_results.update(
                {
                    patient_id: {
                        "npde": npde_patient.squeeze(-1).cpu().numpy(),
                        "time": time_steps_patient.cpu().numpy(),
                    }
                }
            )

        self.npde = npde_results
        return npde_results

    @torch.compile
    def log_likelihood_observation(
        self,
        predictions: torch.Tensor,
        ind_ids_for_etas: Optional[list[str | int]] = None,
    ) -> torch.Tensor | float:
        """Compute the log-likelihood of observations across all chains

        Args:
            predictions (torch.Tensor): Tensor of predictions. Size (num_chains, nb_total_obs, 1)

        Returns:
            torch.Tensor: Log likelihood per patient and per chain. Size (num_chains, nb_patient)
        """
        num_chains_local = predictions.shape[0]

        if ind_ids_for_etas is None:
            ind_ids_for_etas = self.patients

        nb_patients_local = len(ind_ids_for_etas)
        if nb_patients_local == 1:
            patient_id = ind_ids_for_etas[0]
            observations = self.observations_tensors[patient_id]["observations"]
            output_indices = self.observations_tensors[patient_id]["outputs_indices"]
        elif nb_patients_local == self.nb_patients:
            observations = self.full_obs_data
            output_indices = self.full_output_indices
        else:
            raise ValueError(
                "In log_likelihood_observation, nb_patients_local should either be 1 or the total number of patients."
            )

        residuals: torch.Tensor = self.calculate_residuals(
            observations.expand(num_chains_local, -1), predictions
        )

        res_error_var = self.residual_var.expand(num_chains_local, -1).index_select(
            1, output_indices
        )

        assert residuals.shape == res_error_var.shape

        # Log-likelihood of normal distribution
        variance = self.compute_error_variance(predictions, output_indices)

        log_lik_full = -0.5 * (
            torch.log(2 * torch.pi * variance) + (residuals**2 / variance)
        )

        # If only one patient, return likelihood as a float
        if nb_patients_local == 1:
            log_lik_patient = log_lik_full.sum().item()
            return log_lik_patient

        # If several patients, create likelihood tensor
        else:
            log_lik_per_patient = torch.zeros(
                (num_chains_local, self.nb_patients), device=device
            )
            log_lik_per_patient.scatter_add_(
                1,
                self.observation_to_patient_index.expand(num_chains_local, -1),
                log_lik_full,
            )
            return log_lik_per_patient

    def mh_step(
        self,
        current_etas: torch.Tensor,
        current_log_prob: torch.Tensor,
        current_pred: torch.Tensor,
        current_gaussian_params: torch.Tensor,
        step_size: float,
        learning_rate: float,
        target_acceptance_rate: float = 0.234,
        verbose: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        """Perform one step of a Metropolis-Hastings transition kernel

        Args:
            current_etas (torch.Tensor): values of the individual random effects for all patients, on all chains. Size (num_chains, nb_patients, nb_PDU)
            current_log_prob (torch.Tensor): log posterior likelihood of current random effects. Size (num_chains, nb_patients)
            current_pred (torch.Tensor): associated model predictions with current random effects. Size (num_chains, nb_obs_total)
            step_size (torch.Tensor): current value of MH step size,
            learning_rate (float): current learning rate (defined by the optimization algorithm)
            target_acceptance_rate (float, optional): Target for the MCMC acceptance rate. Defaults to 0.234 [1].

            [1] Sherlock C. Optimal Scaling of the Random Walk Metropolis: General Criteria for the 0.234 Acceptance Rule. Journal of Applied Probability. 2013;50(1):1-15. doi:10.1239/jap/1363784420

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[int | str, torch.Tensor], torch.Tensor, float]:
            - updated individual random effects
            - updated log posterior likelihood
            - updated predictions, for each patient of the observation data set
            - updated thetas
            - updated values of log PDUs
            - updated step size
            - a dict of warnings for all patients with predictive variance above threshold
        """
        # Propose new etas
        proposal_noise = (
            torch.randn_like(current_etas, device=device) @ self.omega_pop_lower_chol
        )
        proposal_etas = current_etas + step_size * proposal_noise
        # Compute their log posterior likelihood
        (
            proposal_log_prob,
            proposal_gaussian_params,
            proposal_pred,
        ) = self.log_posterior_etas(proposal_etas)

        assert proposal_log_prob.shape == current_log_prob.shape

        # Define acceptance masks
        deltas: torch.Tensor = proposal_log_prob - current_log_prob
        log_u: torch.Tensor = torch.log(torch.rand_like(deltas, device=device))
        accept_mask: torch.Tensor = log_u < deltas
        accept_mask_parameters = accept_mask.unsqueeze(-1).expand(
            -1, -1, current_etas.shape[-1]
        )
        accept_mask_predictions = accept_mask.index_select(
            1, self.observation_to_patient_index
        )
        # Accept the different variables using the masks
        new_etas = torch.where(accept_mask_parameters, proposal_etas, current_etas).to(
            device
        )
        new_gaussian_params = torch.where(
            accept_mask_parameters, proposal_gaussian_params, current_gaussian_params
        ).to(device)
        new_log_prob = torch.where(accept_mask, proposal_log_prob, current_log_prob).to(
            device
        )
        # Compute the complete likelihood by averaging over chains and summing over patients
        new_complete_likelihood = -2 * new_log_prob.mean(dim=0).sum(dim=0)
        new_pred = torch.where(accept_mask_predictions, proposal_pred, current_pred).to(
            device
        )
        new_acceptance_rate: float = accept_mask.cpu().float().mean().mean().item()
        if verbose:
            print(f"  Acceptance rate: {new_acceptance_rate:.2f}")
        new_step_size: float = step_size * np.exp(
            learning_rate * (new_acceptance_rate - target_acceptance_rate)
        )
        return (
            new_etas,
            new_log_prob,
            new_complete_likelihood,
            new_pred,
            new_gaussian_params,
            new_step_size,
        )

    def map_estimates_descriptors(self) -> pd.DataFrame:

        if not hasattr(self, "cond_dist_samples"):
            print(
                "Conditional distribution has not been run before. Running it with 1000 samples by default...",
                "If more samples needed, run sample_conditional_distribution(nb_samples) before.",
            )
            self.sample_conditional_distribution(1000)

        ebe_physical = self.compute_ebe()

        (nb_samples, nb_patients, nb_descriptors) = ebe_physical.shape

        # Assemble the thetas by adding the PDKs
        theta = torch.cat(
            (
                self.patients_pdk_full.expand(nb_samples, -1, -1),
                ebe_physical,
            ),
            dim=-1,
        )

        if theta is None:
            raise ValueError("No estimation available yet. Run the algorithm first.")

        theta = theta.squeeze()

        assert theta.shape == (
            self.nb_patients,
            self.nb_descriptors,
        ), f"Unexpected dimensions of MAP estimates: {theta.shape}"

        map_per_patient = pd.DataFrame(
            data=theta.cpu().numpy(), columns=self.descriptors
        )
        id_column = self.patients
        map_per_patient["id"] = id_column

        return map_per_patient

    def map_estimates_predictions(self) -> pd.DataFrame:

        if not hasattr(self, "cond_dist_samples"):
            print(
                "Conditional distribution has not been run before. Running it with 1000 samples by default...",
                "If more samples needed, run sample_conditional_distribution(nb_samples) before.",
            )
            self.sample_conditional_distribution(1000)

        ebe_physical = self.compute_ebe()

        (nb_samples, nb_patients, nb_descriptors) = ebe_physical.shape

        # Assemble the thetas by adding the PDKs
        theta = torch.cat(
            (
                self.patients_pdk_full.expand(nb_samples, -1, -1),
                ebe_physical,
            ),
            dim=-1,
        )

        if theta is None:
            raise ValueError(
                "No estimation available yet. Run the optimization algorithm first."
            )
        simulated_tensor, _ = self.predict_outputs_from_theta(theta)
        simulated_df = self.outputs_to_df(simulated_tensor)
        return simulated_df

    def population_predictions(self) -> pd.DataFrame:
        """
        Returns simulated dataframe for null etas, i.e. without individual random effects but keeping the covariables effects.
        """

        # Sample new etas, in order to approximate mean E(y_i) and variance V_i
        null_etas = torch.zeros([1, self.nb_patients, self.nb_PDU])

        # Compute gaussian parameters
        physical_params = self.etas_to_gaussian_params(null_etas)

        # Compute physical parameters
        physical_params = self.gaussian_to_physical_params(physical_params, self.log_MI)

        # Assemble individual parameters
        thetas = self.assemble_individual_parameters(physical_params)

        # Simulate outputs
        simulated_tensor, _ = self.predict_outputs_from_theta(thetas)

        simulated_df = self.outputs_to_df(simulated_tensor)
        return simulated_df

    def variance_level_check(
        self, var_list: tuple[torch.Tensor, ...], threshold: float
    ) -> list:
        warnings = []
        for i in range(self.nb_patients):
            var, _ = var_list[i].cpu().max(0)
            if var > threshold:
                warnings.append(i)
        return warnings

    def sample_conditional_distribution(self, nb_samples: int):
        """
        Returns: cond_dist_samples: dim(nb_samples, nb_patients, nb_PDU)
        """

        if (
            hasattr(self, "cond_dist_samples")
            and nb_samples <= self.cond_dist_samples.shape[0]
        ):
            return self.cond_dist_samples

        init_etas = self.sample_etas(1, self.nb_patients)
        (current_log_prob, current_gaussian_params, current_pred) = (
            self.log_posterior_etas(init_etas)
        )
        current_etas = init_etas
        sample_list = []
        nb_burn_in = 25
        for i in range(nb_burn_in + nb_samples):
            (
                current_etas,
                current_log_prob,
                _,
                current_pred,
                current_gaussian_params,
                _,
            ) = self.mh_step(
                current_etas,
                current_log_prob,
                current_pred,
                current_gaussian_params,
                step_size=0.1,
                learning_rate=0.0,
            )

            if i >= nb_burn_in:
                sample_list.append(current_etas)

        sample_tensor = torch.stack(sample_list)
        etas_samples = torch.squeeze(sample_tensor, 1)
        self.cond_dist_samples = etas_samples
        return self.cond_dist_samples

    def compute_ebe(self):
        """
        Returns: ebe_estimates: dim(nb_patients, nb_PDU)
        """

        if hasattr(self, "ebe_estimates"):
            return self.ebe_estimates

        if not hasattr(self, "cond_dist_samples"):
            print(
                "Conditional distribution has not been run before. Running it with 1000 samples by default...",
                "If more samples needed, run sample_conditional_distribution(nb_samples).",
            )
            samples = self.sample_conditional_distribution(1000)

        else:
            samples = self.cond_dist_samples

        if smoke_test:
            max_iter = 1
        else:
            max_iter = 5000

        # Taking conditional distribution samples means as a starting point for optimization
        init_samples = samples.mean(dim=0)

        nb_PDU = self.nb_PDU
        ebe_etas = torch.zeros((self.nb_patients, self.nb_PDU))
        for i in tqdm(range(self.nb_patients)):

            def objective_function(eta_array):
                eta_tensor = torch.from_numpy(eta_array).float().view(1, 1, nb_PDU)
                with torch.no_grad():
                    log_post = self.log_posterior_etas_single(eta_tensor, i)
                return -log_post

            x0 = init_samples[i].numpy()

            res = minimize(
                objective_function,
                x0,
                method="Nelder-Mead",
                tol=1e-4,
                options={"maxiter": max_iter},
            )
            ebe_etas[i] = torch.from_numpy(res.x)

        ebe_etas = ebe_etas.expand(1, -1, -1)
        ebe_gaussian = self.etas_to_gaussian_params(ebe_etas)
        ebe_physical = self.gaussian_to_physical_params(ebe_gaussian, self.log_MI)
        self.ebe_estimates = ebe_physical
        return ebe_physical

    def log_posterior_etas_single(self, etas: torch.Tensor, patient_ind: int) -> float:
        """
        Args:
            etas: random effects sample for one patient, dim (1, 1, nb_PDU)
            patient_ind: index of a patient in self.patients

        Returns:
            log_posterior: value of log-posterior distribution for given etas and patient
        """

        assert etas.shape == (1, 1, self.nb_PDU)

        patient_id = self.patients[patient_ind]

        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute log-posterior without an associated observations data frame."
            )

        # Proceed from etas to model inputs
        gaussian_params = self.etas_to_gaussian_params(etas, [patient_id])
        physical_params = self.gaussian_to_physical_params(
            gaussian_params, self.log_MI, [patient_id]
        )
        theta = self.assemble_individual_parameters(physical_params, [patient_id])
        full_inputs = self.struc_model_inputs_from_theta(theta, [patient_id])

        predictions, _ = self.predict_outputs_from_theta(
            theta, [patient_id], full_inputs
        )

        log_priors = self._log_prior_etas(etas)
        log_lik_patient = self.log_likelihood_observation(predictions, [patient_id])

        log_posterior = cast(float, log_lik_patient + log_priors)

        return log_posterior
