import torch
from typing import Union, Optional
import pandas as pd
import numpy as np

from .structural_model import StructuralModel
from .utils import device


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
        """
        self.structural_model: StructuralModel = structural_model
        self.pred_var_threshold = pred_var_threshold

        self.MI_names: list[str] = list(init_log_MI.keys())
        self.nb_MI: int = len(self.MI_names)
        self.initial_log_MI = torch.tensor([val for _, val in init_log_MI.items()]).to(
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
                f"Non-matching descriptor set and structural model parameter set:\n{set(self.PDK_names + self.PDU_names + self.MI_names)}\n{set(self.structural_model.parameter_names)}"
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
        self.init_omega = torch.diag(
            torch.as_tensor(
                [float(init_PDU[pdu]["sd"]) for pdu in self.PDU_names], device=device
            )
        )

        # Assemble the list of design matrices from the covariance structure
        self.design_matrices = self._create_all_design_matrices()
        # Store the full design matrix on the device
        self.full_design_matrix = torch.stack(
            [self.design_matrices[p] for p in self.patients]
        ).to(device)

        # Initiate the nlme parameters
        self.log_MI = self.initial_log_MI
        self.population_betas = self.initial_betas
        self.omega_pop = self.init_omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop).to(device)
        self.residual_var = self.init_res_var
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU, device=device),
            covariance_matrix=self.omega_pop,
        )
        self.current_eta_samples = self.sample_individual_etas()
        self.current_map_estimates = self.individual_parameters(
            self.current_eta_samples
        )

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

        # Build the full data set tensors
        self.full_obs_data = torch.cat(
            [self.observations_tensors[p]["observations"] for p in self.patients]
        ).to(device)
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
        """Update the covariance matrix of the NLME model."""
        assert (
            self.omega_pop.shape == omega.shape
        ), f"Wrong omega shape: {omega.shape}, expected: {self.omega_pop.shape}"
        self.omega_pop = omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop).to(device)
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU, device=device),
            covariance_matrix=self.omega_pop,
        )

    def update_res_var(self, residual_var: torch.Tensor) -> None:
        """Update the residual variance of the NLME model, while ensuring it remains positive."""
        assert (
            self.residual_var.shape == residual_var.shape
        ), f"Wrong res var shape: {residual_var.shape}, expected: {self.residual_var.shape}"
        self.residual_var = residual_var.clamp(min=1e-6)

    def update_betas(self, betas: torch.Tensor) -> None:
        """Update the betas of the NLME model."""
        assert (
            self.population_betas.shape == betas.shape
        ), f"Wrong beta shape: {betas.shape}, expected: {self.population_betas.shape}"
        self.population_betas = betas

    def update_log_mi(self, log_MI: torch.Tensor) -> None:
        """Update the model intrinsic parameter values of the NLME model."""
        assert (
            self.log_MI.shape == log_MI.shape
        ), f"Wrong MI shape: {log_MI.shape}, expected: {self.log_MI.shape}"
        self.log_MI = log_MI

    def update_eta_samples(self, eta: torch.Tensor) -> None:
        """Update the model current individual random effect samples."""
        assert (
            self.current_eta_samples.shape == eta.shape
        ), f"Wrong individual samples shape: {eta.shape}, expected: {self.current_eta_samples.shape}"
        self.current_eta_samples = eta

    def update_map_estimates(self, theta: torch.Tensor) -> None:
        """Update the model current maximum a posteriori estimates."""
        assert (
            self.current_map_estimates.shape == theta.shape
        ), f"Wrong individual parameters shape: {theta.shape}, expected: {self.current_map_estimates.shape}"
        self.current_map_estimates = theta

    @torch.compile
    def sample_individual_etas(self) -> torch.Tensor:
        """Sample individual random effects from the current estimate of Omega

        Returns:
            torch.Tensor (size nb_patients x nb_PDUs): individual random effects for all patients in the population
        """
        etas_dist = self.eta_distribution.expand([self.nb_patients])
        etas = etas_dist.sample()
        return etas

    @torch.compile
    def individual_parameters(
        self,
        individual_etas: torch.Tensor,
    ) -> torch.Tensor:
        """Compute individual patient parameters

        Transforms log(MI) (Model intrinsic), betas: log(mu)s & coeffs for covariates and individual random effects (etas) into individual parameters (theta_i), for each set of etas of the list and corresponding design matrix.
        Assumes log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.

        Args:
            individual_etas (torch.Tensor): one set of sampled random effects for each patient
            ind_ids_for_etas (list[Union[str, int]]): list of individual ids corresponding to the sampled etas, used to fetch the design matrices
        Returns:
            torch.Tensor [nb_patients x nb_parameters]: One parameter set for each patient. Dim 0 corresponds to the patients, dim 1 is the parameters
        """
        # Compute the inidividual PDU
        log_thetas_PDU = (
            self.full_design_matrix @ self.population_betas + individual_etas
        )
        # Gather the MI values, and expand them (same for each patient)
        log_MI_expanded = self.log_MI.unsqueeze(0).repeat(self.nb_patients, 1)

        # list the PDK values for each patient, and assemble them in a tensor
        # This step is crucial: we need to ensure the parameters are stored in the correct order
        # PDK, PDU, MI
        thetas = torch.cat(
            (
                self.patients_pdk_full,
                torch.exp(torch.cat((log_thetas_PDU, log_MI_expanded), dim=1)),
            ),
            dim=1,
        )
        return thetas

    @torch.compile
    def struc_model_inputs_from_theta(self, thetas: torch.Tensor) -> torch.Tensor:
        """Return model inputs for all patients

        Args:
            thetas (torch.Tensor): Parameter values per patient (one by row)

        Returns:
            torch.Tensor: the full inputs required to simulate all patients on all time steps
        """

        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute patient predictions without an associated observations data frame."
            )
        # Order the columns of theta, and add a repeat dimension to cover time steps
        theta_expanded = (
            thetas[:, self.model_input_to_descriptor]
            .unsqueeze(1)
            .repeat((1, self.nb_global_time_steps, 1))
        )
        full_inputs = torch.cat(
            (
                theta_expanded,
                self.global_time_steps_expanded,
            ),
            dim=2,
        )
        assert full_inputs.shape == (
            self.nb_patients,
            self.nb_global_time_steps,
            self.nb_descriptors + 1,
        )
        return full_inputs

    def predict_outputs_from_theta(
        self, thetas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model predictions for all patients

        Args:
            thetas (torch.Tensor): Parameter values per patient (one by row)

        Returns:
            list[torch.Tensor]: a tensor of predictions for each patient
        """
        model_inputs = self.struc_model_inputs_from_theta(thetas)
        # shape: [nb_patients, nb_time_steps, nb_params + 1]
        pred_mean, pred_var = self.structural_model.simulate(
            model_inputs,
            self.prediction_index,
            self.chunk_sizes,
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
        outputs_list = outputs.cpu().split(self.chunk_sizes)
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
        torch.Tensor,
        list,
    ]:
        """Compute the log-posterior of a list of random effects

        Args:
            etas (torch.Tensor): Random effects samples

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], DataFrame]:
            - log-posterior likelihood of etas
            - current thetas
            - log values of current pdu estimation (useful for SAEM)
            - list of simulated values for each patient

        """
        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute log-posterior without an associated observations data frame."
            )
        # Get individual parameters in a tensor
        individual_params: torch.Tensor = self.individual_parameters(
            individual_etas=etas,
        )
        # Run the surrogate model
        full_pred, full_var = self.predict_outputs_from_theta(individual_params)
        var_list = full_var.split(self.chunk_sizes)
        # Validate the variance magnitude
        warnings = self.variance_level_check(var_list, self.pred_var_threshold)
        flagged_patients = [individual_params[i, :].tolist() for i in warnings]

        # calculate log-prior of the random samples
        log_priors: torch.Tensor = self._log_prior_etas(etas)

        # group by individual and calculate log-likelihood for each
        log_likelihood_observations = self.log_likelihood_observation(full_pred)

        log_posterior = log_likelihood_observations + log_priors
        current_log_pdu = torch.log(
            individual_params[:, self.nb_PDK : self.nb_PDK + self.nb_PDU]
        )
        return (
            log_posterior,
            individual_params,
            current_log_pdu,
            full_pred,
            flagged_patients,
        )

    @torch.compile
    def calculate_residuals(
        self, observed_data: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        """Calculates residuals based on the error model for a single patient

        Args:
            observed_data: Tensor of observations
            predictions: Tensor of predictions

        Returns:
            torch.Tensor: a tensor of residual values
        """
        if self.error_model_type == "additive":
            return observed_data - predictions
        elif self.error_model_type == "proportional":
            return (observed_data - predictions) / predictions
        else:
            raise ValueError("Unsupported error model type.")

    @torch.compile
    def sum_sq_residuals(self, prediction: torch.Tensor) -> torch.Tensor:
        sq_residuals = torch.square(
            self.calculate_residuals(self.full_obs_data, prediction)
        )
        sum_residuals = torch.zeros(self.nb_outputs, device=device)
        sum_residuals.scatter_add_(0, self.full_output_indices, sq_residuals)
        return sum_residuals

    @torch.compile
    def log_likelihood_observation(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the log-likelihood of observations given predictions and error model, assuming errors follow N(0,sqrt(residual_error_var))
        observed_data: torch.Tensor of observations for one individual
        predictions: torch.Tensor of predictions for one individual organized in the same way as observed_data
        residual_error_var: torch.Tensor of the error for each output, dim: [nb_outputs]
        """
        residuals: torch.Tensor = self.calculate_residuals(
            self.full_obs_data, predictions
        )
        res_error_var = self.residual_var.index_select(0, self.full_output_indices)
        # Log-likelihood of normal distribution
        if self.error_model_type == "additive":
            log_lik_full = -0.5 * (
                torch.log(2 * torch.pi * res_error_var) + (residuals**2 / res_error_var)
            )
        elif self.error_model_type == "proportional":
            variance = res_error_var * torch.square(predictions)
            log_lik_full = -0.5 * (
                torch.log(2 * torch.pi * variance) + (residuals**2 / variance)
            )
        else:
            raise ValueError("Non supported error type.")
        log_lik_per_patient = torch.zeros(self.nb_patients, device=device)
        log_lik_per_patient.scatter_add_(
            0, self.observation_to_patient_index, log_lik_full
        )
        return log_lik_per_patient

    def mh_step(
        self,
        current_etas: torch.Tensor,
        current_log_prob: torch.Tensor,
        current_pred: torch.Tensor,
        current_pdu: torch.Tensor,
        current_thetas: torch.Tensor,
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
        torch.Tensor,
        float,
        list,
    ]:
        """Perform one step of a Metropolis-Hastings transition kernel

        Args:
            current_etas (torch.Tensor): values of the individual random effects for all patients
            current_log_prob (torch.Tensor): log posterior likelihood of current random effects
            current_pred (list[torch.Tensor]): associated model predictions with current random effects
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

        proposal_noise = (
            torch.randn_like(current_etas, device=device) @ self.omega_pop_lower_chol
        )
        proposal_etas = current_etas + step_size * proposal_noise
        (
            proposal_log_prob,
            proposal_theta,
            proposal_log_pdus,
            proposal_pred,
            warnings,
        ) = self.log_posterior_etas(proposal_etas)
        deltas: torch.Tensor = proposal_log_prob - current_log_prob
        log_u: torch.Tensor = torch.log(torch.rand_like(deltas, device=device))
        accept_mask: torch.Tensor = log_u < deltas
        accept_mask_parameters = accept_mask.unsqueeze(-1).expand(
            -1, current_etas.shape[1]
        )
        accept_mask_predictions = accept_mask.index_select(
            0, self.observation_to_patient_index
        )
        new_etas = torch.where(accept_mask_parameters, proposal_etas, current_etas).to(
            device
        )
        new_log_pdus = torch.where(
            accept_mask_parameters, proposal_log_pdus, current_pdu
        ).to(device)
        new_log_prob = torch.where(accept_mask, proposal_log_prob, current_log_prob).to(
            device
        )
        new_complete_likelihood = -2 * new_log_prob.sum(dim=0)
        new_pred = torch.where(accept_mask_predictions, proposal_pred, current_pred).to(
            device
        )
        new_acceptance_rate: float = accept_mask.cpu().float().mean().item()
        if verbose:
            print(f"  Acceptance rate: {new_acceptance_rate:.2f}")
        new_step_size: float = step_size * np.exp(
            learning_rate * (new_acceptance_rate - target_acceptance_rate)
        )
        new_theta = torch.where(
            accept_mask.unsqueeze(1).expand(-1, current_thetas.shape[1]),
            proposal_theta,
            current_thetas,
        ).to(device)
        return (
            new_etas,
            new_log_prob,
            new_complete_likelihood,
            new_pred,
            new_theta,
            new_log_pdus,
            new_step_size,
            warnings,
        )

    def map_estimates_descriptors(self) -> pd.DataFrame:
        theta = self.current_map_estimates
        if theta is None:
            raise ValueError("No estimation available yet. Run the algorithm first.")

        map_per_patient = pd.DataFrame(
            data=theta.cpu().numpy(), columns=self.descriptors
        )
        return map_per_patient

    def map_estimates_predictions(self) -> pd.DataFrame:
        theta = self.current_map_estimates
        if theta is None:
            raise ValueError(
                "No estimation available yet. Run the optimization algorithm first."
            )
        simulated_tensor, _ = self.predict_outputs_from_theta(theta)
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
