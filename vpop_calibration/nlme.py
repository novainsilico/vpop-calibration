import torch
from typing import List, Dict, Union, Tuple, Optional
import pandas as pd

from .structural_model import StructuralModel


class NlmeModel:
    def __init__(
        self,
        structural_model: StructuralModel,
        patients_df: pd.DataFrame,
        init_log_MI: Dict[str, float],
        init_PDU: Dict[str, Dict[str, float]],
        init_res_var: List[float],
        covariate_map: Optional[dict[str, dict[str, dict[str, str | float]]]] = None,
        error_model_type: str = "additive",
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
        """
        self.structural_model: StructuralModel = structural_model

        self.MI_names: List[str] = list(init_log_MI.keys())
        self.nb_MI: int = len(self.MI_names)
        self.initial_log_MI = torch.Tensor([val for _, val in init_log_MI.items()])
        self.PDU_names: List[str] = list(init_PDU.keys())
        self.nb_PDU: int = len(self.PDU_names)

        self.patients_df: pd.DataFrame = patients_df
        self.patients: List[str | int] = self.patients_df["id"].unique().tolist()
        self.nb_patients: int = len(self.patients)
        covariate_columns = self.patients_df.columns.to_list()
        if "protocol_arm" not in covariate_columns:
            self.patients_df["protocol_arm"] = "identity"

        additional_columns: List[str] = self.patients_df.drop(
            ["id", "protocol_arm"], axis=1
        ).columns.tolist()

        init_betas_list: List = []
        if covariate_map is None:
            print(
                f"No covariate map provided. All additional columns in `patients_df` will be handled as known descriptors: {additional_columns}"
            )
            self.covariate_map = None
            self.covariate_names = []
            self.nb_covariates = 0
            self.population_betas_names = self.PDU_names
            init_betas_list = [val["mean"] for _, val in init_PDU.items()]
            self.PDK_names = additional_columns
            self.nb_PDK = len(self.PDK_names)
        else:
            self.covariate_map = covariate_map
            self.population_betas_names: List = []
            covariate_set = set()
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
                        coef_val = coef["value"]
                        self.population_betas_names.append(coef_name)
                        init_betas_list.append(coef_val)
            self.covariate_names = list(covariate_set)
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
                    {patient: torch.Tensor(row[self.PDK_names].values)}
                )
            self.patients_pdk_full = torch.cat(
                [self.patients_pdk[ind] for ind in self.patients]
            )
            print(f"Successfully loaded {self.nb_PDK} known descriptors:")
            print(self.PDK_names)

        if set(self.PDK_names + self.PDU_names + self.MI_names) != set(
            self.structural_model.parameter_names
        ):
            raise ValueError(
                f"Non-matching descriptor set and structural model parameter set:\n{set(self.PDK_names + self.PDU_names + self.MI_names)}\n{set(self.structural_model.parameter_names)}"
            )

        self.descriptors: List[str] = self.PDK_names + self.PDU_names + self.MI_names
        self.nb_descriptors: int = len(self.descriptors)
        # Assume that the descriptors will always be provided to the model in the following order:
        #   PDK, PDU, MI
        self.model_input_to_descriptor = torch.LongTensor(
            [
                self.descriptors.index(param)
                for param in self.structural_model.parameter_names
            ]
        )
        self.initial_betas = torch.Tensor(init_betas_list)
        self.nb_betas: int = len(self.population_betas_names)
        self.outputs_names: List[str] = self.structural_model.output_names
        self.nb_outputs: int = self.structural_model.nb_outputs
        self.error_model_type: str = error_model_type
        self.init_res_var = torch.Tensor(init_res_var)
        self.init_omega = torch.diag(
            torch.tensor([float(pdu["sd"] ** 2) for pdu in init_PDU.values()])
        )

        # Assemble the list of design matrices from the covariance structure
        self.design_matrices = self._create_all_design_matrices()

        # Initiate the nlme parameters
        self.log_MI = self.initial_log_MI
        self.population_betas = self.initial_betas
        self.omega_pop = self.init_omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop)
        self.residual_var = self.init_res_var
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU),
            covariance_matrix=self.omega_pop,
        )

    def _create_design_matrix(self, covariates: Dict[str, float]) -> torch.Tensor:
        """
        Creates the design matrix X_i for a single individual based on the model's covariate map.
        This matrix will be multiplied with population betas so that log(theta_i[PDU]) = X_i @ betas + eta_i.
        """
        design_matrix_X_i = torch.zeros((self.nb_PDU, self.nb_betas))
        col_idx = 0
        for i, PDU_name in enumerate(self.PDU_names):
            design_matrix_X_i[i, col_idx] = 1.0
            col_idx += 1
            if self.covariate_map is not None:
                for covariate in self.covariate_map[PDU_name].keys():
                    design_matrix_X_i[i, col_idx] = float(covariates[covariate])
                    col_idx += 1
        return design_matrix_X_i

    def _create_all_design_matrices(self) -> Dict[Union[str, int], torch.Tensor]:
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

        self.observations_tensors: Dict = {}
        for patient in self.patients:
            this_patient = processed_df.loc[processed_df["id"] == patient]

            tasks_indices = this_patient["task_index"].values
            outputs_indices = [
                self.structural_model.task_idx_to_output_idx[task]
                for task in tasks_indices
            ]

            outputs = torch.Tensor(this_patient["value"].values)

            time_steps = torch.Tensor(this_patient["time"].values)

            self.observations_tensors.update(
                {
                    patient: {
                        "observations": outputs,
                        "time_steps": time_steps,
                        "tasks_indices": torch.LongTensor(tasks_indices),
                        "outputs_indices": torch.LongTensor(outputs_indices),
                    }
                }
            )

    def update_omega(self, omega: torch.Tensor) -> None:
        """Update the covariance matrix of the NLME model."""
        self.omega_pop = omega
        self.omega_pop_lower_chol = torch.linalg.cholesky(self.omega_pop)
        self.eta_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU),
            covariance_matrix=self.omega_pop,
        )

    def update_res_var(self, residual_var: torch.Tensor) -> None:
        """Update the residual variance of the NLME model."""
        self.residual_var = residual_var

    def update_betas(self, betas: torch.Tensor) -> None:
        """Update the betas of the NLME model."""
        self.population_betas = betas

    def update_log_mi(self, log_MI: torch.Tensor) -> None:
        """Update the model intrinsic parameter values of the NLME model."""
        self.log_MI = log_MI

    def sample_individual_etas(self, nb_patients=None) -> torch.Tensor:
        """Sample individual random effects from the current estimate of Omega

        Returns:
            torch.Tensor (size nb_patients x nb_PDUs): individual random effects for all patients in the population
        """
        if nb_patients is None:
            # In case no number of patients is requested, assume we want to sample one set of random effects per patient in the observation data set
            nb_patients = self.nb_patients
        etas_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.nb_PDU),
            covariance_matrix=self.omega_pop,
        ).expand([nb_patients])
        etas = etas_dist.sample()
        return etas

    def individual_parameters(
        self,
        individual_etas: torch.Tensor,
        ind_ids_for_etas: List[Union[str, int]],
    ) -> torch.Tensor:
        """Compute individual patient parameters

        Transforms log(MI) (Model intrinsic), betas: log(mu)s & coeffs for covariates and individual random effects (etas) into individual parameters (theta_i), for each set of etas of the list and corresponding design matrix.
        Assumes log-normal distribution for individual parameters and covariate effects: theta_i[PDU] = mu_pop * exp(eta_i) * exp(covariates_i * cov_coeffs) where eta_i is from N(0, Omega) and theta_i[MI]=MI.

        Args:
            individual_etas (torch.Tensor): one set of sampled random effects for each patient
            ind_ids_for_etas (List[Union[str, int]]): List of individual ids corresponding to the sampled etas, used to fetch the design matrices
        Returns:
            torch.Tensor [nb_patients x nb_parameters]: One parameter set for each patient. Dim 0 corresponds to the patients, dim 1 is the parameters
        """
        nb_patients_for_etas = len(ind_ids_for_etas)
        # Gather the necessary design matrices
        list_design_matrices = [
            self.design_matrices[ind_id] for ind_id in ind_ids_for_etas
        ]
        # stack all design matrices into a single large tensor
        stacked_X = torch.stack(list_design_matrices)
        # Compute the inidividual PDU
        log_thetas_PDU = stacked_X @ self.population_betas + individual_etas
        # Gather the MI values, and expand them (same for each patient)
        log_MI_expanded = self.log_MI.unsqueeze(0).repeat(nb_patients_for_etas, 1)

        # List the PDK values for each patient, and assemble them in a tensor
        if hasattr(self, "patients_pdk"):
            patients_pdk = torch.cat(
                [self.patients_pdk[ind_id] for ind_id in ind_ids_for_etas]
            )
        else:
            patients_pdk = torch.Tensor()
        # This step is crucial: we need to ensure the parameters are stored in the correct order
        # PDK, PDU, MI
        thetas = torch.cat(
            (
                patients_pdk,
                torch.exp(torch.cat((log_thetas_PDU, log_MI_expanded), dim=1)),
            ),
            dim=1,
        )
        return thetas

    def predict_outputs_from_theta(
        self, thetas: torch.Tensor, ind_ids: List[str | int]
    ) -> List[torch.Tensor]:
        """Return model predictions for all patients

        Args:
            thetas (torch.Tensor): Parameter values per patient (one by row)
            ind_ids(List[str | int]): the ids of the patients to be simulated

        Returns:
            List[torch.Tensor]: a tensor of predictions for each patient
        """
        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute patient predictions without an associated observations data frame."
            )
        list_X = []
        list_tasks = []
        for ind_idx, ind in enumerate(ind_ids):
            # Prepare the inputs for the GP
            time_steps = self.observations_tensors[ind]["time_steps"].unsqueeze(-1)
            this_patient_theta_ordered = thetas[ind_idx, self.model_input_to_descriptor]
            thetas_repeated = this_patient_theta_ordered.unsqueeze(0).repeat(
                (time_steps.shape[0], 1)
            )
            # This steps requires that the `time` is passed as the last input column to the structural model
            inputs = torch.cat(
                (
                    thetas_repeated,
                    time_steps,
                ),
                dim=1,
            )
            list_X.append(inputs)
            list_tasks.append(self.observations_tensors[ind]["tasks_indices"])
        pred = self.structural_model.simulate(list_X, list_tasks)
        return pred

    def outputs_to_df(
        self, outputs: List[torch.Tensor], ind_ids: List[str | int]
    ) -> pd.DataFrame:
        """Transform the NLME model outputs to a data frame in order to compare with observed data

        Args:
            outputs (List[torch.Tensor]): Outputs from `self.predict_outputs_from_theta`
            ind_ids (List[str  |  int]): the list of patients that were simulated

        Returns:
            pd.DataFrame: A data frame containing the following columns
            - `id`
            - `output_name`
            - `protocol_arm`
            - `time`
            - `predicted_value`
        """
        df_list = []
        for ind_idx, ind in enumerate(ind_ids):
            time_steps = self.observations_tensors[ind]["time_steps"]
            task_list = self.observations_tensors[ind]["tasks_indices"]
            temp_df = pd.DataFrame(
                {
                    "time": time_steps.numpy(),
                    "id": ind,
                    "task_index": task_list,
                    "predicted_value": outputs[ind_idx].numpy(),
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

        log_priors: torch.Tensor = self.eta_distribution.log_prob(etas)
        return log_priors

    def _log_posterior_etas(
        self, etas: torch.Tensor, ind_ids_for_etas: List[Union[str, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Compute the log-posterior of a list of random effects

        Args:
            etas (torch.Tensor): Random effects samples
            ind_ids_for_etas (List[Union[str, int]]): Patient ids corresponding to each eta

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], DataFrame]:
            - log-posterior likelihood of etas
            - tensor of individual parameters for each patient
            - list of simulated values for each patient

        """
        if not hasattr(self, "observations_tensors"):
            raise ValueError(
                "Cannot compute log-posterior without an associated observations data frame."
            )
        # Get individual parameters in a tensor
        individual_params: torch.Tensor = self.individual_parameters(
            individual_etas=etas,
            ind_ids_for_etas=ind_ids_for_etas,
        )
        observations = [
            self.observations_tensors[ind]["observations"] for ind in ind_ids_for_etas
        ]
        output_indices_list = [
            self.observations_tensors[ind]["outputs_indices"]
            for ind in ind_ids_for_etas
        ]
        res_var = [
            self.residual_var[output_indices] for output_indices in output_indices_list
        ]
        # Run the surrogate model
        full_pred = self.predict_outputs_from_theta(individual_params, ind_ids_for_etas)

        # calculate log-prior of the random samples
        log_priors: torch.Tensor = self._log_prior_etas(etas)

        # group by individual and calculate log-likelihood for each
        list_log_lik_obs: List[torch.Tensor] = list(
            map(self.log_likelihood_observation, full_pred, observations, res_var)
        )

        log_lik_obs = torch.tensor(list_log_lik_obs)
        log_posterior = log_lik_obs + log_priors

        return log_posterior, individual_params, full_pred

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

    def sum_sq_residuals_per_output(
        self, pred: Dict[str | int, torch.Tensor]
    ) -> torch.Tensor:
        sum_residuals = torch.zeros(self.nb_outputs)
        for output_ind in range(self.nb_outputs):
            for patient in self.patients:
                mask = torch.BoolTensor(
                    self.observations_tensors[patient]["outputs_indices"] == output_ind
                )
                n_obs = mask.sum()
                observed = self.observations_tensors[patient]["observations"][mask]
                predicted = pred[patient][mask]
                sum_residuals[output_ind] += (
                    torch.square(self.calculate_residuals(observed, predicted)).sum()
                    / n_obs
                )
        return sum_residuals / self.nb_patients

    def log_likelihood_observation(
        self,
        observed_data: torch.Tensor,
        predictions: torch.Tensor,
        residual_error_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the log-likelihood of observations given predictions and error model, assuming errors follow N(0,sqrt(residual_error_var))
        observed_data: torch.Tensor of observations for one individual
        predictions: torch.Tensor of predictions for one individual organized in the same way as observed_data
        residual_error_var: torch.Tensor of the error for each output, dim: [nb_outputs]
        """
        if torch.any(torch.isinf(predictions)) or torch.any(torch.isnan(predictions)):
            return torch.Tensor([-torch.inf])  # invalid predictions
        residuals: torch.Tensor = self.calculate_residuals(observed_data, predictions)
        # ensure error_std is positive
        res_error_var = torch.maximum(
            torch.full_like(residual_error_var, 1e-6), residual_error_var
        )
        # Log-likelihood of normal distribution
        if self.error_model_type == "additive":
            log_lik = -0.5 * torch.sum(
                torch.log(2 * torch.pi * res_error_var) + (residuals**2 / res_error_var)
            )  # each row of res_error_var is the variance of the residual error corresponding to the same row of residuals (one row = one output)
        elif self.error_model_type == "proportional":
            log_lik = -0.5 * torch.sum(
                torch.log(2 * torch.pi * res_error_var * predictions)
                + (residuals**2 / res_error_var)
            )  # each row of res_error_var is the variance of the residual error corresponding to the same row of residuals (one row = one output)
        else:
            raise ValueError("Non supported error type.")
        return log_lik

    def init_mcmc_sampler(
        self,
        observations_df: pd.DataFrame,
        verbose: bool,
    ) -> None:
        self.add_observations(observations_df)
        self.verbose = verbose

    def mcmc_sample(
        self,
        init_eta_for_all_ind: torch.Tensor,
        proposal_var_eta: torch.Tensor,
        nb_samples: int,
        nb_burn_in: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str | int, torch.Tensor]]:
        """Metropolis-Hastings sampling of individual MCMC

        Performs Metropolis-Hastings sampling for the individuals' etas. The MCMC chains (one per individual) advance in parallel.
        The acceptance criterion for each sample is
                log(random_uniform) < proposed_log_posterior - current_log_posterior

        Returns the mean of sampled etas per individual, the mean of the log_thetas_PDU (individual PDU parameters associated with the sampled etas) and the mean predictions associated with these individual parameters/etas.

        Args:
            init_eta_for_all_ind: torch.Tensor of dim [nb_individuals x nb_PDUs]: the current samples, i.e. starting points for each chain
            nb_samples: int, how many samples will be kept from each chain
            nb_burn_in: int, how many accepted samples are disgarded before we consider that the chain has converged enough

        Returns:

        """
        if not hasattr(self, "observations_tensors"):
            raise ValueError("Please run `init_mcmc_sampler` first.")

        sum_etas = torch.zeros_like(init_eta_for_all_ind)
        sum_log_thetas_PDU = torch.zeros(self.nb_patients, self.nb_PDU)

        predictions_history = {ind: [] for ind in self.patients}
        current_eta_for_all_ind = init_eta_for_all_ind
        current_log_posteriors, _, _ = self._log_posterior_etas(
            init_eta_for_all_ind, self.patients
        )

        sample_counts = torch.zeros(self.nb_patients)
        accepted_counts = torch.zeros(self.nb_patients)
        total_proposals = torch.zeros(self.nb_patients)
        done = torch.full((self.nb_patients, 1), False)

        while not torch.all(done).item():
            active_indices = torch.where(~done)[0]
            active_ind_ids = [self.patients[i] for i in active_indices]

            total_proposals[active_indices] += 1
            proposal_dist = torch.distributions.MultivariateNormal(
                current_eta_for_all_ind[active_indices], proposal_var_eta
            )
            proposed_etas: torch.Tensor = proposal_dist.sample()

            proposed_log_posteriors, patient_parameters, predictions = (
                self._log_posterior_etas(proposed_etas, active_ind_ids)
            )

            deltas: torch.Tensor = (
                proposed_log_posteriors - current_log_posteriors[active_indices]
            )
            accept_mask: torch.Tensor = (
                torch.log(torch.rand(len(active_indices))) < deltas
            )

            # update only the accepted etas and log-posteriors
            current_eta_for_all_ind[active_indices[accept_mask]] = proposed_etas[
                accept_mask
            ]
            current_log_posteriors[active_indices[accept_mask]] = (
                proposed_log_posteriors[accept_mask]
            )
            accepted_counts[active_indices] += accept_mask.int()
            for j, idx in enumerate(active_indices):
                if accept_mask[j] and accepted_counts[idx] >= nb_burn_in:
                    sum_etas[idx] += current_eta_for_all_ind[idx]
                    theta = patient_parameters[j, :]
                    PDUs = theta[self.nb_PDK : self.nb_PDK + self.nb_PDU]
                    sum_log_thetas_PDU[idx] += torch.log(PDUs)

                    # store predictions for averaging later and then use in the M-step
                    accepted_preds_for_ind = predictions[j]
                    predictions_history[self.patients[idx]].append(
                        accepted_preds_for_ind
                    )

                    sample_counts[idx] += 1
                    if sample_counts[idx] == nb_samples:
                        done[idx] = True

        # calculate mean etas and log of individual parameters PDU
        mean_etas = sum_etas / nb_samples
        mean_log_thetas_PDU = sum_log_thetas_PDU / nb_samples

        # calculate mean predictions from the collected history
        mean_predictions = {}
        for id in self.patients:
            patient_preds = torch.stack(predictions_history[id])
            mean_predictions.update({id: patient_preds.mean(dim=0)})

        if self.verbose:
            acceptance_rate = accepted_counts / total_proposals
            print(f"Average acceptance: {acceptance_rate.mean():.2f}")
        return (mean_etas, mean_log_thetas_PDU, mean_predictions)
