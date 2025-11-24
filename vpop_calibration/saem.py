import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.notebook import tqdm
from typing import List, Dict, Union, Optional
from pandas import DataFrame
import pandas as pd
import numpy as np

from .nlme import NlmeModel


# Main SAEM Algorithm Class
class PySaem:
    """
    The method run of this class handles the whole SAEM iterations, alternating between the E-step (sampling individual effects) and the M-step (updating the fixed effects).
    The E-step is mainly performed through the MCMC_Eta_Sampler class, and the M-step directly in this run method.
    The learning rate is updated though the method update_learning_rate.
    SAEM needs initial guesses for all fixed effects that are given by the user.
    They can also set some parameters for the sampling of the etas in the MCMC, and change the simulated annealing factor, applied on omega (covariance of individual effects etas) and the residual error update.
    """

    def __init__(
        self,
        model: NlmeModel,
        observations_df: DataFrame,
        # MCMC parameters for the E-step
        mcmc_first_burn_in: int = 5,
        mcmc_nb_transitions: int = 1,
        nb_phase1_iterations: int = 100,
        nb_phase2_iterations: Union[int, None] = None,
        convergence_threshold: float = 1e-4,
        patience: int = 5,
        learning_rate_power: float = 0.8,
        annealing_factor: float = 0.95,
        init_step_size: float = 0.1,  # stick to the 0.1 - 1 range
        verbose: bool = False,
    ):
        self.model: NlmeModel = model
        self.model.add_observations(observations_df)
        self.observations_df = observations_df
        # MCMC sampling in the E-step parameters
        self.mcmc_first_burn_in: int = mcmc_first_burn_in
        self.mcmc_nb_transitions: int = mcmc_nb_transitions
        # SAEM iteration parameters
        # phase 1 = exploratory: learning rate = 0 and simulated annealing on
        # phase 2 = smoothing: learning rate 1/phase2_iter^factor
        self.nb_phase1_iterations: int = nb_phase1_iterations
        self.nb_phase2_iterations: int = (
            nb_phase2_iterations
            if nb_phase2_iterations is not None
            else nb_phase1_iterations
        )
        # convergence parameters
        self.convergence_threshold: float = convergence_threshold
        self.patience: int = patience
        self.consecutive_converged_iters: int = 0

        # Numerical parameters that depend on the iterations phase
        # The learning rate for the step-size adaptation in E-step sampling
        self.step_size: float = init_step_size / np.sqrt(self.model.nb_PDU)
        self.init_step_size_adaptation: float = 0.5
        self.step_size_learning_rate_power: float = 0.5

        # The learning rate for the stochastic approximation in the M-step
        self.learning_rate_m_step: float = 1.0
        self.learning_rate_power: float = learning_rate_power
        self.annealing_factor: float = annealing_factor

        # Initialize the learning rate and step size adaptation rate
        self.learning_rate_m_step, self.step_size_adaptation = (
            self._compute_learning_rates(0)
        )

        self.verbose = verbose

        # Initialize the random effects to 0
        self.current_etas: torch.Tensor = torch.zeros(
            (self.model.nb_patients, self.model.nb_PDU)
        )

        # Initialize current estimation of patient parameters from the 0 random effects
        (
            self.current_log_prob,
            self.current_thetas,
            self.current_log_pdu,
            self.current_pred,
        ) = self.model.log_posterior_etas(self.current_etas)

        self.history: Dict[str, List[torch.Tensor]] = {
            "log_MI": [self.model.log_MI],
            "population_betas": [self.model.population_betas],
            "population_omega": [self.model.omega_pop],
            "residual_error_var": [self.model.residual_var],
        }
        # pre-compute full design matrix once
        self.X = torch.stack(
            [self.model.design_matrices[ind] for ind in self.model.patients],
            dim=0,
        )

        self.sufficient_stat_gram_matrix = torch.matmul(
            self.X.transpose(1, 2), self.X
        ).sum(dim=0)

        # Initialize sufficient statistics
        self.sufficient_stat_cross_product = (
            self.X.transpose(1, 2) @ self.current_log_pdu.unsqueeze(-1)
        ).sum(dim=0)
        self.sufficient_stat_outer_product = torch.matmul(
            self.current_etas.transpose(0, 1), self.current_etas
        )

    def m_step_update(
        self,
        log_pdu: torch.Tensor,
        s_cross_product: torch.Tensor,
        s_outer_product: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert log_pdu.shape[0] == self.X.shape[0]
        cross_product = (self.X.transpose(1, 2) @ log_pdu.unsqueeze(-1)).sum(dim=0)

        new_s_cross_product = self.stochastic_approximation(
            s_cross_product, cross_product
        )
        new_beta = torch.linalg.solve(
            self.sufficient_stat_gram_matrix, new_s_cross_product
        )
        new_log_pdu = torch.matmul(self.X, new_beta.unsqueeze(0)).squeeze(-1)
        new_etas = log_pdu - new_log_pdu
        outer_product = torch.matmul(new_etas.transpose(0, 1), new_etas)
        new_s_outer_product = self.stochastic_approximation(
            s_outer_product, outer_product
        )
        new_omega = 1 / self.model.nb_patients * new_s_outer_product

        return new_s_cross_product, new_s_outer_product, new_beta.squeeze(-1), new_omega

    def _check_convergence(self) -> bool:
        """Checks for convergence based on the relative change in parameters."""
        current_params = {
            "log_MI": self.model.log_MI,
            "population_betas": self.model.population_betas,
            "population_omega": self.model.omega_pop,
            "residual_error_var": self.model.residual_var,
        }
        all_converged = True
        for name, current_val in current_params.items():
            prev_val = self.prev_params[name]
            abs_diff = torch.abs(current_val - prev_val)
            abs_sum = torch.abs(current_val) + torch.abs(prev_val) + 1e-9
            relative_change = abs_diff / abs_sum
            if torch.any(relative_change > self.convergence_threshold):
                all_converged = False
                break
        return all_converged

    def _compute_learning_rates(self, iteration: int) -> tuple[float, float]:
        """
        Calculates the SAEM learning rate (alpha_k) and Metropolis Hastings step-size (gamma_k).

        Phase 1:
          alpha_k = 1 (exploration)
          gamma_k = c_0 / k^(0.5) , c0 = 2.38 / sqrt(n_PDU)
        Phase 2:
          alpha_k = 1 / (iteration - phase1_iterations + 1) ^ exponent (the iteration index in phase 2)
          gamma_k = 0
        """
        if iteration < self.nb_phase1_iterations:
            learning_rate_m_step = 1.0
            learning_rate_e_step = self.init_step_size_adaptation / (
                np.maximum(1, iteration) ** 0.5
            )
        else:
            learning_rate_m_step = 1.0 / (
                (iteration - self.nb_phase1_iterations + 1) ** self.learning_rate_power
            )
            learning_rate_e_step = 0
        return learning_rate_m_step, learning_rate_e_step

    def stochastic_approximation(
        self, previous: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor:
        return (
            1 - self.learning_rate_m_step
        ) * previous + self.learning_rate_m_step * new

    def one_iteration(self, k: int) -> None:
        if self.verbose:
            print(f"Running iteration {k}")
            print(self.current_thetas.shape)
        # If first iteration, consider burn in
        if k == 0:
            current_iter_burn_in = self.mcmc_first_burn_in
        else:
            current_iter_burn_in = 0

        self.learning_rate_m_step, self.step_size_adaptation = (
            self._compute_learning_rates(k)
        )

        # --- E-step: perform MCMC kernel transitions
        if self.verbose:
            print(f"Current learning rate: {self.learning_rate_m_step: .2f}")
            print("  MCMC sampling")

        for _ in range(current_iter_burn_in + self.mcmc_nb_transitions):
            if self.verbose:
                print(f"  Current MCMC step-size: {self.step_size: .2f}")
            (
                self.current_etas,
                self.current_log_prob,
                self.current_pred,
                self.current_thetas,
                self.current_log_pdu,
                self.step_size,
            ) = self.model.mh_step(
                current_etas=self.current_etas,
                current_log_prob=self.current_log_prob,
                current_pred=self.current_pred,
                current_thetas=self.current_thetas,
                current_pdu=self.current_log_pdu,
                step_size=self.step_size,
                learning_rate=self.step_size_adaptation,
                verbose=self.verbose,
            )

        # --- M-Step: Update Population Means, Omega and Residual variance ---

        # 1. Update residual error variances

        if self.verbose:
            print("  Res var update")

        target_res_var: torch.Tensor = self.model.sum_sq_residuals(self.current_pred)
        current_res_var: torch.Tensor = self.model.residual_var
        if k < self.nb_phase1_iterations:
            target_res_var = torch.max(
                current_res_var * self.annealing_factor, target_res_var
            )

        new_residual_error_var = self.stochastic_approximation(
            current_res_var, target_res_var
        )

        self.model.update_res_var(new_residual_error_var)

        # 2. Update sufficient statistics with stochastic approximation (learning rate)
        # - 1. cross product X^T @ PDUs
        # - 2. Second momdent PDU^T @ PDU

        # 3. Update betas: (X^T X)^-1 @ Cross-product

        # 4. Update Omega (covariance matrix of etas)
        # 1/N_patients (Second moment - Cross-product^T @ beta)
        if self.verbose:
            print("  M-step update:")
        (
            self.sufficient_stat_cross_product,
            self.sufficient_stat_outer_product,
            new_beta,
            new_omega,
        ) = self.m_step_update(
            self.current_log_pdu,
            self.sufficient_stat_cross_product,
            self.sufficient_stat_outer_product,
        )
        self.model.update_betas(new_beta)
        self.model.update_omega(new_omega)

        # 3. Update fixed effects MIs
        if self.model.nb_MI > 0:
            # This step is notoriously under-optimized
            target_log_MI_np = minimize(
                fun=self.MI_objective_function,
                x0=self.model.log_MI.squeeze().numpy(),
                method="L-BFGS-B",
                options={"maxfun": 50},
            ).x
            target_log_MI = torch.from_numpy(target_log_MI_np)
            new_log_MI = self.stochastic_approximation(self.model.log_MI, target_log_MI)

            self.model.update_log_mi(new_log_MI)

        # 4. Update fixed effects betas

        if self.verbose:
            print(
                f"  Updated MIs: {', '.join([f'{torch.exp(logMI).item():.4f}' for logMI in self.model.log_MI])}"
            )
            print(
                f"  Updated Betas: {', '.join([f'{beta:.4f}' for beta in self.model.population_betas.detach().cpu().numpy().flatten()])}"
            )
            print(
                f"  Updated Omega (diag): {', '.join([f'{val.item():.4f}' for val in torch.diag(self.model.omega_pop)])}"
            )
            print(
                f"  Updated Residual Var: {', '.join([f'{res_var:.4f}' for res_var in self.model.residual_var.detach().cpu().numpy().flatten()])}"
            )

        # store history
        self.history["log_MI"].append(self.model.log_MI)
        self.history["population_betas"].append(self.model.population_betas)
        self.history["population_omega"].append(self.model.omega_pop)
        self.history["residual_error_var"].append(self.model.residual_var)

        if k > 0:
            is_converged = self._check_convergence()
            if is_converged:
                self.consecutive_converged_iters += 1
                if self.verbose:
                    print(
                        f"Convergence met. Consecutive iterations: {self.consecutive_converged_iters}/{self.patience}"
                    )
                if self.consecutive_converged_iters >= self.patience:
                    if self.verbose:
                        print(
                            f"\nConvergence reached after {k + 1} iterations. Stopping early."
                        )
            else:
                self.consecutive_converged_iters = 0

        # update prev_params for the next iteration's convergence check
        self.prev_params: Dict[str, torch.Tensor] = {
            "log_MI": self.model.log_MI,
            "population_betas": self.model.population_betas,
            "population_omega": self.model.omega_pop,
            "residual_error_var": self.model.residual_var,
        }
        if self.verbose:
            print("Iter done")

    def MI_objective_function(self, log_MI):
        log_MI_expanded = (
            torch.Tensor(log_MI).unsqueeze(0).repeat((self.model.nb_patients, 1))
        )
        if hasattr(self.model, "patients_pdk"):
            pdk_full = self.model.patients_pdk_full
        else:
            pdk_full = torch.Tensor()
        new_thetas = torch.cat(
            (
                pdk_full,
                torch.cat(
                    (
                        torch.exp(log_MI_expanded),
                        self.current_log_pdu,
                    ),
                    dim=1,
                ),
            ),
            dim=1,
        )
        predictions = self.model.predict_outputs_from_theta(
            new_thetas, self.model.patients
        )
        total_log_lik = 0
        for output_ind in range(self.model.nb_outputs):
            for patient_ind, patient in enumerate(self.model.patients):
                mask = torch.BoolTensor(
                    self.model.observations_tensors[patient]["outputs_indices"]
                    == output_ind
                )
                observed_data = self.model.observations_tensors[patient][
                    "observations"
                ][mask]
                predicted_data = predictions[patient_ind][mask]

                total_log_lik += self.model.log_likelihood_observation(
                    observed_data,
                    predicted_data,
                    self.model.residual_var[output_ind],
                )

        return -total_log_lik

    def run(
        self,
    ) -> None:
        """
        This method starts the SAEM estimation by initiating some class attributes then calling the iterate method.
        returns self.population_betas, self.estimated_population_mus, self.population_omega, self.residual_error_var, self.history
        stores the current state of the estimation so that the iterations can continue later with the continue_iterating method.
        """
        if self.verbose:
            print("Starting SAEM Estimation...")
            print(
                f"Initial Population Betas: {', '.join([f'{beta.item():.2f}' for beta in self.model.population_betas])}"
            )
            print(
                f"Initial Population MIs: {', '.join([f'{torch.exp(logMI).item():.2f}' for logMI in self.model.log_MI])}"
            )
            print(f"Initial Omega:\n{self.model.omega_pop}")
            print(f"Initial Residual Variance: {self.model.residual_var}")

        self.history["log_MI"].append(self.model.log_MI)
        self.history["population_betas"].append(self.model.population_betas)
        self.history["population_omega"].append(self.model.omega_pop)
        self.history["residual_error_var"].append(self.model.residual_var)

        self.prev_params: Dict[str, torch.Tensor] = {
            "log_MI": self.model.log_MI,
            "population_betas": self.model.population_betas,
            "population_omega": self.model.omega_pop,
            "residual_error_var": self.model.residual_var,
        }

        for k in tqdm(range(self.nb_phase1_iterations + self.nb_phase2_iterations)):
            self.one_iteration(k)

        return None

    def plot_convergence_history(
        self,
        true_MI: Optional[Dict[str, float]] = None,
        true_betas: Optional[Dict[str, float]] = None,
        true_sd: Optional[Dict[str, float]] = None,
        true_residual_var: Optional[Dict[str, float]] = None,
    ):
        """
        This method plots the evolution of the estimated parameters (MI, betas, omega, residual error variances) across iterations
        """
        history: Dict[str, List[torch.Tensor]] = self.history
        nb_MI: int = self.model.nb_MI
        nb_betas: int = self.model.nb_betas
        nb_omega_diag_params: int = self.model.nb_PDU
        nb_var_res_params: int = self.model.nb_outputs
        fig, axs = plt.subplots(
            nb_MI + nb_betas + nb_omega_diag_params + nb_var_res_params,
            1,
            figsize=(
                5,
                2 * (nb_betas + nb_omega_diag_params + nb_var_res_params),
            ),
        )
        plot_idx: int = 0
        for j, MI_name in enumerate(self.model.MI_names):
            MI_history = [torch.exp(h[j]).item() for h in history["log_MI"]]
            axs[plot_idx].plot(
                MI_history,
                label=f"Estimated MI for {MI_name} ",
            )
            if true_MI is not None:
                axs[plot_idx].axhline(
                    y=true_MI[MI_name],
                    linestyle="--",
                    label=f"True MI for {MI_name}",
                )
            axs[plot_idx].set_title(f"Convergence of MI ${{{MI_name}}}$")
            axs[plot_idx].set_xlabel("SAEM Iteration")
            axs[plot_idx].set_ylabel("Parameter Value")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True)
            plot_idx += 1
        for j, beta_name in enumerate(self.model.population_betas_names):
            beta_history = [h[j].item() for h in history["population_betas"]]
            axs[plot_idx].plot(
                beta_history,
                label=f"Estimated beta for {beta_name} ",
            )
            if true_betas is not None:
                axs[plot_idx].axhline(
                    y=true_betas[beta_name],
                    linestyle="--",
                    label=f"True beta for {beta_name}",
                )
            axs[plot_idx].set_title(f"Convergence of beta_{beta_name}")
            axs[plot_idx].set_xlabel("SAEM Iteration")
            axs[plot_idx].set_ylabel("Parameter Value")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True)
            plot_idx += 1
        for j, PDU_name in enumerate(self.model.PDU_names):
            omega_diag_history = [
                torch.sqrt(h[j, j]).item() for h in history["population_omega"]
            ]
            axs[plot_idx].plot(
                omega_diag_history,
                label=f"Estimated Omega for {PDU_name}",
            )
            if true_sd is not None:
                axs[plot_idx].axhline(
                    y=true_sd[PDU_name],
                    linestyle="--",
                    label=f"True Omega for {PDU_name}",
                )
            axs[plot_idx].set_title(f"Convergence of Omega for {PDU_name}")
            axs[plot_idx].set_xlabel("SAEM Iteration")
            axs[plot_idx].set_ylabel("Variance")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True)
            plot_idx += 1
        for j, res_name in enumerate(self.model.outputs_names):
            var_res_history = [h[j].item() for h in history["residual_error_var"]]
            axs[plot_idx].plot(
                var_res_history,
                label=f"Estimated residual error variance for {res_name}",
            )
            if true_residual_var is not None:
                axs[plot_idx].axhline(
                    y=true_residual_var[res_name],
                    linestyle="--",
                    label=f"True residual variance for {res_name}",
                )
            axs[plot_idx].set_title("Residual Error var Convergence")
            axs[plot_idx].set_xlabel("SAEM Iteration")
            axs[plot_idx].set_ylabel("var Value")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True)
            plot_idx += 1
        plt.tight_layout()
        plt.show()

    def map_estimates_descriptors(self) -> pd.DataFrame:
        theta = self.current_thetas
        if theta is None:
            raise ValueError("No estimation available yet. Run the algorithm first.")

        map_per_patient = pd.DataFrame(
            data=theta.numpy(), columns=self.model.descriptors
        )
        return map_per_patient

    def map_estimates_predictions(self) -> pd.DataFrame:
        theta = self.current_thetas
        if theta is None:
            raise ValueError(
                "No estimation available yet. Run the optimization algorithm first."
            )
        simulated_tensor = self.model.predict_outputs_from_theta(
            theta, self.model.patients
        )
        simulated_df = self.model.outputs_to_df(simulated_tensor, self.model.patients)
        return simulated_df

    def plot_map_estimates(self) -> None:
        observed = self.observations_df
        simulated_df = self.map_estimates_predictions()

        n_cols = self.model.nb_outputs
        n_rows = self.model.structural_model.nb_protocols
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )

        cmap = plt.get_cmap("Spectral")
        colors = cmap(np.linspace(0, 1, self.model.nb_patients))
        for output_index, output_name in enumerate(self.model.outputs_names):
            for protocol_index, protocol_arm in enumerate(
                self.model.structural_model.protocols
            ):
                obs_loop = observed.loc[
                    (observed["output_name"] == output_name)
                    & (observed["protocol_arm"] == protocol_arm)
                ]
                pred_loop = simulated_df.loc[
                    (simulated_df["output_name"] == output_name)
                    & (simulated_df["protocol_arm"] == protocol_arm)
                ]
                ax = axes[protocol_index, output_index]
                ax.set_xlabel("Time")
                patients_protocol = obs_loop["id"].drop_duplicates().to_list()
                for patient_ind in patients_protocol:
                    patient_num = self.model.patients.index(patient_ind)
                    patient_obs = obs_loop.loc[obs_loop["id"] == patient_ind]
                    patient_pred = pred_loop.loc[pred_loop["id"] == patient_ind]
                    time_vec = patient_obs["time"].values
                    sorted_indices = np.argsort(time_vec)
                    sorted_times = time_vec[sorted_indices]
                    obs_vec = patient_obs["value"].values[sorted_indices]
                    ax.plot(
                        sorted_times,
                        obs_vec,
                        "+",
                        color=colors[patient_num],
                        linewidth=2,
                        alpha=0.6,
                    )
                    if patient_pred.shape[0] > 0:
                        pred_vec = patient_pred["predicted_value"].values[
                            sorted_indices
                        ]
                        ax.plot(
                            sorted_times,
                            pred_vec,
                            "-",
                            color=colors[patient_num],
                            linewidth=2,
                            alpha=0.5,
                        )

                title = f"{output_name} in {protocol_arm}"  # More descriptive title
                ax.set_title(title)
        plt.tight_layout()
        plt.show()
