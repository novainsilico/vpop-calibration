import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from typing import List, Dict, Union, Optional
from pandas import DataFrame
import pandas as pd
import numpy as np

from .nlme import NlmeModel


# Main SAEM Algorithm Class
class PySAEM:
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
        mcmc_burn_in: int = 3,
        mcmc_first_burn_in: int = 30,
        mcmc_nb_samples: int = 10,
        mcmc_proposal_var_scaling_factor: float = 0.2,
        nb_phase1_iterations: int = 100,
        nb_phase2_iterations: Union[int, None] = None,
        convergence_threshold: float = 1e-4,
        patience: int = 5,
        learning_rate_power: float = 0.8,
        annealing_factor: float = 0.95,
        verbose: bool = False,
    ):
        self.model: NlmeModel = model
        self.model.init_mcmc_sampler(observations_df, verbose)
        self.observations_df = observations_df
        # MCMC sampling in the E-step parameters
        self.mcmc_first_burn_in: int = mcmc_first_burn_in
        self.mcmc_burn_in: int = mcmc_burn_in
        self.mcmc_nb_samples: int = mcmc_nb_samples
        self.mcmc_proposal_var_scaling_factor: float = (
            mcmc_proposal_var_scaling_factor  # the variance of the multivariate normal distribution that the next eta from the Markov Chain is sampled from is scaling_factor * omega
        )
        # SAEM iteration parameters
        # phase 1 = exploratory: learning rate = 0 and simulated annealing on
        # phase 2 = smoothing: learning rate 1/phase2_iter^factor
        self.nb_phase1_iterations: int = nb_phase1_iterations
        self.current_phase1_iteration = 0
        self.nb_phase2_iterations: int = (
            nb_phase2_iterations
            if nb_phase2_iterations is not None
            else nb_phase1_iterations
        )
        self.current_phase2_iteration = 0
        # convergence parameters
        self.convergence_threshold: float = convergence_threshold
        self.patience: int = patience
        self.consecutive_converged_iters: int = 0
        # learning rate and simulated annealing (both depending on the phase exploratory/smoothing)
        self.learning_rate_power: float = learning_rate_power
        self.annealing_factor: float = annealing_factor
        # meta
        self.verbose = verbose
        # pySAEM initialization
        self.mean_etas: torch.Tensor = torch.zeros(
            (self.model.nb_patients, self.model.nb_PDU)
        )
        self.history: Dict[str, List[torch.Tensor]] = {
            "log_MI": [],
            "population_betas": [],
            "population_omega": [],
            "residual_error_var": [],
        }
        # pre-compute design matrices once
        self.X_bar = (
            1
            / self.model.nb_patients
            * torch.stack(
                [self.model.design_matrices[ind] for ind in self.model.patients],
                dim=2,
            ).sum(dim=2)
        )
        self.current_map_estimates = None

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

    def _update_learning_rate(self, iteration: int) -> float:
        """
        Calculates the SAEM learning rate (alpha_k).
        Phase 1: alpha_k = 1 (exploration)
        Phase 2: alpha_k = 1 / k_prime, with k_prime = (iteration - phase1_iterations + 1) (the iteration index in phase 2)
        """
        if iteration < self.nb_phase1_iterations:
            return 1.0
        else:
            k_prime: int = (
                iteration - self.nb_phase1_iterations + 1
            )  # iteration index in phase 2
            return 1.0 / (k_prime**self.learning_rate_power)

    def iterate(self) -> None:
        """
        This method handles the main E-step M-step loop.
        """
        current_iter = self.current_phase1_iteration + self.current_phase2_iteration
        total_iter_remaining = (
            self.nb_phase2_iterations + self.nb_phase1_iterations - current_iter
        )

        for k in tqdm(range(current_iter, current_iter + total_iter_remaining)):
            current_alpha_k: float = self._update_learning_rate(k)
            if self.verbose:
                print(
                    f"\n--- SAEM iteration {k+1}/{(self.nb_phase1_iterations+self.nb_phase2_iterations)} (Alpha_k: {current_alpha_k:.3f}) ---"
                )

            # --- E-Step: sample individual random effects (eta_i) ---
            scaling_factor: float = self.mcmc_proposal_var_scaling_factor
            current_mcmc_proposal_var_eta: torch.Tensor = (
                scaling_factor * self.model.omega_pop
                + torch.eye(self.model.nb_PDU) * 1e-7
            )

            if k == 0:
                current_iter_burn_in = self.mcmc_first_burn_in
            else:
                current_iter_burn_in = self.mcmc_burn_in
            if self.verbose:
                print("  MCMC sampling")
            self.mean_etas, mean_log_thetas_PDU, pred = self.model.mcmc_sample(
                init_eta_for_all_ind=self.mean_etas,
                nb_samples=self.mcmc_nb_samples,
                nb_burn_in=current_iter_burn_in,
                proposal_var_eta=current_mcmc_proposal_var_eta,
            )

            # Update the stored Maximum A Posteriori estimates of patient descriptors
            self.current_map_estimates = self.model.individual_parameters(
                self.mean_etas, self.model.patients
            )

            # --- M-Step: Update Population Means, Omega and Residual Error ---

            # 1. Update residual error variances

            if self.verbose:
                print("  Res var update")

            target_res_var: torch.Tensor = self.model.sum_sq_residuals_per_output(pred)
            current_res_var: torch.Tensor = self.model.residual_var
            if k < self.nb_phase1_iterations:
                target_res_var = torch.max(
                    current_res_var * self.annealing_factor, target_res_var
                )

            new_residual_error_var = (
                1 - current_alpha_k
            ) * current_res_var + current_alpha_k * target_res_var

            self.model.update_res_var(new_residual_error_var)

            # 2. Update Omega (covariance matrix of etas)
            if self.verbose:
                print("  Omega update")

            sum_outer_product_etas: torch.Tensor = (
                self.mean_etas.transpose(0, 1) @ self.mean_etas
            )
            target_omega: torch.Tensor = sum_outer_product_etas / self.model.nb_patients
            if k < self.nb_phase1_iterations:
                current_omega_diag = torch.diag(self.model.omega_pop)
                target_omega_diag = torch.diag(target_omega)
                annealed_targed_omega_diag = torch.maximum(
                    current_omega_diag * self.annealing_factor, target_omega_diag
                )
                new_omega_diag = (
                    (1 - current_alpha_k) * current_omega_diag
                    + current_alpha_k * annealed_targed_omega_diag
                )
                new_omega = torch.diag(new_omega_diag)
            else:
                new_omega = (
                    1 - current_alpha_k
                ) * self.model.omega_pop + current_alpha_k * target_omega
            new_omega = (new_omega + new_omega.T) / 2
            new_omega += torch.eye(new_omega.shape[0]) * 1e-6
            self.model.update_omega(new_omega)

            # 3. Update fixed effects MIs
            if self.model.nb_MI > 0:

                if self.verbose:
                    print("  MI update")

                def MI_objective_function(log_MI):
                    log_MI_expanded = (
                        torch.Tensor(log_MI)
                        .unsqueeze(0)
                        .repeat((self.model.nb_patients, 1))
                    )
                    new_thetas = torch.exp(
                        torch.cat((log_MI_expanded, mean_log_thetas_PDU), dim=1)
                    )
                    predictions = self.model.predict_outputs_from_theta(
                        new_thetas, self.model.patients
                    )
                    total_log_lik = 0
                    for output_ind in range(self.model.nb_outputs):
                        for patient_ind, patient in enumerate(self.model.patients):
                            mask = torch.BoolTensor(
                                self.model.observations_tensors[patient][
                                    "outputs_indices"
                                ]
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

                target_log_MI_np = minimize(
                    fun=MI_objective_function,
                    x0=self.model.log_MI.squeeze().numpy(),
                    method="L-BFGS-B",
                    options={"maxfun": 50},
                ).x
                target_log_MI = torch.from_numpy(target_log_MI_np)
                new_log_MI = (
                    1 - current_alpha_k
                ) * self.model.log_MI + current_alpha_k * target_log_MI

                self.model.update_log_mi(new_log_MI)

            # 4. Update fixed effects betas
            if self.verbose:
                print("  Beta update")
            # Compute the matrix X^T Omega^-1 X
            omega_inv = torch.cholesky_inverse(self.model.omega_pop_lower_chol)
            lhs_matrix: torch.Tensor = self.X_bar.T @ omega_inv @ self.X_bar
            log_theta_bar = 1 / self.model.nb_patients * mean_log_thetas_PDU.sum(dim=0)

            # Compute the vector X^T Omega^-1 y
            rhs_vector: torch.Tensor = self.X_bar.T @ (omega_inv @ log_theta_bar)

            target_beta: torch.Tensor = torch.Tensor(
                torch.linalg.solve(
                    lhs_matrix + 1e-6 * torch.eye(self.model.nb_betas),
                    rhs_vector,
                )
            )
            new_beta: torch.Tensor = (
                1 - current_alpha_k
            ) * self.model.population_betas + current_alpha_k * target_beta

            self.model.update_betas(new_beta)

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

            if self.current_phase1_iteration < self.nb_phase1_iterations:
                self.current_phase1_iteration += 1
            else:
                self.current_phase2_iteration += 1

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
                        break
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

        print("\nEstimation Finished.")
        idx: int = 0
        self.estimated_MI_mus = []
        for j in range(self.model.nb_MI):
            self.estimated_MI_mus.append(torch.exp(self.model.log_MI[j]).item())
        for PDU_name in self.model.PDU_names:
            self.estimated_MI_mus.append(
                torch.exp(self.model.population_betas[idx]).item()
            )
            idx += 1
            if self.model.covariate_map and PDU_name in self.model.covariate_map:
                for i in range(len(self.model.covariate_map[PDU_name])):
                    idx += 1

        if self.verbose:
            print(
                f"Estimated MI: {', '.join([f'{MI:.4f}' for MI in self.estimated_MI_mus[:self.model.nb_MI]])}"
            )
            print(
                f"Estimated mus: {', '.join([f'{mu:.4f}' for mu in self.estimated_MI_mus[self.model.nb_MI:]])}"
            )
            print(
                f"Estimated population betas: {', '.join([f'{estimated_beta.item():.4f}' for estimated_beta in self.model.population_betas])}"
            )
            print(
                f"Estimated omega (diagonal): {', '.join([f'{val.item():.4f}' for val in torch.diag(self.model.omega_pop)])}"
            )
            print(
                f"Estimated residual var: {', '.join([f'{val.item():.4f}' for val in self.model.residual_var.flatten()])}"
            )

        return None

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
        self.iterate()
        return None

    def continue_iterating(
        self, nb_phase1_further_iterations=0, nb_phase2_further_iterations=0
    ) -> None:
        """
        This method is to be used when the run method has already run and the user wants to further iterate.
        It updates the iterations necessary and calls the iterate method.
        returns self.population_betas, self.estimated_population_mus, self.population_omega, self.residual_error_var, self.history
        """
        if nb_phase1_further_iterations > 0 and self.current_phase2_iteration > 0:
            raise ValueError(
                "Phase 2 (smoothing) has already started. No further phase1 iterations can be conducted."
            )
        self.nb_phase1_iterations += nb_phase1_further_iterations
        self.nb_phase2_iterations += nb_phase2_further_iterations
        self.iterate()
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
            omega_diag_history = [h[j, j].item() for h in history["population_omega"]]
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
        theta = self.current_map_estimates
        if theta is None:
            raise ValueError("No estimation available yet. Run the algorithm first.")

        map_per_patient = pd.DataFrame(
            data=theta.numpy(), columns=self.model.descriptors
        )
        return map_per_patient

    def map_estimates_predictions(self) -> pd.DataFrame:
        theta = self.current_map_estimates
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

        cmap = plt.cm.get_cmap("Spectral")
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
