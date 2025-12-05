import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm.notebook import tqdm
from typing import Union, Optional
from pandas import DataFrame
import pandas as pd
import numpy as np
from IPython.display import display

from .utils import smoke_test
from .nlme import NlmeModel
from .structural_model import StructuralGp


# Main SAEM Algorithm Class
class PySaem:
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
        init_step_size: float = 0.5,  # stick to the 0.1 - 1 range
        verbose: bool = False,
        optim_max_fun: int = 50,
        live_plot: bool = True,
        plot_frames: int = 20,
        plot_columns: int = 3,
        plot_indiv_figsize: tuple[float, float] = (3.0, 1.2),
        true_log_MI: Optional[dict[str, float]] = None,
        true_log_PDU: Optional[dict[str, dict[str, float | bool]]] = None,
        true_res_var: Optional[list[float]] = None,
        true_covariates: Optional[dict[str, dict[str, dict[str, str | float]]]] = None,
    ):
        """Instantiate an SAEM optimizer for an NLME model

        Args:
            model (NlmeModel): The model to be optimized
            observations_df (DataFrame): The data set containing observations
            mcmc_first_burn_in (int, optional): Number of discarded samples in the first iteration. Defaults to 5.
            mcmc_nb_transitions (int, optional): Number of kernel transitions computed at each iteration. Defaults to 1.
            nb_phase1_iterations (int, optional): Number of iterations in the exploration phase. Defaults to 100.
            nb_phase2_iterations (Union[int, None], optional): Number of iterations in the convergence phase. Defaults to None, implying nb_phase_2 = nb_phase_1.
            convergence_threshold (float, optional): Estimated parameter relative change threshold considered for convergence. Defaults to 1e-4.
            patience (int, optional): Number of iterations of consecutive low relative change considered for early stopping of the algorithm. Defaults to 5.
            learning_rate_power (float, optional): Exponential decay exponent for the M-step learning rate (stochastic approximation). Defaults to 0.8.
            annealing_factor (float, optional): Exploration phase annealing factor for residual and parameter variance. Defaults to 0.95.
            init_step_size (float, optional): Initial MCMC step size scaling factor. Defaults to 0.5.
            optim_max_fun(int): Maximum number of function calls in the scipy.optimize (used for model intrinsic parameters calibration). Defaults to 50.
            verbose (bool): Print various info during iterations. Defaults to False.
            live_plot (bool): Print and update a plot of parameters during iterations. Defaults to True.
            plot_frames (int): Frequency at which the live plot should be updated (number of iterations). The lower the slower. Defaults to 20.
            plot_columns (int): Number of columns to display the convergence plot. Defaults to 3.
            plot_indiv_figsize (tuple[float,float]): individual figure size in the convergence plot (width, height).
        """

        self.model: NlmeModel = model
        self.model.add_observations(observations_df)
        self.observations_df = observations_df
        # MCMC sampling in the E-step parameters
        self.mcmc_first_burn_in: int = mcmc_first_burn_in
        self.mcmc_nb_transitions: int = mcmc_nb_transitions
        # SAEM iteration parameters
        # phase 1 = exploratory: learning rate = 0 and simulated annealing on
        # phase 2 = smoothing: learning rate 1/phase2_iter^factor
        if smoke_test:
            self.nb_phase1_iterations = 1
            self.nb_phase2_iterations = 2
        else:
            self.nb_phase1_iterations: int = nb_phase1_iterations
            self.nb_phase2_iterations: int = (
                nb_phase2_iterations
                if nb_phase2_iterations is not None
                else nb_phase1_iterations
            )
        self.current_phase = 1

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
        if smoke_test:
            self.optim_max_fun = 1
        else:
            self.optim_max_fun = optim_max_fun
        self.live_plot = live_plot
        self.plot_frames = plot_frames
        self.plot_columns = plot_columns
        self.plot_indiv_figsize = plot_indiv_figsize

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

        # Initialize the optimizer history
        self._init_history(
            self.model.population_betas,
            self.model.omega_pop,
            self.model.log_MI,
            self.model.residual_var,
        )
        self.current_iteration: int = 0

        # Initialize the values for convergence checks
        self.prev_params: dict[str, torch.Tensor] = {
            "log_MI": self.model.log_MI,
            "population_betas": self.model.population_betas,
            "population_omega": self.model.omega_pop,
            "residual_error_var": self.model.residual_var,
        }

        # pre-compute full design matrix once
        self.X = torch.stack(
            [self.model.design_matrices[ind] for ind in self.model.patients],
            dim=0,
        )
        # Precompute the gram matrix
        self.sufficient_stat_gram_matrix = torch.matmul(
            self.X.transpose(1, 2), self.X
        ).sum(dim=0)

        # Initialize sufficient statistics
        self.sufficient_stat_cross_product = (
            self.X.transpose(1, 2) @ self.current_log_pdu.unsqueeze(-1)
        ).sum(dim=0)
        self.sufficient_stat_outer_product = torch.matmul(
            self.current_log_pdu.transpose(0, 1), self.current_log_pdu
        )

        self.true_log_MI = true_log_MI
        self.true_log_PDUs = true_log_PDU
        if true_covariates is not None:
            self.true_cov = {
                str(cov["coef"]): float(cov["value"])
                for item in true_covariates.values()
                for cov in item.values()
            }
        if true_res_var is not None:
            self.true_res_var = {
                self.model.outputs_names[k]: val for k, val in enumerate(true_res_var)
            }

        if isinstance(self.model.structural_model, StructuralGp):
            self.training_ranges = {}
            training_samples = np.log(
                self.model.structural_model.gp_model.data.full_df_raw[
                    self.model.PDU_names + self.model.MI_names
                ]
            )
            train_min = training_samples.min(axis=0)
            train_max = training_samples.max(axis=0)
            for param in self.model.PDU_names + self.model.MI_names:
                self.training_ranges.update(
                    {param: {"low": train_min[param], "high": train_max[param]}}
                )

    def m_step_update(
        self,
        log_pdu: torch.Tensor,
        s_cross_product: torch.Tensor,
        s_outer_product: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform the M-step update

        Args:
            log_pdu (torch.Tensor): Current estimation of the log-scaled parameters
            s_cross_product (torch.Tensor): Current sufficient statistics 1 - cross product
            s_outer_product (torch.Tensor): Current sufficient statistics 2 - outer product

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Updated value for
            - sufficient statistics: cross product
            - sufficient statistics: outer product
            - beta parameters
            - omega matrix
        """

        assert log_pdu.shape[0] == self.X.shape[0]
        cross_product = (self.X.transpose(1, 2) @ log_pdu.unsqueeze(-1)).sum(dim=0)
        new_s_cross_product = self._stochastic_approximation(
            s_cross_product, cross_product
        )
        outer_product = torch.matmul(log_pdu.transpose(0, 1), log_pdu)
        new_s_outer_product = self._stochastic_approximation(
            s_outer_product, outer_product
        )

        new_beta = torch.linalg.solve(
            self.sufficient_stat_gram_matrix, new_s_cross_product
        )

        new_log_pdu = torch.matmul(self.X, new_beta.unsqueeze(0)).squeeze(-1)
        # Propose a new value for omega
        new_omega = (
            1
            / self.model.nb_patients
            * (
                new_s_outer_product
                - torch.matmul(new_log_pdu.transpose(0, 1), new_log_pdu)
            )
        )
        new_omega = self._clamp_eigen_values(new_omega)

        return (
            new_s_cross_product,
            new_s_outer_product,
            new_beta.squeeze(-1),
            new_omega,
        )

    def _check_convergence(self, new_params: dict[str, torch.Tensor]) -> bool:
        """Checks for convergence based on the relative change in parameters."""
        all_converged = True
        for name, current_val in new_params.items():
            if current_val.shape[0] > 0:
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
          gamma_k = c_0 / k^(0.5) , c0 = init_step_size_adaptation / sqrt(n_PDU)
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

    def _stochastic_approximation(
        self, previous: torch.Tensor, new: torch.Tensor
    ) -> torch.Tensor:
        """Perform stochastic approximation

        Args:
            previous (torch.Tensor): The current value of the tensor
            new (torch.Tensor): The target value of the tensor

        Returns:
            torch.Tensor: (1 - learning_rate) * previous + learning_rate * new
        """
        assert (
            previous.shape == new.shape
        ), f"Wrong shape in stochastic approximation: {previous.shape}, {new.shape}"
        return (
            1 - self.learning_rate_m_step
        ) * previous + self.learning_rate_m_step * new

    def _simulated_annealing(
        self, current: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Perform simulated annealing

        This function allows to constrain the reduction of certain values by a factor stored in self.annealing_factor

        Args:
            current (torch.Tensor): Current value of the tensor
            target (torch.Tensor): Target value of the tensor

        Returns:
            torch.Tensor: maximum(annealing_factor * current, target)
        """
        return torch.maximum(self.annealing_factor * current, target)

    def _clamp_eigen_values(self, omega: torch.Tensor, min_eigenvalue: float = 1e-6):
        """
        Project a matrix onto the cone of Positive Definite matrices.
        """
        # 1. Ensure symmetry (sometimes float error breaks symmetry slightly)
        omega = 0.5 * (omega + omega.T)

        # 2. Eigen Decomposition
        L, V = torch.linalg.eigh(omega)

        # 3. Clamp eigenvalues
        L_clamped = torch.clamp(L, min=min_eigenvalue)

        # 4. Reconstruct
        matrix_spd = torch.matmul(V, torch.matmul(torch.diag(L_clamped), V.T))

        return matrix_spd

    def _init_history(
        self,
        beta: torch.Tensor,
        omega: torch.Tensor,
        log_mi: torch.Tensor,
        res_var: torch.Tensor,
    ) -> None:
        # Initialize the history
        self.history = {}
        # Add the pdus (mean, variance)
        for i, pdu in enumerate(self.model.PDU_names):
            beta_index = self.model.population_betas_names.index(pdu)
            self.history.update(
                {
                    pdu: {
                        "mu": [beta[beta_index]],
                        "sigma_sq": [omega[beta_index, beta_index]],
                    }
                }
            )
        # Add the covariates
        for i, cov in enumerate(self.model.covariate_coeffs_names):
            beta_index = self.model.population_betas_names.index(cov)
            self.history.update({cov: [beta[beta_index]]})
        # Add Omega
        self.history.update({"omega": [omega]})
        # Add the model intrinsic params
        for i, mi in enumerate(self.model.MI_names):
            self.history.update({mi: [log_mi[i]]})
        # Add the residual variance
        for i, output in enumerate(self.model.outputs_names):
            self.history.update({output: [res_var[i]]})

    def _append_history(
        self,
        beta: torch.Tensor,
        omega: torch.Tensor,
        log_mi: torch.Tensor,
        res_var: torch.Tensor,
    ) -> None:
        # Update the history
        for i, pdu in enumerate(self.model.PDU_names):
            beta_index = self.model.population_betas_names.index(pdu)
            self.history[pdu]["mu"].append(beta[beta_index])
            self.history[pdu]["sigma_sq"].append(omega[beta_index, beta_index])

        for i, cov in enumerate(self.model.covariate_coeffs_names):
            beta_index = self.model.population_betas_names.index(cov)
            self.history[cov].append(beta[beta_index])

        self.history["omega"].append(omega)

        for i, mi in enumerate(self.model.MI_names):
            self.history[mi].append(log_mi[i])

        for i, output in enumerate(self.model.outputs_names):
            self.history[output].append(res_var[i])

    def one_iteration(self, k: int) -> bool:
        """Perform one iteration of SAEM

        Args:
            k (int): the iteration number
        """

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
            target_res_var = self._simulated_annealing(current_res_var, target_res_var)

        new_residual_error_var = self._stochastic_approximation(
            current_res_var, target_res_var
        )

        self.model.update_res_var(new_residual_error_var)

        # 2. Update sufficient statistics with stochastic approximation
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
        # Update beta
        self.model.update_betas(new_beta)

        # Update omega with simulated annealing (phase 1)
        if k < self.nb_phase1_iterations:
            new_omega_diag = torch.diag(new_omega)
            current_omega_diag = torch.diag(self.model.omega_pop)
            annealed_omega_diag = self._simulated_annealing(
                current_omega_diag, new_omega_diag
            )
            new_omega = torch.diag(annealed_omega_diag)
        self.model.update_omega(new_omega)

        # 3. Update fixed effects MIs
        if self.model.nb_MI > 0:
            # This step is notoriously under-optimized
            target_log_MI_np = minimize(
                fun=self.MI_objective_function,
                x0=self.model.log_MI.squeeze().numpy(),
                method="L-BFGS-B",
                options={"maxfun": self.optim_max_fun},
            ).x
            target_log_MI = torch.from_numpy(target_log_MI_np)
            new_log_MI = self._stochastic_approximation(
                self.model.log_MI, target_log_MI
            )

            self.model.update_log_mi(new_log_MI)

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

        # Convergence check
        new_params: dict[str, torch.Tensor] = {
            "log_MI": self.model.log_MI,
            "population_betas": self.model.population_betas,
            "population_omega": self.model.omega_pop,
            "residual_error_var": self.model.residual_var,
        }
        is_converged = self._check_convergence(new_params)

        # store history
        self._append_history(
            self.model.population_betas,
            self.model.omega_pop,
            self.model.log_MI,
            self.model.residual_var,
        )

        # update prev_params for the next iteration's convergence check
        self.prev_params = new_params

        if self.verbose:
            print("Iter done")
        return is_converged

    def MI_objective_function(self, log_MI):
        log_MI_expanded = (
            torch.Tensor(log_MI).unsqueeze(0).repeat((self.model.nb_patients, 1))
        )
        if hasattr(self.model, "patients_pdk"):
            pdk_full = self.model.patients_pdk_full
        else:
            pdk_full = torch.Tensor()
        # Assemble the patient parameters in the right order: PDK, PDU, MI
        new_thetas = torch.cat(
            (
                pdk_full,
                torch.exp(
                    torch.cat(
                        (
                            self.current_log_pdu,
                            log_MI_expanded,
                        ),
                        dim=1,
                    ),
                ),
            ),
            dim=1,
        )
        predictions, _ = self.model.predict_outputs_from_theta(
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

        print("Phase 1 (exploration):")
        (
            self.convergence_plot_handle,
            self.convergence_plot_fig,
            self.convergence_plot_axes,
        ) = self._build_convergence_plot(
            indiv_figsize=self.plot_indiv_figsize, n_cols=self.plot_columns
        )
        for k in tqdm(range(1, self.nb_phase1_iterations)):
            # Run iteration, do not check for convergence in the exploration phase
            # Iteration 0 is in fact already done (initialization)
            _ = self.one_iteration(k)
            self.current_iteration = k
            if (self.live_plot) & (k % self.plot_frames == 0):
                self._update_convergence_plot()

        if self.nb_phase2_iterations > 0:
            self.current_phase = 2
            print("Phase 2 (smoothing):")
            for k in tqdm(
                range(
                    self.nb_phase1_iterations,
                    self.nb_phase1_iterations + self.nb_phase2_iterations,
                )
            ):
                # Run iteration
                is_converged = self.one_iteration(k)
                self.current_iteration = k

                if (self.live_plot) & (k % self.plot_frames == 0):
                    self._update_convergence_plot()
                # Check for convergence, and stop if criterion matched
                if is_converged:
                    self.consecutive_converged_iters += 1
                    if self.verbose:
                        print(
                            f"Convergence met. Consecutive iterations: {self.consecutive_converged_iters}/{self.patience}"
                        )
                    if self.consecutive_converged_iters >= self.patience:
                        print(
                            f"\nConvergence reached after {k + 1} iterations. Stopping early."
                        )
                        self._update_convergence_plot()
                        break
                else:
                    self.consecutive_converged_iters = 0
        self._update_convergence_plot()
        plt.close(self.convergence_plot_fig)
        return None

    def continue_iterating(self, nb_add_iters_ph1=0, nb_add_iters_ph2=0) -> None:
        """
        This method is to be used when the run method has already run and the user wants to further iterate.
        """
        if self.current_phase == 2:
            if nb_add_iters_ph1 > 0:
                print("Smoothing phase has started, cannot add phase 1 iterations.")
                nb_add_iters_ph1 = 0
        if self.current_phase == 1:
            if nb_add_iters_ph1 > 0:
                print("Continuing phase 1 (exploration):")
                for k in tqdm(range(self.nb_phase1_iterations + nb_add_iters_ph1)):
                    # Run iteration, do not check for convergence in the exploration phase
                    _ = self.one_iteration(k)

            print("Switching to Phase 2 (smoothing)")
            self.current_phase = 2

        if nb_add_iters_ph2 > 0:
            for k in tqdm(
                range(
                    self.nb_phase1_iterations
                    + self.nb_phase2_iterations
                    + nb_add_iters_ph1,
                    self.nb_phase1_iterations
                    + self.nb_phase2_iterations
                    + nb_add_iters_ph1
                    + nb_add_iters_ph2,
                )
            ):
                # Run iteration
                is_converged = self.one_iteration(k)
                # Check for convergence, and stop if criterion matched
                if is_converged:
                    self.consecutive_converged_iters += 1
                    if self.verbose:
                        print(
                            f"Convergence met. Consecutive iterations: {self.consecutive_converged_iters}/{self.patience}"
                        )
                    if self.consecutive_converged_iters >= self.patience:
                        print(
                            f"\nConvergence reached after {k + 1} iterations. Stopping early."
                        )
                        break
                else:
                    self.consecutive_converged_iters = 0
        return None

    def _build_convergence_plot(
        self,
        indiv_figsize: tuple[float, float] = (2.0, 1.2),
        n_cols: int = 3,
    ):
        """
        This method plots the evolution of the estimated parameters (MI, betas, omega, residual error variances) across iterations
        """
        history = self.history
        nb_MI: int = self.model.nb_MI
        nb_betas: int = self.model.nb_betas
        nb_omega_diag_params: int = self.model.nb_PDU
        nb_var_res_params: int = self.model.nb_outputs
        nb_plots = nb_MI + nb_betas + nb_omega_diag_params + nb_var_res_params
        nb_cols = n_cols
        nb_rows = int(np.ceil(nb_plots / nb_cols))
        maxiter = self.nb_phase1_iterations + self.nb_phase2_iterations
        fig, axes = plt.subplots(
            nrows=nb_rows,
            ncols=nb_cols,
            figsize=(
                nb_cols * indiv_figsize[0],
                nb_rows * indiv_figsize[1],
            ),
            squeeze=False,
            sharex="all",
        )

        self.traces = {}
        plot_idx: int = 0
        # Plot the MI parameters
        for mi_name in self.model.MI_names:
            row, col = plot_idx // nb_cols, plot_idx % nb_cols
            ax = axes[row, col]
            ax.set_xlim(0, maxiter)
            MI_history = [h.item() for h in history[mi_name]]
            (tr,) = ax.plot(
                MI_history,
            )
            if hasattr(self, "true_log_MI"):
                if self.true_log_MI is not None:
                    ax.axhline(
                        y=self.true_log_MI[mi_name],
                        linestyle="--",
                    )
            if hasattr(self, "training_ranges"):
                if self.training_ranges is not None:
                    ax.fill_between(
                        [0, maxiter],
                        self.training_ranges[mi_name]["low"],
                        self.training_ranges[mi_name]["high"],
                        alpha=0.25,
                    )
            ax.set_title("Model intrinsic {MI_name}")
            ax.grid(True)
            self.traces.update({mi_name: tr})
            plot_idx += 1
        # Plot the PDUs means
        for pdu in self.model.PDU_names:
            row, col = plot_idx // nb_cols, plot_idx % nb_cols
            ax = axes[row, col]
            ax.set_xlim(0, maxiter)
            beta_history = [h.item() for h in history[pdu]["mu"]]
            (tr,) = ax.plot(
                beta_history,
            )
            if hasattr(self, "true_log_PDUs"):
                if self.true_log_PDUs is not None:
                    ax.axhline(
                        y=self.true_log_PDUs[pdu]["mean"],
                        linestyle="--",
                    )
            if hasattr(self, "training_ranges"):
                if self.training_ranges is not None:
                    ax.fill_between(
                        [0, maxiter],
                        self.training_ranges[pdu]["low"],
                        self.training_ranges[pdu]["high"],
                        alpha=0.25,
                    )
            ax.set_title(rf"{pdu}: $\mu$ (log)")
            ax.set_xlabel("")
            ax.grid(True)
            self.traces.update({pdu: {"mu": tr}})
            plot_idx += 1
        # Plot the PDUs sigma
        for pdu in self.model.PDU_names:
            row, col = plot_idx // nb_cols, plot_idx % nb_cols
            ax = axes[row, col]
            ax.set_xlim(0, maxiter)
            beta_history = [h.item() for h in history[pdu]["sigma_sq"]]
            (tr,) = ax.plot(
                beta_history,
            )
            if hasattr(self, "true_log_PDUs"):
                if self.true_log_PDUs is not None:
                    ax.axhline(
                        y=self.true_log_PDUs[pdu]["sd"],
                        linestyle="--",
                    )
            ax.set_title(rf"{pdu}: $\sigma^2$")
            ax.set_xlabel("")
            ax.grid(True)
            self.traces[pdu].update({"sigma_sq": tr})
            plot_idx += 1
        # Plot the coefficients of covariation
        for beta_name in self.model.covariate_coeffs_names:
            row, col = plot_idx // nb_cols, plot_idx % nb_cols
            ax = axes[row, col]
            ax.set_xlim(0, maxiter)
            beta_history = [h.item() for h in history[beta_name]]
            (tr,) = ax.plot(
                beta_history,
            )
            if hasattr(self, "true_cov"):
                if self.true_cov is not None:
                    ax.axhline(
                        y=self.true_cov[beta_name],
                        linestyle="--",
                    )
            ax.set_title(rf"{beta_name}")
            ax.set_xlabel("")
            ax.grid(True)
            self.traces.update({beta_name: tr})
            plot_idx += 1
        # Plot the residual variance
        for res_name in self.model.outputs_names:
            row, col = plot_idx // nb_cols, plot_idx % nb_cols
            ax = axes[row, col]
            ax.set_xlim(0, maxiter)
            var_res_history = [h.item() for h in history[res_name]]
            (tr,) = ax.plot(
                var_res_history,
            )
            if hasattr(self, "true_res_var"):
                if self.true_res_var is not None:
                    ax.axhline(
                        y=self.true_res_var[res_name],
                        linestyle="--",
                    )
            ax.set_title(rf"{res_name}: $\sigma^2$")
            ax.grid(True)
            self.traces.update({res_name: tr})
            plot_idx += 1
        # Turn off extra subplots
        while plot_idx < nb_rows * nb_cols:
            row, col = plot_idx // nb_cols, plot_idx % nb_cols
            ax = axes[row, col]
            ax.set_visible(False)
            plot_idx += 1
        if not smoke_test:
            plt.tight_layout()
            handle = display(fig, display_id=True)
        else:
            handle = None
        return (handle, fig, axes)

    def _update_convergence_plot(self):
        history = self.history
        new_xaxis = np.arange(self.current_iteration + 1)
        # Plot the MI parameters
        for mi_name in self.model.MI_names:
            MI_history = [h.item() for h in history[mi_name]]
            self.traces[mi_name].set_data(new_xaxis, MI_history)
        # Plot the PDUs means
        for pdu in self.model.PDU_names:
            beta_history = [h.item() for h in history[pdu]["mu"]]
            self.traces[pdu]["mu"].set_data(new_xaxis, beta_history)
        # Plot the PDUs sigma
        for pdu in self.model.PDU_names:
            beta_history = [h.item() for h in history[pdu]["sigma_sq"]]
            self.traces[pdu]["sigma_sq"].set_data(new_xaxis, beta_history)
        # Plot the coefficients of covariation
        for beta_name in self.model.covariate_coeffs_names:
            beta_history = [h.item() for h in history[beta_name]]
            self.traces[beta_name].set_data(new_xaxis, beta_history)
        # Plot the residual variance
        for res_name in self.model.outputs_names:
            var_res_history = [h.item() for h in history[res_name]]
            self.traces[res_name].set_data(new_xaxis, var_res_history)

        for ax in self.convergence_plot_axes.flatten():
            ax.autoscale_view(scaley=True, scalex=False)
            ax.relim()
        if self.convergence_plot_handle is not None:
            self.convergence_plot_handle.update(self.convergence_plot_fig)

    def plot_convergence_history(
        self,
        indiv_figsize: tuple[float, float] = (2.0, 1.2),
        n_cols: int = 3,
    ):
        handle, fig, axes = self._build_convergence_plot(
            indiv_figsize=indiv_figsize, n_cols=n_cols
        )
        plt.close(fig)

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
        simulated_tensor, _ = self.model.predict_outputs_from_theta(
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

        if not smoke_test:
            plt.tight_layout()
            plt.show()
