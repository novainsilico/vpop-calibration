import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import minimize
from typing import Callable

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.saem.scheduler import SaemScheduler
from vpop_calibration.saem.estimates import PopEstimates
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step
from vpop_calibration.saem.m_step import MStepState
from vpop_calibration.pynlme.residuals import sum_sq_residuals
from vpop_calibration.saem.utils import (
    simulated_annealing,
    stochastic_approximation,
    covariance_matrix_simulated_annealing,
)
from vpop_calibration.config import device
from vpop_calibration.pynlme.residuals import log_likelihood_observation


class PySaem:
    def __init__(
        self,
        model: StatisticalModel,
        config: SaemConfigDict,
    ):
        self.model: StatisticalModel = model
        self.config = config
        if config.nb_iter_smoothing is None:
            nb_iter_smoothing = config.nb_iter_learning
        else:
            nb_iter_smoothing = config.nb_iter_smoothing
        self.consecutive_converged_iters = 0
        self.scheduler = SaemScheduler(
            nb_iter_burnin=config.nb_iter_burn_in,
            nb_iter_learning=config.nb_iter_learning,
            nb_iter_smoothing=nb_iter_smoothing,
            init_step_adaptation=config.init_step_adaptation,
            learning_rate_power=config.learning_rate_power,
            patience=config.patience,
        )

    def check_convergence(self, prev_est: PopEstimates, current_est: PopEstimates):
        """Checks for convergence based on the relative change in parameters."""
        all_converged = True
        variables_to_check = ["beta", "omega", "psi", "sigma"]
        for name in variables_to_check:
            current_val = current_est._asdict()[name]
            prev_val = prev_est._asdict()[name]
            abs_diff = torch.abs(current_val - prev_val)
            abs_sum = torch.abs(current_val) + torch.abs(prev_val) + 1e-9
            relative_change = abs_diff / abs_sum
            if torch.any(relative_change > self.config.convergence_threshold):
                all_converged = False
                break
        return all_converged

    def update_pop_estimates_convergence_check(
        self, new_estimates: PopEstimates
    ) -> None:
        """Update the optimizer state with new population estimates, also updating the number of converged iterations."""

        if not hasattr(self, "current_estimates"):
            # This is the first iteration
            self.current_estimates = new_estimates
            converged = False
        else:
            self.previous_estimates = self.current_estimates
            self.current_estimates = new_estimates
            converged = self.check_convergence(
                self.previous_estimates,
                self.current_estimates,
            )

        if converged:
            self.consecutive_converged_iters += 1
        else:
            self.consecutive_converged_iters = 0

    def init_state(self):
        """Initiate the optimizer state with first estimates. Ensure this function is called before the optimization starts."""
        # Estimate the log-posterior on current eta samples
        init_samples = self.model.sample_etas(self.model.nb_chains)
        output = self.model.log_posterior_etas_all_patients(init_samples)
        # Give an initial dummy estimate for the total likelihood
        init_likelihood = torch.tensor([0.0])
        # Initialize the step size by incorporating problem dimension
        init_step_size = self.config.init_step_size_unscaled / np.sqrt(
            self.model.nb_pdu
        )
        # Initialize the Metropolis Hastings state variables
        self.mh_state = MetropolisHastingsState(
            etas=init_samples,
            gaussian_params=output.gaussian_params,
            prediction=output.predictions,
            log_prob=output.log_posterior,
            step_size=init_step_size,
            complete_likelihood=init_likelihood,
        )
        self.pop_estimates = PopEstimates(
            beta=self.model.population_betas,
            omega=self.model.omega_pop,
            psi=output.gaussian_params,
            sigma=self.model.residual_var,
            complete_likelihood=init_likelihood,
            model_intrinsic=self.model.log_mi,
        )
        self.sufficient_statistics = MStepState(
            design_matrix=self.model.full_design_matrix,
            init_gaussian_params=output.gaussian_params,
            nb_chains=self.model.nb_chains,
            nb_patients=self.model.nb_patients,
            nb_pdu=self.model.nb_pdu,
        )

    def run(self):
        # Inititate the SAEM state with current estimates and Metropolis Hastings state
        self.init_state()

        # Iterate with the scheduler
        for k in tqdm(self.scheduler):
            self.step()
            # todo: add logs, history and live plotting

    def step(self):
        """One full iteration of SAEM."""

        # Temporarily store the mh state to iterate over it
        current_mh_state = self.mh_state
        # E-step: run Metropolis Hastings transitions
        for _ in range(self.config.nb_mcmc_transitions):
            current_mh_state = mh_step(
                nlme_model=self.model,
                previous_state=current_mh_state,
                learning_rate=self.scheduler.mh_learning_rate,
            )
        # Update the optimizer
        self.mh_state = current_mh_state

        # If in learning or smoothing phase, go through the rest of the iteration
        if self.scheduler.phase != "burnin":
            # M-step:
            # Compute the sum of squared residuals
            sum_sq_res_full = sum_sq_residuals(
                prediction=self.mh_state.prediction,
                observations=self.model.data.full_obs,
                error_model_selector=self.model.error_model_selector,
            )
            sum_sq_res = sum_sq_res_full.sum(dim=0)
            assert sum_sq_res.shape == (
                self.model.nb_outputs,
            ), f"Unexpected residual shape: {sum_sq_res.shape}"

            # Update the residual error variance
            target_res_var: torch.Tensor = (
                sum_sq_res / self.model.data.n_tot_observations_per_output
            )
            current_res_var: torch.Tensor = self.model.residual_var

            if self.scheduler.phase == "learning":
                # Simulated annealing is only considered in learning phase
                target_res_var = simulated_annealing(
                    current=current_res_var,
                    target=target_res_var,
                    factor=self.config.annealing_factor,
                )

            new_res_error_var = stochastic_approximation(
                previous=current_res_var,
                new=target_res_var,
                learning_rate=self.scheduler.stochastic_approximation_rate,
            )

            self.model.update_res_var(new_res_error_var)

            # Propose new values for beta and omega
            mstep_proposal = self.sufficient_statistics.update(
                new_gaussian_params=self.mh_state.gaussian_params,
                learning_rate=self.scheduler.stochastic_approximation_rate,
            )
            self.model.update_betas(mstep_proposal.beta)
            # Applying simulated annealing to omega, if in learning phase
            if self.scheduler.phase == "learning":
                new_omega = covariance_matrix_simulated_annealing(
                    current_omega=self.model.omega_pop,
                    target_omega=mstep_proposal.omega,
                    factor=self.config.annealing_factor,
                )
            else:
                new_omega = mstep_proposal.omega
            self.model.update_omega(new_omega)

            # MI optimization        # 3. Update fixed effects MIs
            if self.model.nb_mi > 0:
                # This step is notoriously under-optimized
                objective_fun = self.build_mi_objective_function(
                    self.mh_state.gaussian_params
                )
                target_log_MI_np = minimize(
                    fun=objective_fun,
                    x0=self.model.log_mi.cpu().squeeze().numpy(),
                    method="Nelder-Mead",
                    options={"maxiter": self.config.optim_max_fun},
                ).x
                target_log_MI = torch.from_numpy(target_log_MI_np).to(device)
                new_log_MI = stochastic_approximation(
                    previous=self.model.log_mi,
                    new=target_log_MI,
                    learning_rate=self.scheduler.stochastic_approximation_rate,
                )

                self.model.update_log_mi(new_log_MI)

            # Update population estimates and check for early convergence
            new_estimates = PopEstimates(
                beta=self.model.population_betas,
                omega=self.model.omega_pop,
                psi=self.mh_state.gaussian_params,
                sigma=self.model.residual_var,
                model_intrinsic=self.model.log_mi,
                complete_likelihood=self.mh_state.complete_likelihood,
            )
            self.update_pop_estimates_convergence_check(new_estimates=new_estimates)

    def build_mi_objective_function(self, gaussian_params: torch.Tensor) -> Callable:
        """Build the objective function to be optimized for model intrinsic parameters estimation."""

        def mi_objective_function(log_MI: np.ndarray):
            mi_tensor = torch.from_numpy(log_MI).to(device)
            # Assemble the patient parameters
            new_physical_params = self.model.convert_gaussian_to_physical(
                gaussian_params, mi_tensor
            )
            new_thetas = self.model.convert_physical_to_thetas_all_patients(
                new_physical_params
            )
            model_input = self.model.convert_thetas_to_model_parameters_all_patients(
                new_thetas
            )
            predictions, _ = self.model.predict_all_patients(model_input)
            total_log_lik = (
                log_likelihood_observation(
                    predictions=predictions,
                    observations=self.model.data.full_obs,
                    error_model_selector=self.model.error_model_selector,
                    sigma=self.model.residual_var,
                )
                .cpu()
                .sum()
                .item()
            )

            return -total_log_lik

        return mi_objective_function
