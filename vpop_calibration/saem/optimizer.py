import numpy as np
import torch
from tqdm import tqdm

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.saem.scheduler import SaemScheduler
from vpop_calibration.saem.estimates import PopEstimates
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step
from vpop_calibration.saem.m_step import MStepState
from vpop_calibration.pynlme.residuals import sum_sq_residuals
from vpop_calibration.saem.utils import simulated_annealing, stochastic_approximation


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

    def update_optimizer_state(
        self,
        new_estimates: PopEstimates,
        new_mh_state: MetropolisHastingsState,
        new_sufficient_statistics: MStepState,
    ):
        """Utilitary function to update the optimizer state at the end of an iteration"""
        self.mh_state = new_mh_state
        self.update_pop_estimates_convergence_check(new_estimates)
        self.sufficient_statistics = new_sufficient_statistics

    def init_state(self):
        """Initiate the optimizer state with first estimates. Ensure this function is called before the optimization starts."""
        # Estimate the log-posterior on current eta samples
        output = self.model.log_posterior_etas_all_patients(
            self.model.eta_samples_chains
        )
        # Give an initial dummy estimate for the total likelihood
        init_likelihood = torch.tensor([0.0])
        # Initialize the step size by incorporating problem dimension
        init_step_size = self.config.init_step_size_unscaled / np.sqrt(
            self.model.nb_pdu
        )
        # Initialize the Metropolis Hastings state variables
        init_mh_state = MetropolisHastingsState(
            etas=self.model.eta_samples_chains,
            gaussian_params=output.gaussian_params,
            prediction=output.predictions,
            log_prob=output.log_posterior,
            step_size=init_step_size,
            complete_likelihood=init_likelihood,
        )
        init_estimates = PopEstimates(
            beta=self.model.population_betas,
            omega=self.model.omega_pop,
            psi=output.gaussian_params,
            sigma=self.model.residual_var,
            complete_likelihood=init_likelihood,
        )
        init_sufficient_stats = MStepState(
            design_matrix=self.model.full_design_matrix,
            init_gaussian_params=output.gaussian_params,
            nb_chains=self.model.nb_chains,
            nb_patients=self.model.nb_patients,
            nb_pdu=self.model.nb_pdu,
        )
        self.update_optimizer_state(
            new_estimates=init_estimates,
            new_mh_state=init_mh_state,
            new_sufficient_statistics=init_sufficient_stats,
        )

    def run(self):
        # Inititate the SAEM state with current estimates and Metropolis Hastings state
        self.init_state()
        for k in tqdm(self.scheduler):
            current_mh_state = self.mh_state
            # E-step
            current_mh_state = mh_step(
                nlme_model=self.model,
                previous_state=current_mh_state,
                learning_rate=self.scheduler.mh_learning_rate,
            )
            # M-step
            if self.scheduler.phase != "burnin":
                self.model.update_eta_samples(current_mh_state.etas)

                sum_sq_res_full = sum_sq_residuals(
                    prediction=current_mh_state.prediction,
                    observations=self.model.data.full_obs,
                    error_model_selector=self.model.error_model_selector,
                )
                sum_sq_res = sum_sq_res_full.sum(dim=0)
                assert sum_sq_res.shape == (
                    self.model.nb_outputs,
                ), f"Unexpected residual shape: {sum_sq_res.shape}"
                target_res_var: torch.Tensor = (
                    sum_sq_res / self.model.data.n_tot_observations_per_output
                )
                current_res_var: torch.Tensor = self.model.residual_var
                if self.scheduler.phase == "learning":
                    target_res_var = simulated_annealing(
                        current=current_res_var,
                        target=target_res_var,
                        factor=self.config.annealing_factor,
                    )

                new_residual_error_var = stochastic_approximation(
                    previous=current_res_var,
                    new=target_res_var,
                    learning_rate=self.scheduler.stochastic_approximation_rate,
                )

                self.model.update_res_var(new_residual_error_var)

                mstep_proposal = self.sufficient_statistics.update(
                    new_gaussian_params=current_mh_state.gaussian_params,
                    learning_rate=self.scheduler.stochastic_approximation_rate,
                )
                self.model.update_betas(mstep_proposal.beta)
                self.model.update_omega(mstep_proposal.omega)
            # MI optimization

    def iterate(self):
        pass
