import numpy as np
import torch

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.saem.scheduler import SaemScheduler
from vpop_calibration.saem.estimates import PopEstimates
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.config import smoke_test
from vpop_calibration.metropolis_hastings import MetropolisHastingsState


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
        init_step_size_scaled = self.config.init_step_size / np.sqrt(self.model.nb_pdu)
        self.scheduler = SaemScheduler(
            nb_iter_burnin=config.nb_iter_burn_in,
            nb_iter_learning=config.nb_iter_learning,
            nb_iter_smoothing=nb_iter_smoothing,
            init_step_size_adaptation=init_step_size_scaled,
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

    def update_pop_estimates(self, new_estimates: PopEstimates) -> None:
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
        self, new_estimates: PopEstimates, new_mh_state: MetropolisHastingsState
    ):
        self.mh_state = new_mh_state
        self.update_pop_estimates(new_estimates)
        # todo: add sufficient statistics update

    def init_state(self):
        """Initiate the optimizer state with first estimates. Ensure this function is called before the optimization starts."""
        output = self.model.log_posterior_etas_all_patients(
            self.model.eta_samples_chains
        )
        # Give an initial dummy estimate for the total likelihood
        init_likelihood = torch.tensor([0.0])
        init_mh_state = MetropolisHastingsState(
            etas=self.model.eta_samples_chains,
            gaussian_params=output.gaussian_params,
            prediction=output.predictions,
            log_prob=output.log_posterior,
            step_size=self.scheduler.mh_learning_rate,
            complete_likelihood=init_likelihood,
        )
        init_estimates = PopEstimates(
            beta=self.model.population_betas,
            omega=self.model.omega_pop,
            psi=output.gaussian_params,
            sigma=self.model.residual_var,
            complete_likelihood=init_likelihood,
        )
        self.update_optimizer_state(
            new_estimates=init_estimates, new_mh_state=init_mh_state
        )

    def run(self):
        # Inititate the SAEM state with current estimates and Metropolis Hastings state
        self.init_state()
        pass

    def iterate(self):
        pass
