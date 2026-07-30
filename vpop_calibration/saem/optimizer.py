import numpy as np
import torch
from tqdm.notebook import tqdm
from scipy.optimize import minimize
from typing import Callable, Any
import pandas as pd

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.saem.scheduler import SaemScheduler
from vpop_calibration.saem.estimates import PopEstimates, IterSummary, check_convergence
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step
from vpop_calibration.saem.m_step import MStepState
from vpop_calibration.saem.utils import (
    simulated_annealing,
    stochastic_approximation,
    covariance_matrix_simulated_annealing,
)
from vpop_calibration.config import device
from vpop_calibration.pynlme.residuals import (
    log_likelihood_observation,
    estimate_error_params,
)
from vpop_calibration.saem.plot import OptimizerPlot
from vpop_calibration.config import smoke_test


class PySaem:
    def __init__(
        self,
        model: StatisticalModel,
        config: SaemConfigDict,
    ):
        self.model: StatisticalModel = model
        self.config = config
        if self.config.nb_iter_smoothing is None:
            self.config = self.config._replace(
                nb_iter_smoothing=self.config.nb_iter_learning
            )
        assert self.config.nb_iter_smoothing is not None

        if smoke_test:
            # Override with test config
            self.config = self.config._replace(
                nb_iter_burnin=1,
                nb_iter_learning=2,
                nb_iter_smoothing=2,
                optim_max_fun=1,
                progress_bars=False,
                live_plot=False,
                logging=False,
            )

        self.consecutive_converged_iters = 0
        self.scheduler = SaemScheduler(
            nb_iter_burnin=self.config.nb_iter_burnin,
            nb_iter_learning=self.config.nb_iter_learning,
            nb_iter_smoothing=self.config.nb_iter_smoothing,
            init_step_adaptation=config.init_step_adaptation,
            learning_rate_power=config.learning_rate_power,
            patience=config.patience,
        )
        self.history = pd.DataFrame()

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
            ebe=output.gaussian_params.mean(dim=0),
            sigma=self.model.residual_var,
            complete_likelihood=init_likelihood,
            model_intrinsic=self.model.log_mi,
        )
        self.sufficient_statistics = MStepState.from_init_gaussian_params(
            design_matrix=self.model.full_design_matrix,
            init_gaussian_params=output.gaussian_params,
            nb_chains=self.model.nb_chains,
            nb_patients=self.model.nb_patients,
            nb_pdu=self.model.nb_pdu,
        )

    def get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            "config": self.config.get_state_dict(),
            "scheduler": self.scheduler.get_state_dict(),
            "history": self.history.to_dict(),
            "consecutive_converged_iters": self.consecutive_converged_iters,
        }
        if hasattr(self, "mh_state"):
            state_dict.update(
                {
                    "mh_state": self.mh_state.get_state_dict(),
                    "pop_estimates": self.pop_estimates.get_state_dict(),
                    "sufficient_statistics": self.sufficient_statistics.get_state_dict(),
                    "has_run": True,
                }
            )
        else:
            state_dict.update({"has_run": False})

        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], model: StatisticalModel
    ) -> "PySaem":
        instance = cls(
            model=model, config=SaemConfigDict.from_state_dict(state_dict["config"])
        )
        instance.scheduler = SaemScheduler.from_state_dict(
            state_dict=state_dict["scheduler"]
        )
        instance.history = pd.DataFrame.from_dict(state_dict["history"])
        instance.consecutive_converged_iters = state_dict["consecutive_converged_iters"]

        if state_dict["has_run"]:
            instance.mh_state = MetropolisHastingsState.from_state_dict(
                state_dict=state_dict["mh_state"]
            )
            instance.pop_estimates = PopEstimates.from_state_dict(
                state_dict=state_dict["pop_estimates"]
            )
            instance.sufficient_statistics = MStepState.from_state_dict(
                state_dict=state_dict["sufficient_statistics"]
            )
        return instance

    def run(self):
        if self.scheduler.iteration == 0:
            # Inititate the SAEM state with current estimates and Metropolis Hastings state
            self.init_state()
        else:
            print(f"Resuming at iteration {self.scheduler.iteration}:")
        try:
            for progress in self.optimization_stream():
                # Push history
                self.history = pd.concat([self.history, progress.to_pandas()])
                # Logging
                if self.config.logging:
                    if (progress.iteration % self.config.logging_frequency == 0) or (
                        progress.iteration == self.scheduler.nb_iter_tot - 1
                    ):
                        progress.print(width=self.config.column_width)
                # Live plotting
                if self.config.live_plot:
                    if (progress.iteration % self.config.plot_frames == 0) or (
                        progress.iteration == self.scheduler.nb_iter_tot - 1
                    ):
                        self.plot_history()

            if self.config.live_plot:
                self.plot.close()
        except KeyboardInterrupt:
            print(f"Interrupted gracefully at iteration {self.scheduler.iteration}.")

            if self.config.live_plot:
                self.plot.close()

    def optimization_stream(self):
        for _ in tqdm(
            self.scheduler,
            total=self.scheduler.nb_iter_tot,
            initial=self.scheduler.iteration,
            disable=not self.config.progress_bars,
        ):
            summary = self.step()
            yield summary

    def step(self) -> IterSummary:
        """One full iteration of SAEM.

        This function is implemented as a generator, yielding the iteration summary.
        """

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
            # maximum-likelihood target for the residual error variance
            target_res_var = estimate_error_params(
                observations=self.model.data.full_obs,
                predictions=self.mh_state.prediction,
                error_model_selector=self.model.error_model_selector,
                sigma=self.model.residual_var,
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

            # 3. Update fixed effects MIs
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

        new_ebe = stochastic_approximation(
            previous=self.pop_estimates.ebe,
            new=self.mh_state.gaussian_params.mean(dim=0),
            learning_rate=self.scheduler.stochastic_approximation_rate,
        )

        # Update population estimates and check for early convergence
        new_estimates = PopEstimates(
            beta=self.model.population_betas,
            omega=self.model.omega_pop,
            ebe=new_ebe,
            sigma=self.model.residual_var,
            model_intrinsic=self.model.log_mi,
            complete_likelihood=self.mh_state.complete_likelihood,
        )
        self.update_pop_estimates_convergence_check(new_estimates=new_estimates)
        # Assemble the iteration summary
        summary = IterSummary.from_pop_estimates(
            iteration=self.scheduler.iteration,
            estimates=new_estimates,
            beta_names=self.model.beta_names,
            pdu_names=self.model.pdu_names,
            covariate_coeff_names=self.model.covariate_coeff_names,
            mi_names=self.model.mi_names,
            output_names=self.model.output_names,
            error_model_selector=self.model.error_model_selector,
        )
        return summary

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
                    sigma=self.model.residual_var,
                )
                .cpu()
                .sum()
                .item()
            )

            return -total_log_lik

        return mi_objective_function

    def update_pop_estimates_convergence_check(
        self, new_estimates: PopEstimates
    ) -> None:
        """Update the optimizer state with new population estimates, also updating the number of converged iterations."""

        if not hasattr(self, "pop_estimates"):
            # This is the first iteration
            self.pop_estimates = new_estimates
            converged = False
        else:
            self.previous_estimates = self.pop_estimates
            self.pop_estimates = new_estimates
            converged = check_convergence(
                prev_est=self.previous_estimates,
                current_est=self.pop_estimates,
                threshold=self.config.convergence_threshold,
            )

        if converged:
            self.consecutive_converged_iters += 1
        else:
            self.consecutive_converged_iters = 0

    def plot_history(self):
        if not hasattr(self, "plot"):
            self.plot = OptimizerPlot(
                self.history,
                nb_tot_iter=self.scheduler.nb_iter_tot,
                facet_size=self.config.facet_size,
                nb_cols=self.config.plot_columns,
            )
        else:
            self.plot.update(self.history)
