from tqdm import tqdm
import torch
import pandas as pd
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np


from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.config import smoke_test
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step


class ConditionalDistribSamples(NamedTuple):
    eta_samples: torch.Tensor
    physical_params_samples: torch.Tensor
    predictions: torch.Tensor
    log_prob: torch.Tensor


class ConditionalDistributionSampler:
    def __init__(
        self,
        nlme_model: StatisticalModel,
    ):
        self.model = nlme_model
        self.live_plot = self.model.config.live_plot
        self.progress_bar = self.model.config.progress_bar
        self.plot_frequency = self.model.config.plot_frequency

    def init_samples(self):
        if smoke_test:
            nb_samples = 2

        if self.live_plot:
            self.build_convergence_plot()

        # Initiate samples
        init_etas = self.model.sample_etas(1)
        init_predictions = self.model.log_posterior_etas_all_patients(init_etas)
        self.current_state = MetropolisHastingsState(
            etas=init_etas,
            gaussian_params=init_predictions.gaussian_params,
            prediction=init_predictions.predictions,
            log_prob=init_predictions.log_posterior,
            step_size=0.1,
            complete_likelihood=init_predictions.predictions.sum(dim=0),
        )
        init_samples = ConditionalDistribSamples(
            eta_samples=init_etas,
            physical_params_samples=self.current_state.gaussian_params,
            predictions=self.current_state.prediction,
            log_prob=self.current_state.log_prob,
        )
        self.samples: list[ConditionalDistribSamples] = [init_samples]
        self.ebe: ConditionalDistribSamples = init_samples
        self.nb_improved_history: list[float] = [0]
        self.mean_improved_history: list[float] = [0]

    def run_sampler(self, nb_samples: int = 100):
        self.init_samples()

        for _ in tqdm(range(nb_samples), disable=not self.progress_bar):
            self.current_state = mh_step(
                self.model, previous_state=self.current_state, learning_rate=0.0
            )
            new_samples = ConditionalDistribSamples(
                eta_samples=self.current_state.etas,
                physical_params_samples=self.current_state.gaussian_params,
                predictions=self.current_state.prediction,
                log_prob=self.current_state.log_prob,
            )
            self.samples.append(new_samples)
            self.update_ebe(new_samples)
            if self.live_plot:
                self.update_convergence_plot()

    def update_ebe(self, new_samples: ConditionalDistribSamples):
        # Assemble as mask to accept or reject new EBEs
        accept_mask = self.ebe.log_prob < new_samples.log_prob
        # size (nb_patients)
        new_eta = torch.where(
            accept_mask.view(-1, 1), new_samples.eta_samples, self.ebe.eta_samples
        )
        new_physical = torch.where(
            accept_mask.view(-1, 1),
            new_samples.physical_params_samples,
            self.ebe.physical_params_samples,
        )
        accept_mask_predictions = accept_mask.index_select(
            1, self.model.data.full_obs.obs_index.id.index_values
        )
        new_pred = torch.where(
            accept_mask_predictions, new_samples.predictions, self.ebe.predictions
        )
        new_log_prob = torch.where(accept_mask, new_samples.log_prob, self.ebe.log_prob)
        self.ebe = ConditionalDistribSamples(
            eta_samples=new_eta,
            physical_params_samples=new_physical,
            predictions=new_pred,
            log_prob=new_log_prob,
        )

        nb_improved = accept_mask.float().sum().item()
        self.nb_improved_history.append(nb_improved)
        mean_improved = accept_mask.float().mean().item()
        self.mean_improved_history.append(mean_improved)

    def build_convergence_plot(self, plot_indiv_figsize=(5.0, 3.0)):
        self.fig, self.axes = plt.subplots(
            2, 1, figsize=plot_indiv_figsize, sharex=True
        )

        for ax in self.axes:
            ax.grid(True)

        self.axes[0].set_title("EBE convergence")

        self.axes[0].set_ylabel("Patients improved")
        self.axes[1].set_ylabel("Mean LL gain")
        self.axes[1].set_xlabel("Iteration")

        (line1_raw,) = self.axes[0].plot([], color="lightgray", linewidth=1)
        (line1_ma,) = self.axes[0].plot([], linewidth=2)
        (line2_raw,) = self.axes[1].plot([], color="lightgray", linewidth=1)
        (line2_ma,) = self.axes[1].plot([], linewidth=2)

        self.traces = {
            "num_improved": line1_raw,
            "num_improved_ma": line1_ma,
            "mean_gain": line2_raw,
            "mean_gain_ma": line2_ma,
        }
        if not smoke_test:
            self.handle = display(self.fig, display_id=True)

    def update_convergence_plot(self):

        x = np.arange(len(self.nb_improved_history))

        mean_gain_ma = moving_average(self.mean_improved_history, window=20)
        num_improved_ma = moving_average(self.nb_improved_history, window=20)

        self.traces["num_improved"].set_data(
            x,
            self.nb_improved_history,
        )
        self.traces["num_improved_ma"].set_data(
            x,
            num_improved_ma,
        )
        self.traces["mean_gain"].set_data(
            x,
            self.mean_improved_history,
        )
        self.traces["mean_gain_ma"].set_data(
            x,
            mean_gain_ma,
        )

        if not smoke_test:
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view()

        if hasattr(self, "handle"):
            if self.handle is not None:
                self.handle.update(self.fig)


def moving_average(x: list[float], window: int = 20) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if len(x) < window:
        return x_arr
    return np.convolve(x_arr, np.ones(window) / window, mode="same")
