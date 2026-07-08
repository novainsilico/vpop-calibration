from tqdm.notebook import tqdm
import torch
import pandas as pd
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import uuid


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
        self.max_samples = self.model.config.max_samples
        if smoke_test:
            self.max_samples = 2

    def init_samples(self):

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
        init_physical = self.model.convert_gaussian_to_physical(
            psi=self.current_state.gaussian_params, log_mi=self.model.log_mi
        )
        init_samples = ConditionalDistribSamples(
            eta_samples=init_etas,
            physical_params_samples=init_physical,
            predictions=self.current_state.prediction,
            log_prob=self.current_state.log_prob,
        )
        self.samples: list[ConditionalDistribSamples] = [init_samples]
        self.ebe: ConditionalDistribSamples = init_samples
        self.nb_improved_history: list[float] = [0]
        self.mean_improved_history: list[float] = [0]
        self.indiv_log_prob: np.ndarray = init_samples.log_prob.cpu().numpy()

    def run_sampler(self, nb_samples: int = 100):
        if not hasattr(self, "ebe"):
            self.init_samples()
        else:
            print(f"Sampling already started, adding {nb_samples} new samples.")

        if self.live_plot:
            self.build_convergence_plot()

        if smoke_test:
            nb_samples = 2
        try:
            for i in self.sampling_stream(nb_samples):
                if self.live_plot:
                    self.update_convergence_plot()
            plt.close(self.fig)
        except KeyboardInterrupt:
            print("Interrupting sampling.")
            plt.close(self.fig)

    def sampling_stream(self, nb_samples: int):
        for i in tqdm(range(nb_samples), disable=not self.progress_bar):
            self.current_state = mh_step(
                self.model, previous_state=self.current_state, learning_rate=0.0
            )
            new_physical = self.model.convert_gaussian_to_physical(
                psi=self.current_state.gaussian_params, log_mi=self.model.log_mi
            )
            new_samples = ConditionalDistribSamples(
                eta_samples=self.current_state.etas,
                physical_params_samples=new_physical,
                predictions=self.current_state.prediction,
                log_prob=self.current_state.log_prob,
            )
            self.samples.append(new_samples)
            # Clip the list of samples to keep only the last max_samples values
            self.update_ebe(new_samples)
            self.clip_samples()
            yield i

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
        self.indiv_log_prob = np.concat((self.indiv_log_prob, new_log_prob), axis=0)

    def clip_samples(self):
        self.nb_improved_history = self.nb_improved_history[-self.max_samples :]
        self.indiv_log_prob = self.indiv_log_prob[-self.max_samples :, :]
        self.samples = self.samples[-self.max_samples :]

    def build_convergence_plot(self, plot_indiv_figsize=(5.0, 5.0)):
        self.fig, self.axes = plt.subplots(
            2, 1, figsize=plot_indiv_figsize, sharex=True
        )

        for ax in self.axes:
            ax.grid(True)

        self.axes[0].set_title("EBE convergence")
        self.axes[0].set_ylabel("Patients improved")

        self.axes[1].set_ylabel("Individual LL")
        self.axes[1].set_xlabel("Iteration")

        (line1_raw,) = self.axes[0].plot([], color="lightgray", linewidth=1)
        (line1_ma,) = self.axes[0].plot([], linewidth=2)
        patient_lines = []
        for _ in range(self.model.nb_patients):
            (line,) = self.axes[1].plot([], linewidth=1)
            patient_lines.append(line)

        self.traces = {
            "num_improved": line1_raw,
            "num_improved_ma": line1_ma,
            "individual": patient_lines,
        }
        if not smoke_test:
            self.handle = display(self.fig, display_id=True)

    def update_convergence_plot(self):

        nb_iters_so_far = len(self.nb_improved_history)
        x = np.arange(nb_iters_so_far)

        num_improved_ma = moving_average(self.nb_improved_history, window=20)

        self.traces["num_improved"].set_data(
            x,
            self.nb_improved_history,
        )
        self.traces["num_improved_ma"].set_data(
            x,
            num_improved_ma,
        )
        for i in range(self.model.nb_patients):
            self.traces["individual"][i].set_data(x, self.indiv_log_prob[:, i])

        if not smoke_test:
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view()

            if hasattr(self, "handle"):
                if self.handle is not None:
                    self.handle.update(self.fig)

    @property
    def ebe_parameters_df(self) -> pd.DataFrame:
        theta = self.model.convert_physical_to_thetas_all_patients(
            self.ebe.physical_params_samples
        )
        df = self.add_unique_id(self.model.convert_theta_to_dataframe(theta))
        return df

    @property
    def ebe_predictions_df(self) -> pd.DataFrame:
        df = self.model.data.full_obs.to_pandas(prediction=self.ebe.predictions)
        return df

    @property
    def total_samples(self) -> ConditionalDistribSamples:
        eta = torch.cat([s.eta_samples for s in self.samples], dim=0)
        physical = torch.cat([s.physical_params_samples for s in self.samples])
        pred = torch.stack([s.predictions for s in self.samples], dim=0)
        log_prob = torch.stack([s.log_prob for s in self.samples])
        out = ConditionalDistribSamples(
            eta_samples=eta,
            physical_params_samples=physical,
            predictions=pred,
            log_prob=log_prob,
        )
        return out

    @property
    def total_samples_parameters_df(self) -> pd.DataFrame:
        all_df = []
        for sample in self.samples:
            this_sample_theta = self.model.convert_physical_to_thetas_all_patients(
                sample.physical_params_samples
            )
            this_sample_df = self.add_unique_id(
                self.model.convert_theta_to_dataframe(this_sample_theta)
            )
            all_df.append(this_sample_df)
        total_df = pd.concat(all_df)
        return total_df

    @property
    def total_samples_predictions_df(self) -> pd.DataFrame:
        all_df = []
        for i, sample in enumerate(self.samples):
            this_sample_df = self.add_unique_id(
                self.model.data.full_obs.to_pandas(prediction=sample.predictions)
            )
            this_sample_df["batch_id"] = i
            all_df.append(this_sample_df)
        total_df = pd.concat(all_df)
        return total_df

    def add_unique_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create  a new `id` column with unique values, store the patient id in `id_ref`."""
        out_df = df.rename(columns={"id": "id_ref"})
        new_ids = {patient: str(uuid.uuid4()) for patient in self.model.patients}
        out_df["id"] = out_df["id_ref"].map(new_ids)
        return out_df


def moving_average(x: list[float], window: int = 20) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if len(x) < window:
        return x_arr
    return np.convolve(x_arr, np.ones(window) / window, mode="same")
