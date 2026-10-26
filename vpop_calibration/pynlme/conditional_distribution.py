from math import ceil

from vpop_calibration.compatibility import tqdm
import torch
import pandas as pd
from typing import NamedTuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from IPython.display import display
except ImportError:
    display = None


from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.config import smoke_test, device, default_dtype
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step
from vpop_calibration.utils import reproducible_uuid4


class ConditionalDistribSamples(NamedTuple):
    eta_samples: torch.Tensor
    physical_params_samples: torch.Tensor
    predictions: torch.Tensor
    log_prob: torch.Tensor

    def get_state_dict(self) -> dict[str, Any]:
        return {k: v.detach().cpu().numpy().tolist() for k, v in self._asdict().items()}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "ConditionalDistribSamples":
        return cls(
            **{
                k: torch.as_tensor(v, device=device, dtype=default_dtype)
                for k, v in state_dict.items()
            }
        )


class ConditionalDistributionSampler:
    def __init__(self, nlme_model: StatisticalModel):
        self.model = nlme_model
        self.live_plot = self.model.config.live_plot
        self.progress_bar = self.model.config.progress_bar
        self.plot_frequency = self.model.config.plot_frequency
        self.max_samples = self.model.config.max_samples

    def init_samples(self, optimizer_last_state: MetropolisHastingsState | None = None):
        if optimizer_last_state is not None:
            # Initialize the MCMC sampler from the SAEM optimizer's last state
            self.current_state = MetropolisHastingsState(
                etas=optimizer_last_state.etas,
                gaussian_params=optimizer_last_state.gaussian_params,
                prediction=optimizer_last_state.prediction,
                log_prob=optimizer_last_state.log_prob,
                step_size=optimizer_last_state.step_size,
                complete_likelihood=optimizer_last_state.complete_likelihood,
            )
        else:
            # No state provided so we just sample fresh etas
            # Initiate samples
            init_etas = self.model.sample_etas(self.model.nb_chains)
            init_predictions = self.model.log_posterior_etas_all_patients(init_etas)
            self.current_state = MetropolisHastingsState(
                etas=init_etas,
                gaussian_params=init_predictions.gaussian_params,
                prediction=init_predictions.predictions,
                log_prob=init_predictions.log_posterior,
                step_size=0.1,
                complete_likelihood=-2
                * init_predictions.log_posterior.mean(dim=0).sum(dim=0),
            )
        init_physical = self.model.convert_gaussian_to_physical(
            psi=self.current_state.gaussian_params,
            log_mi=self.model.log_mi,
            surv_coeffs=self.model.surv_coeffs,
        )
        init_samples = ConditionalDistribSamples(
            eta_samples=self.current_state.etas,
            physical_params_samples=init_physical,
            predictions=self.current_state.prediction,
            log_prob=self.current_state.log_prob,
        )
        self.samples: list[ConditionalDistribSamples] = [init_samples]
        self.map: ConditionalDistribSamples = self.best_log_prob_over_chains(
            init_samples
        )
        self.nb_improved_history: list[float] = [0]
        self.indiv_log_prob: np.ndarray = self.map.log_prob.detach().cpu().numpy()

    def get_state_dict(self) -> dict[str, Any]:
        if hasattr(self, "map"):
            state_dict = {
                "current_state": self.current_state.get_state_dict(),
                "map": self.map.get_state_dict(),
                "last_sample": self.samples[-1].get_state_dict(),
                "has_run": True,
            }
        else:
            state_dict = {"has_run": False}
        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], model: StatisticalModel
    ) -> "ConditionalDistributionSampler":
        instance = cls(model)
        if state_dict["has_run"]:
            instance.current_state = MetropolisHastingsState.from_state_dict(
                state_dict=state_dict["current_state"]
            )
            init_samples = ConditionalDistribSamples.from_state_dict(
                state_dict=state_dict["last_sample"]
            )
            instance.samples = [init_samples]
            instance.map = ConditionalDistribSamples.from_state_dict(
                state_dict=state_dict["map"]
            )
            instance.nb_improved_history = [0]
            instance.indiv_log_prob = init_samples.log_prob.detach().cpu().numpy()

        return instance

    def run_sampler(
        self,
        nb_samples: int = 100,
        optimizer_last_state: MetropolisHastingsState | None = None,
    ):
        # How many MCMC chains
        if optimizer_last_state is not None:
            nc = optimizer_last_state.etas.shape[0]
        else:
            nc = self.model.nb_chains
        # We will run the conditional sampler for nb_iter which will
        # produce nc x nb_iter samples >= nb_samples. All of them are kept, as
        # the MAP may be among the extra ones
        nb_iter = ceil(nb_samples / nc)
        if not hasattr(self, "map"):
            self.init_samples(optimizer_last_state)
            # we reuse the last optimizer state to initialize the list of
            # samples so we need one fewer iteration
            nb_iter -= 1
        else:
            nb_existing_samples = len(self.samples) * nc
            # Wrap this in a proper logger
            print(
                f"{nb_existing_samples} conditional samples already in cache, running the sampler for {nb_iter} more iterations."
            )

        if self.live_plot:
            self.build_convergence_plot()

        if smoke_test:
            nb_iter = 1
        try:
            for i in self.sampling_stream(nb_iter):
                if self.live_plot and i % ceil(self.plot_frequency / nc) == 0:
                    self.update_convergence_plot()
            if self.live_plot:
                plt.close(self.fig)
        except KeyboardInterrupt:
            print("Interrupting sampling.")
            if self.live_plot:
                self.update_convergence_plot()
                plt.close(self.fig)

    def sampling_stream(self, nb_iter: int):
        for i in tqdm(range(nb_iter), disable=not self.progress_bar):
            self.current_state = mh_step(
                self.model, previous_state=self.current_state, learning_rate=0.0
            )
            new_physical = self.model.convert_gaussian_to_physical(
                psi=self.current_state.gaussian_params,
                log_mi=self.model.log_mi,
                surv_coeffs=self.model.surv_coeffs,
            )
            new_samples = ConditionalDistribSamples(
                eta_samples=self.current_state.etas,
                physical_params_samples=new_physical,
                predictions=self.current_state.prediction,
                log_prob=self.current_state.log_prob,
            )
            self.samples.append(new_samples)
            best_over_chains = self.best_log_prob_over_chains(new_samples)
            self.update_map(best_over_chains)
            # Clip the list of samples to keep only the last max_samples values
            self.clip_samples()
            yield i

    def best_log_prob_over_chains(self, samples: ConditionalDistribSamples):
        """
        Takes a sample with etas @ (C, P, D) and returns a sample with etas @ (1, P, D) that
        maximizes the log_prob for each patient across all chains
        """
        best_log_prob, best_indices = samples.log_prob.max(dim=0)  # (P,)
        best_indices_obs = best_indices.index_select(
            dim=0, index=self.model.data.full_obs.obs_index.id.index_values
        )  # (Y,)
        return ConditionalDistribSamples(
            eta_samples=samples.eta_samples.take_along_dim(
                indices=best_indices[None, :, None], dim=0
            ),  # (1, P, D)
            physical_params_samples=samples.physical_params_samples.take_along_dim(
                indices=best_indices[None, :, None], dim=0
            ),  # (1, P, D + F)
            predictions=samples.predictions.take_along_dim(
                indices=best_indices_obs[None, :], dim=0
            ),  # (1, Y)
            log_prob=best_log_prob.unsqueeze(0),  # (1, P)
        )

    def update_map(self, best_over_chains: ConditionalDistribSamples):
        accept_mask = self.map.log_prob < best_over_chains.log_prob  # (1, P)
        assert accept_mask.shape == (1, self.model.nb_patients), f"{accept_mask.shape=}"
        # Update the eta samples of the MAP:
        # take any sample across the chains that is better than MAP, otherwise
        # keep the current MAP
        new_eta = torch.where(
            accept_mask[:, :, None],  # (1, P, 1)
            best_over_chains.eta_samples,
            self.map.eta_samples,
        )  # (1, P, D)
        new_physical = torch.where(
            accept_mask[:, :, None],  # (1, P, 1)
            best_over_chains.physical_params_samples,
            self.map.physical_params_samples,
        )  # (1, P, D + F)
        accept_mask_predictions = accept_mask.index_select(
            dim=1, index=self.model.data.full_obs.obs_index.id.index_values
        )  # (1, Y)
        new_pred = torch.where(
            accept_mask_predictions, best_over_chains.predictions, self.map.predictions
        )  # (1, Y)
        new_log_prob = torch.where(
            accept_mask, best_over_chains.log_prob, self.map.log_prob
        )  # (1, P)
        self.map = ConditionalDistribSamples(
            eta_samples=new_eta,
            physical_params_samples=new_physical,
            predictions=new_pred,
            log_prob=new_log_prob,
        )
        # How many patients have had their MAP improved this time, used for progress plot
        nb_improved = accept_mask.sum().item()
        self.nb_improved_history.append(nb_improved)
        self.indiv_log_prob = np.concat(
            (self.indiv_log_prob, new_log_prob.detach().cpu().numpy()), axis=0
        )  # (N, P)

    def clip_samples(self):
        # max_samples counts iterations: up to C * max_samples samples are kept
        self.nb_improved_history = self.nb_improved_history[-self.max_samples :]
        self.indiv_log_prob = self.indiv_log_prob[-self.max_samples :, :]
        self.samples = self.samples[-self.max_samples :]

    def build_convergence_plot(self, plot_indiv_figsize=(5.0, 5.0)):
        self.fig, self.axes = plt.subplots(
            2, 1, figsize=plot_indiv_figsize, sharex=True
        )

        for ax in self.axes:
            ax.grid(True)

        self.axes[0].set_title("MAP convergence")
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
    def map_parameters_df(self) -> pd.DataFrame:
        theta = self.model.convert_physical_to_thetas_all_patients(
            self.map.physical_params_samples
        )
        df = self.add_unique_id(self.model.convert_theta_to_dataframe(theta))
        return df

    @property
    def map_predictions_df(self) -> pd.DataFrame:
        df = self.model.data.full_obs.to_pandas(prediction=self.map.predictions)
        return df

    @property
    def total_samples(self) -> ConditionalDistribSamples:
        """All samples, from every iteration and chain, concatenated along dim 0"""
        eta = torch.cat([s.eta_samples for s in self.samples], dim=0)  # (N, P, D)
        physical = torch.cat(
            [s.physical_params_samples for s in self.samples], dim=0
        )  # (N, P, D + F)
        pred = torch.cat([s.predictions for s in self.samples], dim=0)  # (N, Y)
        log_prob = torch.cat([s.log_prob for s in self.samples], dim=0)  # (N, P)
        return ConditionalDistribSamples(
            eta_samples=eta,
            physical_params_samples=physical,
            predictions=pred,
            log_prob=log_prob,
        )

    @property
    def total_samples_parameters_df(self) -> pd.DataFrame:
        """A data-frame with physical parameters sampled from the conditional
        distributions from every sampler iteration and chain, concatenated along dim 0
        """
        all_df = []
        for iteration, sample in enumerate(self.samples):
            theta = self.model.convert_physical_to_thetas_all_patients(
                sample.physical_params_samples
            )  # (C, P, nb_descriptors)
            df = self.model.convert_theta_to_dataframe(theta)  # C x P rows
            nb_chains = theta.shape[0]
            df["iteration"] = iteration
            df["chain"] = np.repeat(np.arange(nb_chains), self.model.nb_patients)
            all_df.append(df)
        return self.add_unique_id(pd.concat(all_df, ignore_index=True))

    def add_unique_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create  a new `id` column with unique values, store the patient id in `id_ref`.

        Each (iteration, chain, patient) triplet gets its own id when the dataframe has
        `iteration` and `chain` columns, so that samples of a same patient can be told apart.
        """
        out_df = df.rename(columns={"id": "id_ref"})
        keys = [col for col in ("iteration", "chain") if col in out_df] + ["id_ref"]
        groups = out_df.groupby(keys, sort=False).ngroup().to_numpy()
        new_ids = np.array([str(reproducible_uuid4()) for _ in range(groups.max() + 1)])
        out_df["id"] = new_ids[groups]
        return out_df


def moving_average(x: list[float], window: int = 20) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if len(x) < window:
        return x_arr
    return np.convolve(x_arr, np.ones(window) / window, mode="same")
