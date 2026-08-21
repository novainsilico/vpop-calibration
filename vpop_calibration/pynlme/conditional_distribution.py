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
from vpop_calibration.pynlme.fim.estimator import FimEstimator
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

    def __eq__(self, other) -> bool:
        compared_attributes = [
            "eta_samples",
            "physical_params_samples",
            "predictions",
            "log_prob",
        ]

        for elem in compared_attributes:
            torch.testing.assert_close(getattr(self, elem), getattr(other, elem))
        return True


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
        self.total_iters = 0
        self.compute_fim = False
        if self.compute_fim:
            self.fim_estimator = FimEstimator(self.model)

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
            psi=self.current_state.gaussian_params,
            log_mi=self.model.log_mi,
            surv_coeffs=self.model.surv_coeffs,
        )
        init_samples = ConditionalDistribSamples(
            eta_samples=init_etas,
            physical_params_samples=init_physical,
            predictions=self.current_state.prediction,
            log_prob=self.current_state.log_prob,
        )
        self.samples: list[ConditionalDistribSamples] = [init_samples]
        self.map: ConditionalDistribSamples = init_samples
        self.nb_improved_history: list[float] = [0]
        self.indiv_log_prob: np.ndarray = init_samples.log_prob.detach().cpu().numpy()

    def get_state_dict(self) -> dict[str, Any]:
        if hasattr(self, "map"):
            state_dict = {
                "current_state": self.current_state.get_state_dict(),
                "map": self.map.get_state_dict(),
                "last_sample": self.samples[-1].get_state_dict(),
                "has_run": True,
                "compute_fim": getattr(self, "compute_fim", False),
                "fim_burn_in": getattr(self, "fim_burn_in", 0),
                "total_iters": getattr(self, "total_iters", 0),
            }
            if getattr(self, "compute_fim", False):
                state_dict["fim_estimator"] = self.fim_estimator.get_state_dict()
        else:
            state_dict = {"has_run": False}
        return state_dict

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], model: StatisticalModel
    ) -> "ConditionalDistributionSampler":
        instance = cls(model)
        instance.compute_fim = state_dict.get("compute_fim", False)
        instance.fim_burn_in = state_dict.get("fim_burn_in", 0)
        instance.total_iters = state_dict.get("total_iters", 0)
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
            instance.indiv_log_prob = init_samples.log_prob.cpu().numpy()
            if instance.compute_fim and "fim_estimator" in state_dict:
                instance.fim_estimator = FimEstimator.from_state_dict(
                    state_dict["fim_estimator"], model
                )
        return instance

    def run_sampler(
        self,
        nb_samples: int = 100,
        compute_fim: bool | None = None,
        fim_burn_in: int = 50,
    ):
        if compute_fim is not None:
            self.compute_fim = compute_fim
            if self.compute_fim and not hasattr(self, "fim_estimator"):
                self.fim_estimator = FimEstimator(self.model)

        self.fim_burn_in = fim_burn_in

        if not hasattr(self, "map"):
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
            if self.live_plot:
                plt.close(self.fig)
        except KeyboardInterrupt:
            print("Interrupting sampling.")
            if self.live_plot:
                self.update_convergence_plot()
                plt.close(self.fig)

    def sampling_stream(self, nb_samples: int):
        for i in tqdm(range(nb_samples), disable=not self.progress_bar):
            if not hasattr(self, "total_iters"):
                self.total_iters = len(self.samples) - 1
            self.total_iters += 1
            self.current_state = mh_step(
                self.model, previous_state=self.current_state, learning_rate=0.0
            )
            if getattr(self, "compute_fim", False) and self.total_iters > getattr(
                self, "fim_burn_in", 0
            ):
                self.fim_estimator.accumulate(
                    self.current_state.gaussian_params.unsqueeze(0),
                    max_history=self.max_samples,
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
            # Clip the list of samples to keep only the last max_samples values
            self.update_map(new_samples)
            self.clip_samples()
            yield i

    def update_map(self, new_samples: ConditionalDistribSamples):
        # Assemble as mask to accept or reject new MAPs
        accept_mask = self.map.log_prob < new_samples.log_prob
        # size (nb_patients)
        new_eta = torch.where(
            accept_mask.view(-1, 1), new_samples.eta_samples, self.map.eta_samples
        )
        new_physical = torch.where(
            accept_mask.view(-1, 1),
            new_samples.physical_params_samples,
            self.map.physical_params_samples,
        )
        accept_mask_predictions = accept_mask.index_select(
            1, self.model.data.full_obs.obs_index.id.index_values
        )
        new_pred = torch.where(
            accept_mask_predictions, new_samples.predictions, self.map.predictions
        )
        new_log_prob = torch.where(accept_mask, new_samples.log_prob, self.map.log_prob)
        self.map = ConditionalDistribSamples(
            eta_samples=new_eta,
            physical_params_samples=new_physical,
            predictions=new_pred,
            log_prob=new_log_prob,
        )

        nb_improved = accept_mask.double().sum().item()
        self.nb_improved_history.append(nb_improved)
        self.indiv_log_prob = np.concat(
            (self.indiv_log_prob, new_log_prob.detach().cpu().numpy()), axis=0
        )

    def clip_samples(self):
        self.nb_improved_history = self.nb_improved_history[-self.max_samples :]
        self.indiv_log_prob = self.indiv_log_prob[-self.max_samples :, :]
        self.samples = self.samples[-self.max_samples :]

    def build_convergence_plot(self, plot_indiv_figsize=(5.0, 5.0)):
        nb_plots = 3 if self.compute_fim else 2

        figsize = (plot_indiv_figsize[0], plot_indiv_figsize[1] * (nb_plots / 2))

        self.fig, self.axes = plt.subplots(nb_plots, 1, figsize=figsize, sharex=True)

        for ax in self.axes:
            ax.grid(True)

        self.axes[0].set_title("MAP convergence")
        self.axes[0].set_ylabel("Patients improved")

        self.axes[1].set_ylabel("Individual LL")
        self.axes[-1].set_xlabel("Iteration")

        if self.compute_fim:
            self.axes[2].set_ylabel("FIM Standard Errors (Fixed Effects)")
            self.axes[2].set_yscale("log")

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
        if self.compute_fim:
            self.traces["fim_se"] = []
            self.fim_plot_indices = []
            for i, name in enumerate(self.fim_estimator.parameter_names):
                if not (
                    "omega" in name.lower()
                    or "residual" in name.lower()
                    or "sigma" in name.lower()
                ):
                    (line,) = self.axes[2].plot([], linewidth=1.5, label=name)
                    self.traces["fim_se"].append(line)
                    self.fim_plot_indices.append(i)

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

        if self.compute_fim and hasattr(self, "fim_estimator"):
            var_history = self.fim_estimator.state.variance_history
            if var_history:
                x_fim = x[-len(var_history) :]
                var_array = np.array(var_history)
                with np.errstate(invalid="ignore"):
                    se_array = np.where(var_array > 0, np.sqrt(var_array), np.nan)
                for j, param_idx in enumerate(self.fim_plot_indices):
                    self.traces["fim_se"][j].set_data(x_fim, se_array[:, param_idx])

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
        """Create  a new `id` column with unique values, store the patient id in `id_ref`.

        This function is intended to be used on dataframes before concatenating rows together.
        """
        out_df = df.rename(columns={"id": "id_ref"})
        new_ids = {
            patient: str(reproducible_uuid4()) for patient in self.model.patients
        }
        out_df["id"] = out_df["id_ref"].map(new_ids)
        return out_df


def moving_average(x: list[float], window: int = 20) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if len(x) < window:
        return x_arr
    return np.convolve(x_arr, np.ones(window) / window, mode="same")
