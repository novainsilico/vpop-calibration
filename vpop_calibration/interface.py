import pandas as pd
from pandera.typing import DataFrame
from typing import Literal, NamedTuple

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.structural_model import StructuralModel
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics
from vpop_calibration.saem import PySaem
from vpop_calibration.pynlme.plot import (
    check_surrogate_validity_gp,
    plot_individual_map_estimates,
    plot_all_individual_map_estimates,
    plot_map_vs_posterior,
    plot_map_estimates,
    plot_weighted_residuals,
    plot_map_estimates_gof,
)


class SaemConfigDict(NamedTuple):
    mcmc_first_burn_in: int = 5
    mcmc_nb_transitions: int = 1
    nb_phase1_iterations: int = 100
    nb_phase2_iterations: int | None = None
    convergence_threshold: float = 1e-4
    patience: int = 5
    learning_rate_power: float = 0.8
    annealing_factor: float = 0.95
    init_step_size: float = 0.5  # stick to the 0.1 - 1 range
    verbose: bool = False
    optim_max_fun: int = 500
    live_plot: bool = True
    plot_frames: int = 20
    plot_columns: int = 3
    num_chains = 1


class NlmeConfigDict(NamedTuple):
    nb_chains: int = 1


class Config(NamedTuple):
    saem: SaemConfigDict = SaemConfigDict()
    nlme: NlmeConfigDict = NlmeConfigDict()


class NlmeModel:
    def __init__(
        self,
        df: pd.DataFrame,
        prior_params: dict,
        structural_model: StructuralModel,
        optim: Literal["saem"] = "saem",
        config: Config = Config(),
    ):
        obs_data = ObsData(DataFrame(df))
        nlme_params = MixedEffectParameters.model_validate(prior_params)
        self.statistical_model = StatisticalModel(
            structural_model=structural_model,
            dataset=obs_data,
            prior_params=nlme_params,
            **config.nlme._asdict(),
        )
        if optim == "saem":
            self.optimizer = PySaem(
                model=self.statistical_model, **config.saem._asdict()
            )
        else:
            raise NotImplemented
        self.diagnostics = ModelDiagnostics(self.statistical_model)

    def plot_individual_map_estimates(self, *args, **kwargs):
        plot_individual_map_estimates(self.diagnostics, *args, **kwargs)

    def plot_all_individual_map_estimates(self, *args, **kwargs):
        plot_all_individual_map_estimates(self.diagnostics, *args, **kwargs)

    def plot_map_vs_posterior(self, *args, **kwargs):
        plot_map_vs_posterior(self.diagnostics, *args, **kwargs)

    def plot_map_estimates(self, *args, **kwargs):
        plot_map_estimates(self.diagnostics, *args, **kwargs)

    def plot_weighted_residuals(self, *args, **kwargs):
        plot_weighted_residuals(self.diagnostics, *args, **kwargs)

    def plot_map_estimates_gof(self, *args, **kwargs):
        plot_map_estimates_gof(self.diagnostics, *args, **kwargs)

    def check_surrogate_validity_gp(self, *args, **kwargs):
        check_surrogate_validity_gp(self.diagnostics, *args, **kwargs)
