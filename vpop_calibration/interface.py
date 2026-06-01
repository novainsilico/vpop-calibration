import pandas as pd
from pandera.typing import DataFrame
from typing import Literal

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


class NlmeModel:
    def __init__(
        self,
        df: pd.DataFrame,
        prior_params: dict,
        structural_model: StructuralModel,
        optim: Literal["saem"] = "saem",
    ):
        obs_data = ObsData(DataFrame(df))
        nlme_params = MixedEffectParameters.model_validate(prior_params)
        self.statistical_model = StatisticalModel(
            structural_model=structural_model,
            dataset=obs_data,
            prior_params=nlme_params,
        )
        if optim == "saem":
            self.optimizer = PySaem(self.statistical_model)
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
