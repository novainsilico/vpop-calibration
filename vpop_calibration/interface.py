import pandas as pd
from pandera.typing import DataFrame
from typing import Literal, NamedTuple
from functools import wraps

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.structural_model import StructuralModel
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics
from vpop_calibration.saem.optimizer import PySaem
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.pynlme.plot import PlottingUtility


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
            self.optimizer = PySaem(model=self.statistical_model, config=config.saem)
        else:
            raise NotImplemented
        self.diagnostics = ModelDiagnostics(self.statistical_model)
        self.plot = PlottingUtility(self.diagnostics)
