import pandas as pd
from pandera.typing import DataFrame
from typing import Literal, NamedTuple, Any
import json
import os

from vpop_calibration.structural_model import StructuralModel
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics
from vpop_calibration.pynlme.plot import PlottingUtility
from vpop_calibration.pynlme.config import NlmeConfigDict
from vpop_calibration.saem.optimizer import PySaem
from vpop_calibration.saem.config import SaemConfigDict
from vpop_calibration.pynlme.plot import PlottingUtility
from vpop_calibration.pynlme.config import NlmeConfigDict
from vpop_calibration.utils import seed_everything


class Config(NamedTuple):
    seed: int = 0
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
        seed_everything(config.seed)
        obs_data = ObsData(DataFrame(df))
        nlme_params = MixedEffectParameters.model_validate(prior_params)
        self.statistical_model = StatisticalModel(
            structural_model=structural_model,
            dataset=obs_data,
            prior_params=nlme_params,
            config=config.nlme,
        )
        if optim == "saem":
            self.optimizer = PySaem(model=self.statistical_model, config=config.saem)
        else:
            raise NotImplemented
        self.diagnostics = ModelDiagnostics(self.statistical_model)
        self.plot = PlottingUtility(self.diagnostics)

    def get_state_dict(self) -> dict[str, Any]:
        state = {
            "statistical_model": self.statistical_model.get_state_dict(),
            "optimizer": self.optimizer.get_state_dict(),
            "diagnostics": self.diagnostics.get_state_dict(),
        }
        return state

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        df: pd.DataFrame,
        structural_model: StructuralModel,
    ) -> "NlmeModel":
        obs_data = ObsData(DataFrame(df))
        instance = cls.__new__(cls)
        instance.statistical_model = StatisticalModel.from_state_dict(
            state_dict=state_dict["statistical_model"],
            dataset=obs_data,
            structural_model=structural_model,
        )
        instance.optimizer = PySaem.from_state_dict(
            state_dict["optimizer"], model=instance.statistical_model
        )
        instance.diagnostics = ModelDiagnostics.from_state_dict(
            state_dict["diagnostics"], instance.statistical_model
        )
        instance.plot = PlottingUtility(instance.diagnostics)
        return instance

    def save(self, f: str | bytes | os.PathLike):
        state_dict = self.get_state_dict()
        with open(f, "w") as file:
            json.dump(state_dict, file)

    @classmethod
    def load(
        cls,
        f: str | bytes | os.PathLike,
        df: pd.DataFrame,
        struct_model: StructuralModel,
    ) -> "NlmeModel":
        with open(f, "r") as file:
            state_dict = json.load(file)
        return NlmeModel.from_state_dict(
            state_dict=state_dict, df=df, structural_model=struct_model
        )
