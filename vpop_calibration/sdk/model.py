import pandas as pd
import json
from pandera.typing import DataFrame
from typing import Any

from vpop_calibration.pynlme.data import ObsData
from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
    StructuralSimwork,
)
from vpop_calibration.interface import Config
from vpop_calibration.pynlme.params import MixedEffectParameters
from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.saem.optimizer import PySaem
from vpop_calibration.pynlme.diagnostics import ModelDiagnostics


class NlmeInterface:
    def __init__(
        self,
        data_table: pd.DataFrame,
        user_input: dict,
        config: Config,
        simwork_model: SimworkModelBinding,
        protocol_design: pd.DataFrame | None,
        categorical_attributes: pd.DataFrame | None,
    ):
        config = config._replace(saem=config.saem._replace(mode="cli"))
        structural_model = StructuralSimwork(
            model=simwork_model,
            protocol_design=protocol_design,
            categorical_attributes=categorical_attributes,
        )

        obs_data = ObsData(DataFrame(data_table))
        nlme_params = MixedEffectParameters.model_validate(user_input)
        self.statistical_model = StatisticalModel(
            structural_model=structural_model,
            dataset=obs_data,
            prior_params=nlme_params,
            config=config.nlme,
        )
        self.optimizer = PySaem(model=self.statistical_model, config=config.saem)
        self.diagnostics = ModelDiagnostics(self.statistical_model)

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
        structural_model: StructuralSimwork,
    ) -> "NlmeInterface":
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
        return instance


def create_nlme_interface(
    data_table: pd.DataFrame,
    user_input: dict,
    config: Config,
    model_path: str,
    solving_options_path: str,
    protocol_design: pd.DataFrame | None,
    struct_model_inputs: list[str],
    struct_model_outputs: list[str],
    categorical_attributes: pd.DataFrame | None,
) -> NlmeInterface:

    simwork_model_binding = SimworkModelBinding(
        path_to_model=model_path,
        path_to_solving_options=solving_options_path,
        inputs=struct_model_inputs,
        outputs=struct_model_outputs,
    )

    nlme_interface = NlmeInterface(
        data_table=data_table,
        user_input=user_input,
        config=config,
        simwork_model=simwork_model_binding,
        protocol_design=protocol_design,
        categorical_attributes=categorical_attributes,
    )

    return nlme_interface


def export_nlme_model(model: NlmeInterface) -> str:
    state_dict = model.get_state_dict()
    payload = json.dumps(state_dict)
    return payload


def load_nlme_model(
    payload: str,
    data_table: pd.DataFrame,
    model_path: str,
    solving_options_path: str,
    protocol_design: pd.DataFrame | None,
    struct_model_inputs: list[str],
    struct_model_outputs: list[str],
    categorical_attributes: pd.DataFrame | None,
):

    # Override the output mode to ensure no plots or progress bars are shown
    simwork_model_binding = SimworkModelBinding(
        path_to_model=model_path,
        path_to_solving_options=solving_options_path,
        inputs=struct_model_inputs,
        outputs=struct_model_outputs,
    )

    structural_model = StructuralSimwork(
        model=simwork_model_binding,
        protocol_design=protocol_design,
        categorical_attributes=categorical_attributes,
    )

    state_dict = json.loads(payload)
    nlme_model = NlmeInterface.from_state_dict(
        df=data_table,
        state_dict=state_dict,
        structural_model=structural_model,
    )

    return nlme_model
