import pandas as pd
import json

from vpop_calibration.structural_model.simwork import (
    SimworkModelBinding,
    StructuralSimwork,
)
from vpop_calibration.interface import Config, NlmeModel


def create_nlme_model(
    data_table: pd.DataFrame,
    user_input: dict,
    config: Config,
    model_path: str,
    solving_options_path: str,
    protocol_design: pd.DataFrame | None,
    struct_model_inputs: list[str],
    struct_model_outputs: list[str],
    categorical_attributes: pd.DataFrame | None,
) -> NlmeModel:

    # Override the output mode to ensure no plots or progress bars are shown
    config = config._replace(saem=config.saem._replace(mode="cli"))
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

    nlme_model = NlmeModel(
        df=data_table,
        prior_params=user_input,
        structural_model=structural_model,
        config=config,
    )

    return nlme_model


def export_nlme_model(model: NlmeModel) -> str:
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
    nlme_model = NlmeModel.from_state_dict(
        df=data_table,
        state_dict=state_dict,
        structural_model=structural_model,
    )

    return nlme_model
