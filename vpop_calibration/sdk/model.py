import pandas as pd

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
