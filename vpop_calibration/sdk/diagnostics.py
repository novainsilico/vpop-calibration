from vpop_calibration.interface import NlmeModel
from vpop_calibration.pynlme.diagnostics import WeightedResidualsSchema

from typing import NamedTuple
import pandera.pandas as pa
import pandas as pd


class DiagnosticsConfig(NamedTuple):
    conditional_distrib: bool = True
    nb_samples: int = 100
    iwres: bool = True
    pwres: bool = True
    npde: bool = True


class SamplesSchema(pa.DataFrameModel):
    # Morally: this is a vpop schema
    id: str  # unique patient id
    id_ref: str  # corresponding real patient id
    descriptor_name: str
    descriptor_value: float


def pivot_table_to_vpop_schema(df: pd.DataFrame) -> pa.typing.DataFrame[SamplesSchema]:
    pivotted = df.melt(
        id_vars=["id", "id_ref"],
        var_name="descriptor_name",
        value_name="descriptor_value",
    )
    validated = SamplesSchema.validate(pivotted)
    return validated


class DiagnosticsOutput(NamedTuple):
    iwres: pa.typing.DataFrame[WeightedResidualsSchema] | None
    pwres: pa.typing.DataFrame[WeightedResidualsSchema] | None
    npde: pa.typing.DataFrame[WeightedResidualsSchema] | None
    conditional_samples: pa.typing.DataFrame[SamplesSchema] | None
    ebe_samples: pa.typing.DataFrame[SamplesSchema] | None


def run_diagnostics(model: NlmeModel, config: DiagnosticsConfig) -> DiagnosticsOutput:
    # Run the estimation tasks
    if config.conditional_distrib:
        model.diagnostics.sample_conditional_distribution(nb_samples=config.nb_samples)

    if config.iwres:
        model.diagnostics.compute_iwres()

    if config.pwres:
        model.diagnostics.compute_pwres()

    if config.npde:
        model.diagnostics.compute_npde()

    full_samples = pivot_table_to_vpop_schema(
        model.diagnostics.sampler.total_samples_parameters_df
    )

    ebe_samples = pivot_table_to_vpop_schema(
        model.diagnostics.sampler.ebe_parameters_df
    )
    # Format the output
    out = DiagnosticsOutput(
        iwres=model.diagnostics.iwres,
        pwres=model.diagnostics.pwres,
        npde=model.diagnostics.npde,
        conditional_samples=full_samples,
        ebe_samples=ebe_samples,
    )
    return out
