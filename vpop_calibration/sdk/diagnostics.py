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
    id: str = pa.Field(unique=True)  # unique patient id
    id_ref: str  # corresponding real patient id


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

    full_samples = SamplesSchema.validate(
        model.diagnostics.sampler.total_samples_parameters_df
    )

    ebe_samples = SamplesSchema.validate(model.diagnostics.sampler.ebe_parameters_df)
    # Format the output
    out = DiagnosticsOutput(
        iwres=model.diagnostics.iwres,
        pwres=model.diagnostics.pwres,
        npde=model.diagnostics.npde,
        conditional_samples=full_samples,
        ebe_samples=ebe_samples,
    )
    return out
