from vpop_calibration.interface import NlmeModel

from typing import NamedTuple
import pandera.pandas as pa


class DiagnosticsConfig(NamedTuple):
    conditional_distrib: bool = True
    nb_samples: int = 100
    ebe: bool = True
    iwres: bool = True
    pwres: bool = True
    npde: bool = True


class ResidualsSchema(pa.DataFrameModel):
    id: str
    time: float
    output_name: str
    residual: float


class SamplesSchema(pa.DataFrameModel):
    id: str  # unique patient id
    ref_id: str  # corresponding real patient id
    descriptor_name: str
    descriptor_value: float


class DiagnosticsOutput(NamedTuple):
    iwres: pa.typing.DataFrame[ResidualsSchema] | None
    pwres: pa.typing.DataFrame[ResidualsSchema] | None
    npde: pa.typing.DataFrame[ResidualsSchema] | None
    conditional_samples: pa.typing.DataFrame[SamplesSchema] | None
    ebe_samples: pa.typing.DataFrame[SamplesSchema] | None


def run_diagnostics(model: NlmeModel, config: DiagnosticsConfig):
    # Run the estimation tasks
    if config.conditional_distrib:
        model.diagnostics.sample_conditional_distribution(nb_samples=config.nb_samples)

    if config.ebe:
        model.diagnostics.compute_ebe()

    if config.iwres:
        model.diagnostics.compute_iwres()

    if config.pwres:
        model.diagnostics.compute_pwres()

    if config.npde:
        model.diagnostics.compute_npde()

    # Format the output
