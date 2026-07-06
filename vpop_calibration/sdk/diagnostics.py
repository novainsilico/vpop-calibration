from vpop_calibration.interface import NlmeModel
from vpop_calibration.pynlme.diagnostics import WeightedResidualsSchema

from typing import NamedTuple
import pandera.pandas as pa


class DiagnosticsConfig(NamedTuple):
    conditional_distrib: bool = True
    nb_samples: int = 100
    ebe: bool = True
    iwres: bool = True
    pwres: bool = True
    npde: bool = True


class SamplesSchema(pa.DataFrameModel):
    # Morally: this is a vpop schema
    id: str  # unique patient id
    ref_id: str  # corresponding real patient id
    descriptor_name: str
    descriptor_value: float


class DiagnosticsOutput(NamedTuple):
    iwres: pa.typing.DataFrame[WeightedResidualsSchema] | None
    pwres: pa.typing.DataFrame[WeightedResidualsSchema] | None
    npde: pa.typing.DataFrame[WeightedResidualsSchema] | None
    conditional_samples: pa.typing.DataFrame[SamplesSchema] | None
    ebe_samples: pa.typing.DataFrame[SamplesSchema] | None


def run_diagnostics(model: NlmeModel, config: DiagnosticsConfig) -> DiagnosticsOutput:
    # Run the estimation tasks
    if config.conditional_distrib:
        model.diagnostics.sample_conditional_distribution(
            nb_samples=config.nb_samples, disable_progress_bar=True
        )

    if config.ebe:
        model.diagnostics.compute_ebe()

    if config.iwres:
        model.diagnostics.compute_iwres()

    if config.pwres:
        model.diagnostics.compute_pwres()

    if config.npde:
        model.diagnostics.compute_npde()

    # Format the output
    out = DiagnosticsOutput(
        iwres=model.diagnostics.iwres,
        pwres=model.diagnostics.pwres,
        npde=model.diagnostics.npde,
        conditional_samples=None,
        ebe_samples=None,
    )
    return out
