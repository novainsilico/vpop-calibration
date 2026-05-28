import torch

from vpop_calibration.nlme_model.indexing import ObservationIndex


class StructuralModel:
    def __init__(
        self,
        parameter_names: list[str],
        output_names: list[str],
        protocol_arms: list[str],
        task_names: list[str],
    ):
        """Initialize a structural model

        Args:
            parameter_names (list[str]): _description_
            output_names (list[str]): _description_
            protocol_arms (list[str]): _description_
            tasks (list[str]): _description_
            task_idx_to_output_idx (list[str]): _description_
            task_idx_to_protocol (list[str]): _description_
        """
        self.parameter_names: list[str] = parameter_names
        self.output_names: list[str] = output_names
        self.protocol_arms: list[str] = protocol_arms
        self.task_names: list[str] = task_names

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: ObservationIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Not implemented")
