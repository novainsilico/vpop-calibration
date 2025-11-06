import torch

from gp import GP


class StructuralModel:
    def __init__(
        self,
        parameter_names,
        output_names,
        protocol_arms,
        tasks,
        task_idx_to_output_idx,
        task_idx_to_protocol,
    ):
        """Initialize a structural model

        Args:
            parameter_names (_type_): _description_
            output_names (_type_): _description_
            protocol_arms (_type_): _description_
            tasks (_type_): _description_
            task_idx_to_output_idx (_type_): _description_
            task_idx_to_protocol (_type_): _description_
        """
        self.parameter_names = parameter_names
        self.nb_parameters = len(self.parameter_names)
        self.output_names = output_names
        self.nb_outputs = len(self.output_names)
        self.protocols = protocol_arms
        self.tasks = tasks
        self.nb_protocols = len(self.protocols)
        self.task_idx_to_output_idx = task_idx_to_output_idx
        self.task_idx_to_protocol = task_idx_to_protocol

    def simulate(self, X: torch.Tensor, task_list: torch.LongTensor) -> torch.Tensor:
        raise ValueError("Not implemented")


class StructuralGp(StructuralModel):
    def __init__(self, gp_model: GP):
        """Create a structural model from a GP

        Args:
            gp_model (GP): The trained GP
        """
        # list the GP parameters, except time, as it will be handled differently in the NLME model
        parameter_names = [p for p in gp_model.parameter_names if p != "time"]
        super().__init__(
            parameter_names,
            gp_model.output_names,
            gp_model.protocol_arms,
            gp_model.tasks,
            gp_model.task_idx_to_output_idx,
            gp_model.task_idx_to_protocol,
        )
        self.gp_model = gp_model

    def simulate(self, X: torch.Tensor, task_list: torch.LongTensor) -> torch.Tensor:
        # X must be ordered like parameter names from the GP
        pred, _ = self.gp_model.predict_long_scaled(X, task_list)
        return pred
