import torch


from vpop_calibration.structural_model.base import StructuralModel
from vpop_calibration.model.gp import GP
from vpop_calibration.pynlme.indexing import ObservationIndex


class StructuralGp(StructuralModel):
    def __init__(self, gp_model: GP):
        """Create a structural model from a GP

        Args:
            gp_model (GP): The trained GP
        """
        # list the GP parameters, except time, as it will be handled differently in the NLME model
        parameter_names = [p for p in gp_model.data.parameter_names if p != "time"]
        super().__init__(
            parameter_names=parameter_names,
            output_names=gp_model.data.output_names,
            protocol_arms=gp_model.data.protocol_arms,
            task_names=gp_model.data.tasks,
        )
        self.gp_model = gp_model
        self.training_ranges = {}
        training_samples = self.gp_model.data.full_df_raw[self.parameter_names]
        train_min = training_samples.min(axis=0)
        train_max = training_samples.max(axis=0)
        for param in self.parameter_names:
            self.training_ranges.update(
                {param: {"low": train_min[param], "high": train_max[param]}}
            )

    def simulate(
        self,
        X: torch.Tensor,
        prediction_index: ObservationIndex,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_chains, nb_patients, nb_timesteps, nb_params = X.shape

        # Simulate the GP
        X_vertical = X.view(-1, nb_params)
        out_cat, var_cat = self.gp_model.predict_wide_scaled(X_vertical)

        nb_obs_per_chain = prediction_index.id.index_values.shape[0]
        prediction_index_expanded = (
            torch.arange(num_chains).repeat_interleave(nb_obs_per_chain),
            prediction_index.id.index_values.repeat(num_chains),
            prediction_index.time.index_values.repeat(num_chains),
            prediction_index.task.index_values.repeat(num_chains),
        )
        out_wide = out_cat.view(num_chains, nb_patients, nb_timesteps, -1)
        var_wide = var_cat.view(num_chains, nb_patients, nb_timesteps, -1)

        # Retrieve the necessary rows and columns to transform into a single column tensor
        y = out_wide[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        var = var_wide[prediction_index_expanded].view(num_chains, nb_obs_per_chain)
        return y, var
