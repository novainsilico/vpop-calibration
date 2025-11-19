import math
import torch
import numpy as np
import pandas as pd
from typing import List
from functools import reduce


class TrainingDataSet:
    def __init__(
        self,
        training_df: pd.DataFrame,
        descriptors: List[str],
        training_proportion: float = 0.7,
        log_lower_limit: float = 1e-10,
        log_inputs: List[str] = [],
        log_outputs: List[str] = [],
        data_already_normalized: bool = False,
    ):
        """Instantiate a TrainingDataSet container

        The data container is used to process training data, normalize inputs and outputs, and provide utilitary methods to transform inputs and outputs for a PyTorch model. In particular, it transforms output names and protocol arms into individual tasks by cartesian product, and it provides mappings between tasks and protcol/output.

        Args:
            training_df (pd.DataFrame): the training data. Should contain the columns [`id`, `output_name`, `protocol_arm`, *descriptors, `value`]
            descriptors (List[str]): the names of the columns of `training_df` which correspond to descriptors on which to train the model
            training_proportion (float, optional): Proportion of patients to be used as training vs. validation. Defaults to 0.7.
            log_lower_limit(float): epsilon value that is added to all rescaled value to avoid numerical errors when log-scaling variables
            log_inputs (List[str]): the list of parameter inputs which should be rescaled to log when fed to the GP. Avoid adding time here, or any parameter that takes 0 as a value.
            log_outputs (list[str]): list of model outptus which should be rescaled to log
            data_already_normalized(bool): set to True if the data set is preprocessed and no normalization / scaling is required
        """

        # Process the supplied data set
        self.full_df_raw = training_df

        declared_columns = self.full_df_raw.columns.to_list()
        # Input validation
        if not ("id" in declared_columns):
            raise ValueError("Training data should contain an `id` column.")
        if not ("output_name" in declared_columns):
            raise ValueError("Training data should contain an `output_name` column.")
        if not ("value" in declared_columns):
            raise ValueError("Training data should contain a `value` column.")
        if not set(descriptors) <= set(declared_columns):
            raise ValueError(
                f"The provided inputs are not declared in the data set: {descriptors}."
            )

        self.parameter_names = descriptors
        self.training_proportion = training_proportion
        self.nb_parameters = len(self.parameter_names)
        self.data_already_normalized = data_already_normalized
        if not ("protocol_arm" in declared_columns):
            self.full_df_raw["protocol_arm"] = "identity"
        self.protocol_arms = self.full_df_raw["protocol_arm"].unique().tolist()
        self.nb_protocol_arms = len(self.protocol_arms)
        self.output_names = self.full_df_raw["output_name"].unique().tolist()
        self.nb_outputs = len(self.output_names)
        self.log_lower_limit = log_lower_limit
        self.log_inputs = log_inputs
        self.log_inputs_indices = [
            self.parameter_names.index(p) for p in self.log_inputs
        ]
        self.log_outputs = log_outputs

        # Ensure input df has a consistent shape (and remove potential extra columns)
        self.full_df_raw = self.full_df_raw[
            ["id"] + self.parameter_names + ["output_name", "protocol_arm", "value"]
        ]

        # Gather the list of patients in the training data
        self.patients = self.full_df_raw["id"].unique()
        self.nb_patients = self.patients.shape[0]

        print(
            f"Successfully loaded a training data set with {self.nb_patients} patients. The following outputs are available:\n{self.output_names}\n and the following protocol arms:\n{self.protocol_arms}"
        )
        # Construct the list of tasks, mapping from output name and protocol arm to task number
        self.tasks: List[str] = [
            output + "_" + protocol
            for protocol in self.protocol_arms
            for output in self.output_names
        ]
        self.nb_tasks = len(self.tasks)
        # Map tasks to output names
        self.task_to_output = {
            output_name + "_" + protocol_arm: output_name
            for output_name in self.output_names
            for protocol_arm in self.protocol_arms
        }
        # Map task index to output index
        self.task_idx_to_output_idx = {
            self.tasks.index(k): self.output_names.index(v)
            for k, v in self.task_to_output.items()
        }
        # Map task to protocol arm
        self.task_to_protocol = {
            output_name + "_" + protocol_arm: protocol_arm
            for output_name in self.output_names
            for protocol_arm in self.protocol_arms
        }
        # Map task index to protocol arm
        self.task_idx_to_protocol = {
            self.tasks.index(k): v for k, v in self.task_to_protocol.items()
        }
        # List tasks that should be rescaled to log
        self.log_tasks = [
            task for task in self.tasks if self.task_to_output[task] in self.log_outputs
        ]
        self.log_tasks_indices = [self.tasks.index(task) for task in self.log_tasks]

        ## Data processing
        # Pivot the data to the correct shape for GP training
        self.full_df_reshaped = self.pivot_input_data(self.full_df_raw)

        # Normalize the inputs and the outputs (only if required)
        if self.data_already_normalized == True:
            self.normalized_df = self.full_df_reshaped
        else:
            self.full_df_reshaped[self.log_inputs + self.log_tasks] = (
                self.full_df_reshaped[self.log_inputs + self.log_tasks].apply(
                    lambda val: np.log(np.maximum(val, self.log_lower_limit))
                )
            )

            self.normalized_df, mean, std = self.normalize_data(
                self.full_df_reshaped, ["id"]
            )
            self.normalizing_input_mean, self.normalizing_input_std = (
                mean.loc[self.parameter_names],
                std.loc[self.parameter_names],
            )
            self.normalizing_output_mean, self.normalizing_output_std = (
                torch.Tensor(mean.loc[self.tasks].values),
                torch.Tensor(std.loc[self.tasks].values),
            )

        # Compute the number of patients for training
        self.nb_patients_training = math.floor(
            self.training_proportion * self.nb_patients
        )
        self.nb_patients_validation = self.nb_patients - self.nb_patients_training

        if self.training_proportion != 1:  # non-empty validation data set
            if self.nb_patients_training == self.nb_patients:
                raise ValueError(
                    "Training proportion too high for the number of sets of parameters: all would be used for training. Set training_proportion as 1 if that is your intention."
                )

            # Randomly mixing up patients
            mixed_patients = np.random.permutation(self.patients)

            self.training_patients = mixed_patients[: self.nb_patients_training]
            self.validation_patients = mixed_patients[self.nb_patients_training :]

            self.training_df_normalized: pd.DataFrame = self.normalized_df.loc[
                self.normalized_df["id"].isin(self.training_patients)
            ]
            self.validation_df_normalized: pd.DataFrame = self.normalized_df.loc[
                self.normalized_df["id"].isin(self.validation_patients)
            ]
            self.X_validation = torch.Tensor(
                self.validation_df_normalized[self.parameter_names].values
            )
            self.Y_validation = torch.Tensor(
                self.validation_df_normalized[self.tasks].values
            )

        else:  # no validation data set provided
            self.training_df_normalized = self.normalized_df
            self.validation_df = None
            self.X_validation = None
            self.Y_validation = None

        self.X_training: torch.Tensor = torch.Tensor(
            self.training_df_normalized[self.parameter_names].values
        )
        self.Y_training: torch.Tensor = torch.Tensor(
            self.training_df_normalized[self.tasks].values
        )

    def pivot_input_data(self, data_in: pd.DataFrame) -> pd.DataFrame:
        """Pivot and reorder columns from a data frame to feed to the model

        This method is used at initialization on the training data frame), and when plotting the model performance against existing data.

        Args:
            data_in (pd.DataFrame): Input data frame, containing the following columns
            - `id`: patient id
            - one column per descriptor, the same descriptors as self.parameter_names should be present
            - `output_name`: the name of the output
            - `protocol_arm`: the name of the protocol arm
            - `value`: the observed value

        Returns:
            pd.DataFrame: A validated and pivotted dataframe with one column per task (`outputName_protocolArm`), and one row per observation
        """

        # util function to rename columns as `output_protocol`
        def join_if_two(tup: List[str]) -> str:
            if tup[0] == "":
                return tup[1]
            elif tup[1] == "":
                return tup[0]
            else:
                return "_".join(tup)

        # Pivot the data set
        reshaped_df = data_in.pivot(
            index=["id"] + self.parameter_names,
            columns=["output_name", "protocol_arm"],
            values="value",
        ).reset_index()
        reshaped_df.columns = list(map(join_if_two, reshaped_df.columns.to_series()))

        assert set(reshaped_df.columns) == set(
            ["id"] + self.parameter_names + self.tasks
        ), "Incomplete training data set provided."

        return reshaped_df

    def normalize_data(
        self, data_in: pd.DataFrame, ignore: List[str]
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Normalize a data frame with respect to its mean and std, ignoring certain columns."""
        selected_columns = data_in.columns.difference(ignore)
        norm_data = data_in
        mean = data_in[selected_columns].mean()
        std = data_in[selected_columns].std()
        norm_data[selected_columns] = (norm_data[selected_columns] - mean) / std
        return norm_data, mean, std

    def unnormalize_output_wide(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize wide outputs (all tasks included) from the model."""
        unnormalized = data * self.normalizing_output_std + self.normalizing_output_mean
        unnormalized[:, self.log_tasks_indices] = torch.exp(
            unnormalized[:, self.log_tasks_indices]
        )

        return unnormalized

    def unnormalize_output_long(
        self, data: torch.Tensor, task_indices: torch.LongTensor
    ) -> torch.Tensor:
        """Unnormalize long outputs (one row per task) from the model."""
        rescaled_data = data
        for task_idx in range(self.nb_tasks):
            log_task = self.tasks[task_idx] in self.log_tasks
            mask = torch.BoolTensor(task_indices == task_idx)
            rescaled_data[mask] = (
                rescaled_data[mask] * self.normalizing_output_std[task_idx]
                + self.normalizing_output_mean[task_idx]
            )
            if log_task:
                rescaled_data[mask] = torch.exp(rescaled_data[mask])
        return rescaled_data

    def normalize_inputs_df(self, inputs_df: pd.DataFrame) -> torch.Tensor:
        """Normalize new inputs provided to the model as a data frame, and convert them to a tensor."""
        selected_cols = self.normalizing_input_mean.index.tolist()
        norm_data = inputs_df[selected_cols]
        norm_data.loc[:, self.log_inputs] = norm_data.loc[:, self.log_inputs].transform(
            "log"
        )
        norm_data = (
            norm_data - self.normalizing_input_mean
        ) / self.normalizing_input_std
        return torch.Tensor(norm_data.values)

    def normalize_inputs_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize new inputs provided to the model as a tensor. The columns of the input tensor should be the same as [self.descriptors]"""
        X = inputs
        X[:, self.log_inputs_indices] = torch.log(X[:, self.log_inputs_indices])
        mean = torch.Tensor(self.normalizing_input_mean.values)
        std = torch.Tensor(self.normalizing_input_std.values)
        norm_X = (X - mean) / std

        return norm_X

    def pivot_outputs_longer(
        self, comparison_df: pd.DataFrame, Y: torch.Tensor, name: str
    ) -> pd.DataFrame:
        """Given wide outputs from a model and a comparison data frame (wide format), add the patient descriptors and reshape to a long format, with a `protocol_arm` and an `output_name` column."""
        # Assuming Y is a wide output from the model, its columns are self.tasks
        base_df = pd.DataFrame(
            data=Y.detach().float().numpy(),
            columns=self.tasks,
        )
        # The rows are assumed to correspond to the rows of the comparison data frame
        base_df[["id"] + self.parameter_names] = comparison_df[
            ["id"] + self.parameter_names
        ]
        # Pivot the data frame to a long format, separating the task names into protocol arm and output name
        long_df = (
            pd.wide_to_long(
                df=base_df,
                stubnames=self.output_names,
                i=["id"] + self.parameter_names,
                j="protocol_arm",
                sep="_",
                suffix=".*",
            )
            .reset_index()
            .melt(
                id_vars=["id"] + self.parameter_names + ["protocol_arm"],
                value_vars=self.output_names,
                var_name="output_name",
                value_name=name,
            )
        )
        return long_df

    def get_data_inputs(
        self, data_set: str | pd.DataFrame
    ) -> tuple[torch.Tensor, pd.DataFrame, pd.DataFrame, bool]:
        """Process a new data set of inputs and format them for a surrogate model to use

        The new data may be incomplete. The function expects a long data table (unpivotted). This function is under-optimized, and should not be used during training.

        Args:
            data_set (str | pd.DataFrame):
            Either "training" or "validation" OR
            An input data frame on which to predict with the GP. Should contain the following columns
            - `id`
            - one column per descriptor
            - `protocol_name`
            - `value` (Optional)

        Returns:
            torch.Tensor: the inputs to provide to a surrogate model for predicting the same values as provided in the data set
            pd.DataFrame: the processed data frame, in a wide format
            pd.DataFrame: the original data frame, in a long format
            bool: a flag, True if the value column is dummy in the output data frames
        """
        if isinstance(data_set, str):
            if data_set == "training":
                patients = self.training_patients
            elif data_set == "validation":
                patients = self.validation_patients
            else:
                raise ValueError(
                    f"Incorrect data set choice: {data_set}. Use `training` or `validation`"
                )
            new_data = self.full_df_raw.loc[self.full_df_raw["id"].isin(patients)]
        elif isinstance(data_set, pd.DataFrame):
            new_data = data_set
        else:
            raise ValueError(
                "`predict_new_data` expects either a str (`training`|`validation`) or a data frame."
            )

        # Validate the content of the new data frame
        new_columns = new_data.columns.to_list()
        if not "protocol_arm" in new_columns:
            new_protocols = ["identity"]
        else:
            new_protocols = new_data["protocol_arm"].unique().tolist()
        new_outputs = new_data["output_name"].unique().tolist()
        if not (set(new_protocols) <= set(self.protocol_arms)):
            raise ValueError(
                "Supplied data frame contains a different set of protocol arms."
            )
        if not (set(new_outputs) <= set(self.output_names)):
            raise ValueError(
                "Supplied data frame contains a different set of model outputs."
            )
        if not (set(self.parameter_names) <= set(new_columns)):
            raise ValueError(
                "All model descriptors are not supplied in the new data frame."
            )

        # Flag the case where no observed value was supplied
        remove_value = False
        if not "value" in new_columns:
            remove_value = True
            # Add a dummy `value` column
            new_data["value"] = 1.0

        wide_df = self.pivot_input_data(new_data)
        tensor_inputs_wide = self.normalize_inputs_df(wide_df)

        return tensor_inputs_wide, wide_df, new_data, remove_value

    def merge_predictions_long(
        self,
        pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        wide_df: pd.DataFrame,
        long_df: pd.DataFrame,
        remove_value: bool,
    ) -> pd.DataFrame:
        """Merge model predictions with an observation data frame

        Args:
            pred (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Predictions from a model: mean, low bound, high bound. Predictions are expected in a wide format (as many columns as tasks)
            wide_df (pd.DataFrame): The comparison data frame, in a wide format
            long_df (pd.DataFrame): The comparison data frame, in a long format
            remove_value (bool): True if the value column should be ignored

        Returns:
            pd.DataFrame: A merged data frame in a long format, identical to the initial data, with additional columns [`pred_mean`, `pred_low`, `pred_high`]
        """
        pred_mean, pred_low, pred_high = pred
        # Reshape these outputs into a long format
        mean_df = self.pivot_outputs_longer(wide_df, pred_mean, "pred_mean")
        low_df = self.pivot_outputs_longer(wide_df, pred_low, "pred_low")
        high_df = self.pivot_outputs_longer(wide_df, pred_high, "pred_high")
        # Merge the model results with the long format data frame
        out_df = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["id"] + self.parameter_names + ["protocol_arm", "output_name"],
                how="left",
            ),
            [long_df, mean_df, low_df, high_df],
        )
        # Remove the dummy value column if it was added during the data processing
        if remove_value:
            out_df = out_df.drop(columns=["value"])
        return out_df
