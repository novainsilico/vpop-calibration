from matplotlib import pyplot as plt
import math
import torch
import gpytorch
from tqdm import tqdm
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Optional, cast
from functools import reduce

torch.set_default_dtype(torch.float64)
gpytorch.settings.cholesky_jitter(1e-6)


class SVGP(gpytorch.models.ApproximateGP):
    """The internal GP class used to create surrogate models, interfacing with gpytorch's API"""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        nb_params: int,
        nb_outputs: int,
        var_dist: str = "Chol",
        var_strat: str = "IMV",
        kernel: str = "RBF",
        jitter: float = 1e-6,
        nb_mixtures: int = 4,  # only for the SMK kernel
    ):
        """_summary_

        Args:
            inducing_points (torch.Tensor): Initial choice for the inducing points
            nb_params (int): Number of input parameters
            nb_outputs (int): Number of outputs (tasks)
            var_dist (str, optional): Variational distribution choice. Defaults to "Chol".
            var_strat (str, optional): Variational strategy choice. Defaults to "IMV".
            kernel (str, optional): Kernel choice. Defaults to "RBF".
            jitter (float, optional): Jitter value (for numerical stability). Defaults to 1e-6.
            nb_mixtures (int, optional): Number of mixtures for the SMK kernel. Defaults to 4.
        """
        assert var_dist == "Chol", f"Unsupported variational distribution: {var_dist}"
        assert var_strat in [
            "IMV",
            "LMCV",
        ], f"Unsupported variational strategy {var_strat}"
        assert kernel in ["RBF", "SMK"], f"Unsupported kernel {kernel}"

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.shape[0],
            batch_shape=torch.Size([nb_outputs]),
            mean_init_std=1e-3,
        )

        if var_strat == "IMV":
            variational_strategy = (
                gpytorch.variational.IndependentMultitaskVariationalStrategy(
                    gpytorch.variational.VariationalStrategy(
                        self,
                        inducing_points,
                        variational_distribution,
                        learn_inducing_locations=True,
                        jitter_val=jitter,
                    ),
                    num_tasks=nb_outputs,
                )
            )
        elif var_strat == "LMCV":
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self,
                    inducing_points,
                    variational_distribution,
                    learn_inducing_locations=True,
                    jitter_val=jitter,
                ),
                num_tasks=nb_outputs,
                num_latents=nb_outputs,
                latent_dim=-1,
            )
        else:
            raise ValueError(f"Unsupported variational strategy {var_strat}")

        super().__init__(variational_strategy)

        # Todo : allow for different mean choices
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([nb_outputs])
        )

        if kernel == "RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([nb_outputs]),
                    ard_num_dims=nb_params,
                    jitter=jitter,
                ),
                batch_shape=torch.Size([nb_outputs]),
            )
        elif kernel == "SMK":
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
                batch_size=nb_outputs,
                num_mixtures=nb_mixtures,
                ard_num_dims=nb_params,
                jitter=jitter,
            )
        else:
            raise ValueError(f"Unsupported kernel {kernel}")

    def forward(self, x: torch.Tensor):
        mean_x = cast(torch.Tensor, self.mean_module(x))
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    """GP surrogate model"""

    def __init__(
        self,
        training_df: pd.DataFrame,
        descriptors: List[str],
        var_dist: str = "Chol",  # only Cholesky currently supported
        var_strat: str = "IMV",  # either IMV (Independent Multitask Variational) or LMCV (Linear Model of Coregionalization Variational)
        kernel: str = "RBF",  # RBF or SMK
        data_already_normalized: bool = False,
        nb_training_iter: int = 400,
        training_proportion: float = 0.7,
        nb_inducing_points: int = 200,
        log_inputs: List[str] = [],
        nb_latents: Optional[int] = None,
        # by default we will use nb_latents = nb_outputs
        mll: str = "ELBO",  # ELBO or PLL
        learning_rate: Optional[float] = None,  # optional
        lr_decay: Optional[float] = None,
        num_mixtures: int = 4,  # only for the SMK kernel
        jitter: float = 1e-6,
    ):
        """Instantiate a GP model on a training data frame

        Args:
            training_df (pd.DataFrame): Training data frame containing the following columns:
              - `id`: the id of the patient, str or int
              - `descriptors`: one column per patient descriptor (including `time`, if necessary)
              - `output_name`: the name of simulated model output
              - `value`: the simulated value (for a given patient, protocol arm and output name)
              - `protocol_arm` (optional): the protocol arm on which this patient was simulated. If not provided, `identity` will be used
            descriptors (List[str]): the names of the columns of `training_df` which correspond to descriptors on which to train the GP
            var_dist (str, optional): Variational distribution choice. Defaults to "Chol".
            nb_training_iter (int, optional): Number of iterations for training. Defaults to 400.
            training_proportion (float, optional): Proportion of patients to be used as training vs. validation. Defaults to 0.7.
            nb_inducing_points (int, optional): Number of inducing points to be used for variational inference. Defaults to 200.
            log_inputs (List[str]): the list of parameter inputs which should be rescaled to log when fed to the GP. Avoid adding time here, or any parameter that takes 0 as a value.
            nb_latents (Optional[int], optional): Number of latents. Defaults to None, implying that nb_latents = nb_tasks will be used
            mll (str, optional): Marginal log likelihood choice. Defaults to "ELBO" (other option "PLL")
            learning_rate (Optional[float]): learning rate initial value. Defaults to 0.001 (in torch.optim.Adam)
            lr_decay (Optional[float]): learning rate decay rate.
            num_mixtures (int): Number of mixtures used in the SMK kernel. Not used for other kernel choices. Default to 4.
            jitter: Jitter value for numerical stability

        Comments:
            The GP will learn nb_tasks = nb_outputs * nb_protocol_arms, i.e. one predicted task per model output per protocol arm.

        """
        # Define GP parameters
        self.var_dist = var_dist
        self.var_strat = var_strat
        self.kernel = kernel
        self.nb_training_iter = nb_training_iter
        self.training_proportion = training_proportion
        self.nb_inducing_points = nb_inducing_points
        self.learning_rate = learning_rate
        self.mll = mll
        self.num_mixtures = num_mixtures
        self.jitter = jitter
        if lr_decay is not None:
            self.lr_decay = lr_decay

        # Process the supplied data set
        self.full_df_raw = training_df
        declared_columns = self.full_df_raw.columns.to_list()

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
        self.nb_parameters = len(self.parameter_names)
        self.data_already_normalized = data_already_normalized
        if not ("protocol_arm" in declared_columns):
            self.full_df_raw["protocol_arm"] = "identity"
        self.protocol_arms = self.full_df_raw["protocol_arm"].unique().tolist()
        self.nb_protocol_arms = len(self.protocol_arms)
        self.output_names = self.full_df_raw["output_name"].unique().tolist()
        self.nb_outputs = len(self.output_names)
        self.log_inputs = log_inputs
        self.log_inputs_indices = [
            self.parameter_names.index(p) for p in self.log_inputs
        ]

        # Ensure input df has a consistent shape (and remove potential extra columns)
        self.full_df_raw = self.full_df_raw[
            ["id"] + self.parameter_names + ["output_name", "protocol_arm", "value"]
        ]

        # Gather the list of patients in the training data
        self.patients = self.full_df_raw["id"].unique()
        self.nb_patients = self.patients.shape[0]

        # Construct the list of tasks for the GP, mapping from output name and protocol arm to task number
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

        if nb_latents:
            self.nb_latents = nb_latents
        else:
            self.nb_latents = self.nb_tasks

        # Pivot the data to the correct shape for GP training
        self.full_df_reshaped = self.pivot_input_data(self.full_df_raw)

        # Normalize the inputs and the outputs (only if required)
        if self.data_already_normalized == True:
            self.normalized_df = self.full_df_reshaped
        else:
            self.full_df_reshaped[self.log_inputs] = self.full_df_reshaped[
                self.log_inputs
            ].apply(np.log)

            self.normalized_df, mean, std = normalize_data(
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
            self.validation_df_normalized: pd.DataFrame | None = self.normalized_df.loc[
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

        # Create inducing points
        self.inducing_points = self.X_training[
            torch.randperm(self.X_training.shape[0])[: self.nb_inducing_points],
            :,
        ]

        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.nb_tasks, has_global_noise=True, has_task_noise=True
        )
        self.model = SVGP(
            self.inducing_points,
            self.nb_parameters,
            self.nb_tasks,
            self.var_dist,
            self.var_strat,
            self.kernel,
            self.jitter,
            self.num_mixtures,
        )

    def pivot_input_data(self, data_in: pd.DataFrame) -> pd.DataFrame:
        """Pivot and reorder columns from a data frame to feed to the GP

        This method is used at initialization on the training data frame), and when plotting the GP performance against existing data.

        Args:
            data_in (pd.DataFrame): Input data frame, containing the following columns
            - `id`: patient id
            - one column per descriptor, the same descriptors as self.parameter_names should be present
            - `output_name`: the name of the output
            - `protocol_arm`: the name of the protocol arm
            - `value`: the observed value

        Returns:
            pd.DataFrame: A validated and reshaped dataframe with as many rows, and one column per task (`outputName_protocolArm`)
        """

        # util function to rename columns as `output_protocol`
        def join_if_two(tup):
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
        reshaped_df.columns = list(map(join_if_two, reshaped_df.columns))

        assert set(reshaped_df.columns) == set(
            ["id"] + self.parameter_names + self.tasks
        ), "Incomplete training data set provided."

        return reshaped_df

    def unnormalize_output_wide(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize wide outputs (all tasks included) from the GP."""
        unnormalized = data * self.normalizing_output_std + self.normalizing_output_mean

        return unnormalized

    def unnormalize_output_long(
        self, data: torch.Tensor, task_indices: torch.LongTensor
    ) -> torch.Tensor:
        """Unnormalize long outputs (one row per task) from the GP."""
        rescaled_data = data
        for task in range(self.nb_tasks):
            mask = torch.BoolTensor(task_indices == task)
            rescaled_data[mask] = (
                rescaled_data[mask] * self.normalizing_output_std[task]
                + self.normalizing_output_mean[task]
            )
        return rescaled_data

    def normalize_inputs_df(self, inputs_df: pd.DataFrame) -> torch.Tensor:
        """Normalize new inputs provided to the GP as a data frame, and convert them to a tensor."""
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
        """Notmalize new inputs provided to the GP as a tensor. The columns of the input tensor should be the same as [self.parameter_names]"""
        X = inputs
        X[:, self.log_inputs_indices] = torch.log(X[:, self.log_inputs_indices])
        mean = torch.Tensor(self.normalizing_input_mean.values)
        std = torch.Tensor(self.normalizing_input_std.values)
        norm_X = (X - mean) / std

        return norm_X

    def train(self, mini_batching=False, mini_batch_size=None):
        # TRAINING

        # set model and likelihood in training mode
        self.model.train()
        self.likelihood.train()

        # initialize the adam optimizer
        params_to_optim = [
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ]
        if self.learning_rate is None:
            optimizer = torch.optim.Adam(params_to_optim)
        else:
            optimizer = torch.optim.Adam(params_to_optim, lr=self.learning_rate)
        if hasattr(self, "lr_decay"):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_decay
            )
        else:
            scheduler = None

        # set the marginal log likelihood
        if self.mll == "ELBO":
            mll = VariationalELBO(
                self.likelihood, self.model, num_data=self.Y_training.size(0)
            )
        elif self.mll == "PLL":
            mll = PredictiveLogLikelihood(
                self.likelihood, self.model, num_data=self.Y_training.size(0)
            )
        else:
            raise ValueError(f"Invalid MLL choice ({self.mll}). Choose ELBO or PLL.")

        # keep track of the loss
        losses_list = []
        epochs = tqdm(range(self.nb_training_iter))

        # Batch training loop
        if mini_batching:
            # set the mini_batch_size to a power of two of the total size -4
            if mini_batch_size == None:
                power = math.floor(math.log2(self.X_training.shape[0])) - 4
                mini_batch_size = 2**power
            self.mini_batch_size = mini_batch_size

            # prepare mini-batching
            train_dataset = TensorDataset(self.X_training, self.Y_training)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.mini_batch_size,
                shuffle=True,
            )

            # main training loop
            for _ in epochs:
                epoch_losses = []
                for batch_params, batch_outputs in train_loader:
                    optimizer.zero_grad()  # zero gradients from previous iteration
                    output = self.model(batch_params)  # recalculate the prediction
                    loss = -cast(torch.Tensor, mll(output, batch_outputs))
                    loss.backward()  # compute the gradients of the parameters that can be changed
                    epoch_losses.append(loss.item())
                    optimizer.step()
                epoch_loss = sum(epoch_losses) / len(epoch_losses)
                epochs.set_postfix({"loss": epoch_loss})
                losses_list.append(epoch_loss)
                if scheduler is not None:
                    scheduler.step()

        # Full data set training loop
        else:
            for _ in epochs:
                optimizer.zero_grad()  # zero gradients from previous iteration
                output = self.model(
                    self.X_training
                )  # calculate the prediction with current parameters
                loss = -cast(torch.Tensor, mll(output, self.Y_training))
                loss.backward()  # compute the gradients of the parameters that can be changed
                losses_list.append(loss.item())
                optimizer.step()
                epochs.set_postfix({"loss": loss.item()})
                if scheduler is not None:
                    scheduler.step()
        self.losses = torch.tensor(losses_list)

    def predict_wide(self, X):
        """Predict mean and interval confidence values for a given input tensor (expects normalized inputs). This function outputs normalized values of tasks in a wide format."""
        # set model and likelihood in evaluation mode
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            prediction = cast(
                gpytorch.distributions.MultitaskMultivariateNormal,
                self.likelihood(self.model(X)),
            )
            return (
                prediction.mean,
                prediction.confidence_region()[0],
                prediction.confidence_region()[1],
            )

    def predict_wide_scaled(self, X):
        """Predict mean and interval confidence values for a given input tensor (normalized inputs). This function outputs rescaled values."""
        pred = self.predict_wide(X)
        if self.data_already_normalized:
            return pred
        else:
            return tuple(map(self.unnormalize_output_wide, pred))

    def predict_long_scaled(
        self, X: torch.Tensor, tasks: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict outputs from the GP in a long format (one row per task)"""

        self.model.eval()
        self.likelihood.eval()
        inputs = self.normalize_inputs_tensor(X)
        with torch.no_grad():
            pred = self.model(inputs, task_indices=tasks)
        out_mean = self.unnormalize_output_long(pred.mean, task_indices=tasks)
        return out_mean, pred.variance

    def plot_loss(self) -> None:
        # plot the loss over iterations
        iterations = torch.linspace(1, self.nb_training_iter, self.nb_training_iter)

        plt.plot(iterations, self.losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.show()

    def RMSE(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Given two tensors of same shape, compute the Root Mean Squared Error on each column (outputs)."""
        return torch.sqrt(torch.pow(y1 - y2, 2).sum(dim=0) / y1.shape[0])

    def eval_perf(self):
        """Evaluate the model performance on its training data set and validation data set (normalized inputs and ouptuts)."""
        (
            self.Y_training_predicted_mean,
            _,
            _,
        ) = self.predict_wide(self.X_training)
        self.RMSE_training = self.RMSE(self.Y_training_predicted_mean, self.Y_training)
        print(
            "Root mean squared error on training data set (for each output x scenario):"
        )
        print(self.RMSE_training.tolist())
        if self.Y_validation is not None:
            (
                self.Y_validation_predicted_mean,
                _,
                _,
            ) = self.predict_wide(self.X_validation)
            self.RMSE_validation = self.RMSE(
                self.Y_validation_predicted_mean, self.Y_validation
            )
            print(
                "Root mean squared error on validation data set (for each output x scenario):"
            )
            print(self.RMSE_validation.tolist())

    def pivot_outputs_longer(
        self, comparison_df: pd.DataFrame, Y: torch.Tensor, name: str
    ) -> pd.DataFrame:
        """Given wide outputs from the GP and a comparison data frame, add the patient descriptors and reshape to a long format, with a `protocol_arm` and an `output_name` column."""
        # Assuming Y is a wide output from the GP, its columns are self.tasks
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

    def predict_new_data(self, data_set: str | pd.DataFrame) -> pd.DataFrame:
        """Process a new data set of inputs and predict using the GP

        The new data may be incomplete. The function expects a long data table (unpivotted). This function is under-optimized, and should not be used during optimization.

        Args:
            data_set (str | pd.DataFrame):
            Either "training" or "validation" OR
            An input data frame on which to predict with the GP. Should contain the following columns
            - `id`
            - one column per descriptor
            - `protocol_name`

        Returns:
            pd.DataFrame: Same data frame as new_data, with additional columns
            - `pred_mean`
            - `pred_low`
            - `pred_high`
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

        processed_df = self.pivot_input_data(new_data)
        norm_inputs = self.normalize_inputs_df(processed_df)

        pred_mean, pred_low, pred_high = self.predict_wide_scaled(norm_inputs)
        mean_df = self.pivot_outputs_longer(processed_df, pred_mean, "pred_mean")
        low_df = self.pivot_outputs_longer(processed_df, pred_low, "pred_low")
        high_df = self.pivot_outputs_longer(processed_df, pred_high, "pred_high")
        out_df = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["id"] + self.parameter_names + ["protocol_arm", "output_name"],
                how="left",
            ),
            [new_data, mean_df, low_df, high_df],
        )
        if remove_value:
            # When no observed value was supplied, the output from `self.pivot_input_data` has an additional column `value` containing dummy values, just for pivotting and merging (serves as a non-NA flag)
            # This column needs to be removed
            out_df = out_df.drop(columns=["value"])
        return out_df

    def plot_obs_vs_predicted(self, data_set: pd.DataFrame | str, logScale=None):
        """Plots the observed vs. predicted values on the training or validation data set, or on a new data set."""

        obs_vs_pred = self.predict_new_data(data_set)
        outputs = obs_vs_pred["output_name"].unique().tolist()
        nb_outputs = len(outputs)
        protocol_arms = obs_vs_pred["protocol_arm"].unique().tolist()
        nb_protocol_arms = len(protocol_arms)
        patients = obs_vs_pred["id"].unique().tolist()

        n_cols = nb_outputs
        n_rows = nb_protocol_arms
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )

        if not logScale:
            logScale = [True] * nb_outputs

        for output_index, output_name in enumerate(outputs):
            for protocol_index, protocol_arm in enumerate(protocol_arms):
                log_viz = logScale[output_index]
                ax = axes[protocol_index, output_index]
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")
                data_to_plot = obs_vs_pred.loc[
                    (obs_vs_pred["protocol_arm"] == protocol_arm)
                    & (obs_vs_pred["output_name"] == output_name)
                ]
                for ind in patients:
                    patient_data = data_to_plot.loc[data_to_plot["id"] == ind]
                    obs_vec = patient_data["value"]
                    pred_vec = patient_data["pred_mean"]
                    ax.plot(
                        obs_vec,
                        pred_vec,
                        "o",
                        linewidth=1,
                        alpha=0.6,
                    )

                min_val = data_to_plot["value"].min().min()
                max_val = data_to_plot["value"].max().max()
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "-",
                    linewidth=1,
                    alpha=0.5,
                    color="black",
                )
                ax.fill_between(
                    [min_val, max_val],
                    [min_val / 2, max_val / 2],
                    [min_val * 2, max_val * 2],
                    linewidth=1,
                    alpha=0.25,
                    color="black",
                )
                title = f"{output_name} in {protocol_arm}"  # More descriptive title
                ax.set_title(title)
                if log_viz:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
        plt.tight_layout()
        if isinstance(data_set, str):
            plt.suptitle(f"Observed vs predicted values for the {data_set} data set")
        plt.show()

    # plot function
    def plot_individual_solution(self, patient_number):
        """Plot the model prediction (and confidence interval) vs. the input data for a single patient from the GP's internal data set. Can be either training or validation patient."""
        patient_index = self.patients[patient_number]
        input_df = self.full_df_raw.loc[self.full_df_raw["id"] == patient_index]
        obs_vs_pred = self.predict_new_data(input_df)
        ncols = self.nb_outputs
        nrows = self.nb_protocol_arms
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(9.0 * self.nb_outputs, 4.0), squeeze=False
        )

        patient_params = obs_vs_pred[self.parameter_names]
        for output_index, output_name in enumerate(self.output_names):
            for protocol_index, protocol_arm in enumerate(self.protocol_arms):
                data_to_plot = obs_vs_pred.loc[
                    (obs_vs_pred["output_name"] == output_name)
                    & (obs_vs_pred["protocol_arm"] == protocol_arm)
                ]
                time_steps = np.array(data_to_plot["time"].values)
                sorted_indices = np.argsort(time_steps)
                sorted_time_steps = time_steps[sorted_indices]
                ax = axes[protocol_index, output_index]
                ax.set_xlabel("Time")
                ax.plot(
                    sorted_time_steps,
                    data_to_plot["value"].values[sorted_indices],
                    ".-",
                    color="C0",
                    linewidth=2,
                    alpha=0.6,
                    label=output_name,
                )  # true values

                # Plot GP prediction
                ax.plot(
                    sorted_time_steps,
                    data_to_plot["pred_mean"].values[sorted_indices],
                    "-",
                    color="C3",
                    linewidth=2,
                    alpha=0.5,
                    label="GP prediction for " + output_name + " (mean)",
                )
                ax.fill_between(
                    sorted_time_steps,
                    data_to_plot["pred_low"].values[sorted_indices],
                    data_to_plot["pred_high"].values[sorted_indices],
                    alpha=0.5,
                    color="C3",
                    label="GP prediction for " + output_name + " (CI)",
                )

                ax.legend(loc="upper right")
                title = f"{output_name} in {protocol_arm} for patient {patient_number}"
                ax.set_title(title)

                param_text = "Parameters:\n"
                for name in self.parameter_names:
                    param_text += f"  {name}: {patient_params[name].values[0]:.3f}\n"  # Format to 4 decimal places

                ax.text(
                    1.02,
                    0.98,
                    param_text,
                    transform=ax.transAxes,  # Coordinate system is relative to the axis
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5, ec="k"),
                )

        plt.tight_layout()
        plt.show()

    def plot_all_solutions(self, data_set: str | pd.DataFrame):
        """Plot the overlapped observations and model predictions for all patients, either on one the GP's intrinsic data sets, or on a new data set."""

        obs_vs_pred = self.predict_new_data(data_set)
        outputs = obs_vs_pred["output_name"].unique().tolist()
        nb_outputs = len(outputs)
        protocol_arms = obs_vs_pred["protocol_arm"].unique().tolist()
        nb_protocol_arms = len(protocol_arms)
        patients = obs_vs_pred["id"].unique().tolist()

        n_cols = nb_outputs
        n_rows = nb_protocol_arms
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )

        cmap = plt.cm.get_cmap("Spectral")
        colors = cmap(np.linspace(0, 1, len(patients)))
        for output_index, output_name in enumerate(outputs):
            for protocol_index, protocol_arm in enumerate(protocol_arms):
                data_to_plot = obs_vs_pred.loc[
                    (obs_vs_pred["output_name"] == output_name)
                    & (obs_vs_pred["protocol_arm"] == protocol_arm)
                ]
                ax = axes[protocol_index, output_index]
                ax.set_xlabel("Time")
                for patient_num, patient_ind in enumerate(patients):
                    patient_data = data_to_plot.loc[data_to_plot["id"] == patient_ind]
                    time_vec = patient_data["time"].values
                    sorted_indices = np.argsort(time_vec)
                    sorted_times = time_vec[sorted_indices]
                    obs_vec = patient_data["value"].values[sorted_indices]
                    pred_vec = patient_data["pred_mean"].values[sorted_indices]
                    ax.plot(
                        sorted_times,
                        obs_vec,
                        "+",
                        color=colors[patient_num],
                        linewidth=2,
                        alpha=0.6,
                    )
                    ax.plot(
                        sorted_times,
                        pred_vec,
                        "-",
                        color=colors[patient_num],
                        linewidth=2,
                        alpha=0.5,
                    )

                title = f"{output_name} in {protocol_arm}"  # More descriptive title
                ax.set_title(title)
        if isinstance(data_set, str):
            plt.suptitle(f"Observed vs predicted values for the {data_set} data set")
        plt.tight_layout()
        plt.show()


def normalize_data(
    data_in: pd.DataFrame, ignore: List[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalize a data frame with respect to its mean and std, ignoring certain columns."""
    selected_columns = data_in.columns.difference(ignore)
    norm_data = data_in
    mean = data_in[selected_columns].mean()
    std = data_in[selected_columns].std()
    norm_data[selected_columns] = (norm_data[selected_columns] - mean) / std
    return norm_data, mean, std
