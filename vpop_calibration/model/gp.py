from matplotlib import pyplot as plt
import math
import torch
import gpytorch
from tqdm.notebook import tqdm
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Optional, cast

from .data import TrainingDataSet
from .plot import (
    plot_all_solutions,
    plot_individual_solution,
    plot_obs_vs_predicted,
    plot_loss,
)

torch.set_default_dtype(torch.float64)
gpytorch.settings.cholesky_jitter(1e-6)


class SVGP(gpytorch.models.ApproximateGP):
    """The internal GP class used to create surrogate models, interfacing with gpytorch's API"""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        nb_params: int,
        nb_tasks: int,
        nb_latents: int,
        var_dist: str = "Chol",
        var_strat: str = "IMV",
        kernel: str = "RBF",
        deep_kernel: bool = False,
        jitter: float = 1e-6,
        nb_mixtures: int = 4,  # only for the SMK kernel
        nb_features: int = 10,
    ):
        """_summary_

        Args:
            inducing_points (torch.Tensor): Initial choice for the inducing points
            nb_params (int): Number of input parameters
            nb_outputs (int): Number of outputs (tasks)
            var_dist (str, optional): Variational distribution choice. Defaults to "Chol".
            var_strat (str, optional): Variational strategy choice. Defaults to "IMV".
            kernel (str, optional): Kernel choice. Defaults to "RBF".
            deep_kernel (bool, optional): Add a neural network feature extractor in the kernel
            jitter (float, optional): Jitter value (for numerical stability). Defaults to 1e-6.
            nb_mixtures (int, optional): Number of mixtures for the SMK kernel. Defaults to 4.
            nb_features (int, optional): Number of features for the deep kernel. Defaults to 10.
        """
        assert var_dist == "Chol", f"Unsupported variational distribution: {var_dist}"
        if var_strat == "LMCV":
            self.batch_size = nb_latents
        elif var_strat == "IMV":
            self.batch_size = nb_tasks
        else:
            self.batch_size = nb_tasks

        self.kernel_type = kernel
        self.deep_kernel = deep_kernel
        if deep_kernel:
            self.kernel_size = nb_features
        else:
            self.kernel_size = nb_params

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.shape[0],
            batch_shape=torch.Size([self.batch_size]),
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
                    num_tasks=nb_tasks,
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
                num_tasks=nb_tasks,
                num_latents=nb_latents,
                latent_dim=-1,
            )
        else:
            raise ValueError(f"Unsupported variational strategy {var_strat}")

        super().__init__(variational_strategy)

        # Todo : allow for different mean choices
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.batch_size])
        )

        if kernel == "RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    batch_shape=torch.Size([self.batch_size]),
                    ard_num_dims=self.kernel_size,
                    jitter=jitter,
                ),
                batch_shape=torch.Size([self.batch_size]),
            )
        elif kernel == "SMK":
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
                batch_size=nb_tasks,
                num_mixtures=nb_mixtures,
                ard_num_dims=self.kernel_size,
                jitter=jitter,
            )
        elif kernel == "Matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    batch_size=nb_tasks,
                    num_mixtures=nb_mixtures,
                    ard_num_dims=self.kernel_size,
                    jitter=jitter,
                ),
                batch_shape=torch.Size([self.batch_size]),
            )
        else:
            raise ValueError(f"Unsupported kernel {kernel}")
        if deep_kernel:
            self.feature_extractor = LargeFeatureExtractor(nb_params, nb_features)
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x: torch.Tensor):
        if self.deep_kernel:
            proj_x = self.feature_extractor(x)
            proj_x = self.scale_to_bounds(proj_x)
            mean_x = cast(torch.Tensor, self.mean_module(proj_x))
            covar_x = self.covar_module(proj_x)
        else:
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
        deep_kernel: bool = True,
        nb_training_iter: int = 400,
        training_proportion: float = 0.7,
        nb_inducing_points: int = 200,
        data_already_normalized: bool = False,
        log_lower_limit: float = 1e-10,
        log_inputs: List[str] = [],
        log_outputs: List[str] = [],
        nb_latents: Optional[int] = None,
        # by default we will use nb_latents = nb_outputs
        mll: str = "ELBO",  # ELBO or PLL
        learning_rate: Optional[float] = None,  # optional
        lr_decay: Optional[float] = None,
        num_mixtures: int = 4,  # only for the SMK kernel
        nb_features: int = 10,  # only for the DK kernel
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
            log_outputs (list[str]): list of model outptus which should be rescaled to log
            log_lower_limit(float): epsilon value that is added to all rescaled value to avoid numerical errors when log-scaling variables
            nb_latents (Optional[int], optional): Number of latents. Defaults to None, implying that nb_latents = nb_tasks will be used
            mll (str, optional): Marginal log likelihood choice. Defaults to "ELBO" (other option "PLL")
            learning_rate (Optional[float]): learning rate initial value. Defaults to 0.001 (in torch.optim.Adam)
            lr_decay (Optional[float]): learning rate decay rate.
            num_mixtures (int): Number of mixtures used in the SMK kernel. Not used for other kernel choices. Default to 4.
            nb_features (int): Number of features used in the deep kernel. Not used for other kernel choices. Default to 10.
            jitter: Jitter value for numerical stability

        Comments:
            The GP will learn nb_tasks = nb_outputs * nb_protocol_arms, i.e. one predicted task per model output per protocol arm.

        """
        # Define GP parameters
        self.var_dist = var_dist
        self.var_strat = var_strat
        self.kernel = kernel
        self.deep_kernel = deep_kernel
        self.nb_training_iter = nb_training_iter
        self.nb_inducing_points = nb_inducing_points
        self.learning_rate = learning_rate
        self.mll_name = mll
        self.num_mixtures = num_mixtures
        self.nb_features = nb_features
        self.jitter = jitter
        if lr_decay is not None:
            self.lr_decay = lr_decay

        self.data = TrainingDataSet(
            training_df,
            descriptors,
            training_proportion,
            log_lower_limit,
            log_inputs,
            log_outputs,
            data_already_normalized,
        )

        if nb_latents is None:
            self.nb_latents = self.data.nb_tasks
        else:
            self.nb_latents = nb_latents
        # Create inducing points
        self.inducing_points = self.data.X_training[
            torch.randperm(self.data.X_training.shape[0])[: self.nb_inducing_points],
            :,
        ]

        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.data.nb_tasks, has_global_noise=True, has_task_noise=True
        )
        self.model = SVGP(
            inducing_points=self.inducing_points,
            nb_params=self.data.nb_parameters,
            nb_tasks=self.data.nb_tasks,
            nb_latents=self.nb_latents,
            var_dist=self.var_dist,
            var_strat=self.var_strat,
            kernel=self.kernel,
            jitter=self.jitter,
            nb_mixtures=self.num_mixtures,
            nb_features=self.nb_features,
        )

        # set the marginal log likelihood
        if self.mll_name == "ELBO":
            self.mll = VariationalELBO(
                self.likelihood, self.model, num_data=self.data.Y_training.size(0)
            )
        elif self.mll_name == "PLL":
            self.mll = PredictiveLogLikelihood(
                self.likelihood, self.model, num_data=self.data.Y_training.size(0)
            )
        else:
            raise ValueError(f"Invalid MLL choice ({self.mll}). Choose ELBO or PLL.")

    def train(
        self, mini_batching: bool = False, mini_batch_size: Optional[int] = None
    ) -> None:
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

        # keep track of the loss
        losses_list = []
        epochs = tqdm(range(self.nb_training_iter), desc="Epochs", position=0)
        with gpytorch.settings.observation_nan_policy("fill"):
            # Batch training loop
            if mini_batching:
                # set the mini_batch_size to a power of two of the total size -4
                if mini_batch_size == None:
                    power = np.maximum(
                        math.floor(math.log2(self.data.X_training.shape[0])) - 4, 1
                    )
                    self.mini_batch_size: int = math.floor(2**power)
                else:
                    self.mini_batch_size = mini_batch_size

                # prepare mini-batching
                train_dataset = TensorDataset(
                    self.data.X_training, self.data.Y_training
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.mini_batch_size,
                    shuffle=True,
                )

                # main training loop
                for _ in epochs:
                    for batch_params, batch_outputs in tqdm(
                        train_loader, desc="Batch progress", position=1, leave=False
                    ):
                        optimizer.zero_grad()  # zero gradients from previous iteration
                        output = self.model(batch_params)  # recalculate the prediction
                        loss = -cast(torch.Tensor, self.mll(output, batch_outputs))
                        loss.backward()  # compute the gradients of the parameters that can be changed
                        epochs.set_postfix({"loss": loss.item()})
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            # Full data set training loop
            else:
                for _ in epochs:
                    optimizer.zero_grad()  # zero gradients from previous iteration
                    output = self.model(
                        self.data.X_training
                    )  # calculate the prediction with current parameters
                    loss = -cast(torch.Tensor, self.mll(output, self.data.Y_training))
                    loss.backward()  # compute the gradients of the parameters that can be changed
                    losses_list.append(loss.item())
                    optimizer.step()
                    epochs.set_postfix({"loss": loss.item()})
                    if scheduler is not None:
                        scheduler.step()
            self.losses = losses_list

    def _predict_training(self, X: torch.Tensor):
        """Internal method used to predict normalized outputs on normalized inputs."""
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            pred = cast(
                gpytorch.distributions.MultitaskMultivariateNormal,
                self.likelihood(self.model(X)),
            )

        return pred.mean, pred.confidence_region

    def predict_wide_scaled(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and interval confidence values for a given input tensor (normalized inputs). This function outputs rescaled values."""
        self.model.eval()
        self.likelihood.eval()
        inputs = self.data.normalize_inputs_tensor(X)

        with torch.no_grad():
            pred = self.model(inputs)

        if self.data.data_already_normalized:
            out_mean = pred.mean
        else:
            out_mean = self.data.unnormalize_output_wide(pred.mean)
        return out_mean, pred.variance

    def predict_long_scaled(
        self, X: torch.Tensor, tasks: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict outputs from the GP in a long format (one row per task)"""

        self.model.eval()
        self.likelihood.eval()
        inputs = self.data.normalize_inputs_tensor(X)
        with torch.no_grad():
            pred = self.model(inputs, task_indices=tasks)
        if self.data.data_already_normalized:
            out_mean = pred.mean
        else:
            out_mean = self.data.unnormalize_output_long(pred.mean, task_indices=tasks)
        return out_mean, pred.variance

    def plot_loss(self) -> None:
        # plot the loss over iterations
        iterations = np.arange(1, self.nb_training_iter + 1)
        losses = np.array(self.losses)
        plot_loss(iterations, losses)

    def RMSE(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Given two tensors of same shape, compute the Root Mean Squared Error on each column (outputs)."""
        assert y1.shape == y2.shape
        # Ignore potential NaN values in the RMSE computation
        mask = (~torch.isnan(y1)) * (~torch.isnan(y2))
        rmse = torch.where(mask, torch.sqrt(torch.pow(y1 - y2, 2)), 0)
        return torch.sqrt(rmse.sum(dim=0) / mask.sum(dim=0))

    def eval_perf(self):
        """Evaluate the model performance on its training data set and validation data set (normalized inputs and ouptuts)."""

        def print_task_rmse(index, val):
            print(
                f"    Output: {self.data.output_names[self.data.task_idx_to_output_idx[index]]}, protocol: {self.data.task_idx_to_protocol[index]}, RMSE: {val:.4f}"
            )

        (
            self.Y_training_predicted_mean,
            _,
        ) = self._predict_training(self.data.X_training)
        self.RMSE_training = self.RMSE(
            self.Y_training_predicted_mean, self.data.Y_training
        )
        print("Training data set:")

        for i, err in enumerate(self.RMSE_training):
            print_task_rmse(i, err.item())

        if not (self.data.X_validation is None) and not (
            self.data.Y_validation is None
        ):
            (
                self.Y_validation_predicted_mean,
                _,
            ) = self._predict_training(self.data.X_validation)
            self.RMSE_validation = self.RMSE(
                self.Y_validation_predicted_mean, self.data.Y_validation
            )
            print("Validation data set:")
            for i, err in enumerate(self.RMSE_validation):
                print_task_rmse(i, err.item())

    def predict_new_data(self, data_set: str | pd.DataFrame) -> pd.DataFrame:
        """Process a new data set of inputs and predict using the GP

        The new data may be incomplete. The function expects a long data table (unpivotted). This function is under-optimized, and should not be used during training.

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
        # Fetch the data
        X_wide, wide_df, long_df, remove_value = self.data.get_data_inputs(data_set)

        # Simulate the model - using a wide output (all tasks predicted for each observation)
        pred = self.predict_wide_scaled(X_wide)
        out_df = self.data.merge_predictions_long(pred, wide_df, long_df, remove_value)
        return out_df

    def plot_obs_vs_predicted(self, data_set: pd.DataFrame | str, logScale=None):
        """Plots the observed vs. predicted values on the training or validation data set, or on a new data set."""

        obs_vs_pred = self.predict_new_data(data_set)
        plot_obs_vs_predicted(obs_vs_pred, logScale)

    # plot function
    def plot_individual_solution(self, patient_number: int):
        """Plot the model prediction (and confidence interval) vs. the input data for a single patient from the GP's internal data set. Can be either training or validation patient."""
        patient_index = self.data.patients[patient_number]
        input_df = self.data.full_df_raw.loc[
            self.data.full_df_raw["id"] == patient_index
        ]
        obs_vs_pred = self.predict_new_data(input_df)
        plot_individual_solution(obs_vs_pred)

    def plot_all_solutions(self, data_set: str | pd.DataFrame):
        """Plot the overlapped observations and model predictions for all patients, either on one the GP's intrinsic data sets, or on a new data set."""

        obs_vs_pred = self.predict_new_data(data_set)
        plot_all_solutions(obs_vs_pred)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, n_params, n_features):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module("linear1", torch.nn.Linear(n_params, 1000))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(1000, 500))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(500, 50))
        self.add_module("relu3", torch.nn.ReLU())
        self.add_module("linear4", torch.nn.Linear(50, n_features))
