from collections.abc import Iterable
from typing import Any

import pandas as pd
import torch
from tqdm.notebook import tqdm

from vpop_calibration.pynlme.fim.display import display_show_summary, show_table
from vpop_calibration.pynlme.fim.standard_error import (
    compute_relative_standard_errors,
    compute_standard_errors,
    invert_fim,
)
from vpop_calibration.pynlme.fim.likelihood_derivation import compute_fim_components
from vpop_calibration.pynlme.fim.parametrization import flatten, get_parameter_names
from vpop_calibration.pynlme.fim.state import FimState
from vpop_calibration.pynlme.fim.utils import (
    history_dataframe,
    matrix_dataframe,
    rse_dataframe,
    summary_dataframe,
)
from vpop_calibration.pynlme.model import StatisticalModel


class FimEstimator:
    """Stochastic approximation of the observed Fisher Information Matrix."""

    def __init__(self, model: StatisticalModel, state: FimState | None = None) -> None:
        self.model = model
        nb_params = len(get_parameter_names(model))
        self.state = (
            state if state is not None else FimState.initialize(nb_params=nb_params)
        )

    # --- Parameters
    @property
    def parameter_names(self) -> list[str]:
        return get_parameter_names(self.model)

    # --- Accumulation
    def accumulate(
        self, gaussian_params: torch.Tensor, max_history: int | None = None
    ) -> FimState:
        statistics = compute_fim_components(
            model=self.model,
            flat=flatten(self.model),
            gaussian_params=gaussian_params,
        )
        self.state = self.state.accumulate(
            statistics, nb_new=gaussian_params.shape[0], max_history=max_history
        )
        return self.state

    def accumulate_stream(
        self,
        gaussian_params_stream: Iterable[torch.Tensor],
        progress_bar: bool = True,
    ) -> "FimEstimator":
        for gaussian_params in tqdm(
            gaussian_params_stream, disable=not progress_bar, desc="FIM"
        ):
            self.accumulate(gaussian_params)
        return self

    @classmethod
    def from_samples(
        cls,
        model: StatisticalModel,
        gaussian_params_stream: Iterable[torch.Tensor],
        progress_bar: bool = True,
    ) -> "FimEstimator":
        """Build an estimator from a stream of posterior samples."""
        return cls(model).accumulate_stream(
            gaussian_params_stream, progress_bar=progress_bar
        )

    # --- Results
    @property
    def fim(self) -> torch.Tensor:
        """Observed Fisher Information Matrix, as given by Louis' formula."""
        self._check_has_samples()
        return self.state.fim

    @property
    def covariance_matrix(self) -> torch.Tensor:
        return invert_fim(self.fim)

    @property
    def standard_errors(self) -> torch.Tensor:
        return compute_standard_errors(
            covariance_matrix=self.covariance_matrix,
        )

    @property
    def rse(self) -> torch.Tensor:
        """Relative Standard Error (RSE), in percent."""
        return compute_relative_standard_errors(
            standard_errors=self.standard_errors,
            estimates=flatten(self.model),
        )

    def _check_has_samples(self) -> None:
        if self.state.nb_samples == 0:
            raise RuntimeError(
                "No posterior sample accumulated yet: the FIM is undefined. "
            )

    # --- Tables
    def get_history_df(self) -> pd.DataFrame:
        return history_dataframe(self.state.variance_history, self.parameter_names)

    def get_fim_df(self) -> pd.DataFrame:
        return matrix_dataframe(self.fim, self.parameter_names)

    def get_covariance_df(self) -> pd.DataFrame:
        return matrix_dataframe(self.covariance_matrix, self.parameter_names)

    def get_rse_df(self) -> pd.DataFrame:
        return rse_dataframe(
            estimates=flatten(self.model),
            standard_errors=self.standard_errors,
            relative_standard_errors=self.rse,
            names=self.parameter_names,
        )

    def get_summary_df(self) -> pd.DataFrame:
        return summary_dataframe(
            estimates=flatten(self.model),
            standard_errors=self.standard_errors,
            relative_standard_errors=self.rse,
            names=self.parameter_names,
        )

    # --- Display
    def show_fim(self) -> None:
        """Display the FIM, and return it as a DataFrame."""
        show_table(self.get_fim_df(), "Fisher Information Matrix (FIM) :")

    def show_covariance_matrix(self) -> None:
        """Display the covariance matrix, and return it as a DataFrame."""
        show_table(self.get_covariance_df(), "Covariance Matrix :")

    def show_rse(self) -> None:
        """Display estimates, standard errors and RSE, and return them as a DataFrame."""
        display_show_summary(self.get_rse_df(), "Standard Error:")

    def show_summary(self) -> None:
        """Display the summary of important parameters (fixed effects)."""
        display_show_summary(self.get_summary_df(), "Fixed Effects Summary:")

    def get_state_dict(self) -> dict[str, Any]:
        return self.state.get_state_dict()

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], model: StatisticalModel
    ) -> "FimEstimator":
        return cls(model=model, state=FimState.from_state_dict(state_dict))
