from typing import Any

import pandas as pd
import torch

from vpop_calibration.pynlme.fim.display import display_show_summary, show_table
from vpop_calibration.pynlme.fim.standard_error import (
    compute_relative_standard_errors,
    compute_standard_errors,
    invert_fim,
)
from vpop_calibration.pynlme.fim.likelihood_derivation import compute_fim_components
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

    def __init__(self, model: StatisticalModel) -> None:
        self.model = model
        nb_params = len(model.flat_parameter_names)
        self.state = FimState.initialize(
            nb_params=nb_params,
            nb_burnin=model.config.fim_burn_in,
            history_size=self.model.config.max_samples,
            learning_rate_decay_exponent=model.config.fim_accumulation_decay_power,
        )

    @property
    def parameter_names(self) -> list[str]:
        return self.model.flat_parameter_names

    # --- Accumulation
    def accumulate(self, etas: torch.Tensor) -> None:
        statistics = compute_fim_components(
            model=self.model,
            theta_flat=self.model.flatten(),
            etas=etas,
            eps=self.model.config.fim_finite_differences_eps,
        )
        self.state = self.state.accumulate(statistics)
        return None

    # --- Results
    @property
    def fim(self) -> torch.Tensor | None:
        """Observed Fisher Information Matrix, as given by Louis' formula."""
        return self.state.fim

    @property
    def covariance_matrix(self) -> torch.Tensor | None:
        if self.fim is None:
            return None
        else:
            return invert_fim(self.fim)

    @property
    def standard_errors(self) -> torch.Tensor | None:
        if self.covariance_matrix is None:
            return None
        else:
            return compute_standard_errors(
                covariance_matrix=self.covariance_matrix,
            )

    @property
    def rse(self) -> torch.Tensor | None:
        """Relative Standard Error (RSE), in percent."""
        if self.standard_errors is None:
            return None
        else:
            return compute_relative_standard_errors(
                standard_errors=self.standard_errors,
                estimates=self.model.flatten(),
            )

    # --- Tables
    def get_history_df(self) -> pd.DataFrame:
        return history_dataframe(self.state.fim_diagonal_history, self.parameter_names)

    def get_fim_df(self) -> pd.DataFrame | None:
        if self.fim is None:
            return None
        else:
            return matrix_dataframe(self.fim, self.parameter_names)

    def get_covariance_df(self) -> pd.DataFrame | None:
        if self.covariance_matrix is None:
            return None
        else:
            return matrix_dataframe(self.covariance_matrix, self.parameter_names)

    def get_rse_df(self) -> pd.DataFrame | None:
        if self.standard_errors is None or self.rse is None:
            return None
        else:
            return rse_dataframe(
                estimates=self.model.flatten(),
                standard_errors=self.standard_errors,
                relative_standard_errors=self.rse,
                names=self.parameter_names,
            )

    def get_summary_df(self) -> pd.DataFrame | None:
        if self.standard_errors is None or self.rse is None:
            return None
        else:
            return summary_dataframe(
                estimates=self.model.flatten(),
                standard_errors=self.standard_errors,
                relative_standard_errors=self.rse,
                names=self.parameter_names,
            )

    # --- Display
    def show_fim(self) -> None:
        """Display the FIM, and return it as a DataFrame."""
        df = self.get_fim_df()
        if df is None:
            print("Burn in phase not finished.")
        else:
            show_table(df, "Fisher Information Matrix (FIM) :")

    def show_covariance_matrix(self) -> None:
        """Display the covariance matrix, and return it as a DataFrame."""
        df = self.get_covariance_df()
        if df is None:
            print("Burn in phase not finished.")
        else:
            show_table(df, "Covariance Matrix :")

    def show_rse(self) -> None:
        """Display estimates, standard errors and RSE, and return them as a DataFrame."""

        df = self.get_rse_df()
        if df is None:
            print("Burn in phase not finished.")
        else:
            show_table(df, "Standard Error:")

    def show_summary(self) -> None:
        """Display the summary of important parameters."""
        df = self.get_summary_df()
        if df is None:
            print("Burn in phase not finished.")
        else:
            display_show_summary(df, "Population parameters Summary::")

    def get_state_dict(self) -> dict[str, Any]:
        return {"state": self.state.get_state_dict()}

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], model: StatisticalModel
    ) -> "FimEstimator":
        instance = cls(model=model)
        instance.state = FimState.from_state_dict(state_dict=state_dict["state"])
        return instance
