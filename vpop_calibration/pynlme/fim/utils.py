from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

""" Assembly of the results table """


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def matrix_dataframe(matrix: torch.Tensor, names: Sequence[str]) -> pd.DataFrame:
    """Square matrix indexed by parameter names."""
    return pd.DataFrame(_to_numpy(matrix), index=list(names), columns=list(names))


def history_dataframe(
    variance_history: Sequence[Sequence[float]], names: Sequence[str]
) -> pd.DataFrame:
    """Standard error history along the stochastic approximation iterations."""
    if not variance_history:
        return pd.DataFrame()

    df = pd.DataFrame(pd.DataFrame(variance_history, columns=list(names)))
    df = df.clip(lower=0) ** 0.5
    df.insert(0, "iteration", range(1, len(df) + 1))
    return df


def rse_dataframe(
    estimates: torch.Tensor,
    standard_errors: torch.Tensor,
    relative_standard_errors: torch.Tensor,
    names: Sequence[str],
) -> pd.DataFrame:
    """Estimates, standard errors and relative standard errors."""
    df = pd.DataFrame(
        {
            "Estimate": _to_numpy(estimates),
            "SE": _to_numpy(standard_errors),
            "RSE (%)": _to_numpy(relative_standard_errors),
        },
        index=list(names),
    )
    df.index.name = "Parameter"
    return df


def summary_dataframe(
    estimates: torch.Tensor,
    standard_errors: torch.Tensor,
    relative_standard_errors: torch.Tensor,
    names: list[str],
) -> pd.DataFrame:

    keep_indices = [
        i
        for i, name in enumerate(names)
        if not (
            "omega" in name.lower()
            or "residual" in name.lower()
            or "sigma" in name.lower()
        )
    ]

    filtered_names = [names[i] for i in keep_indices]
    filtered_est = estimates[keep_indices].detach().cpu().numpy()
    filtered_se = standard_errors[keep_indices].detach().cpu().numpy()
    filtered_rse = relative_standard_errors[keep_indices].detach().cpu().numpy()

    return pd.DataFrame(
        {
            "Estimate": filtered_est,
            "Standard Error": filtered_se,
            "RSE (%)": filtered_rse,
        },
        index=filtered_names,
    )
