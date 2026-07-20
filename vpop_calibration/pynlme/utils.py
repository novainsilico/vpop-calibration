from vpop_calibration.pynlme.params import (
    PatientDescriptorUnknown,
    ModelIntrinsicParam,
    Constraint,
    MixedEffectParameters,
    TransformFunction,
)
from vpop_calibration.config import device

import torch
from typing import Callable
import numpy as np
import scipy.stats as stats
from scipy.special import expit
from typing import get_args
import matplotlib.pyplot as plt


def init_transform_tensors(
    param_dict: dict[str, PatientDescriptorUnknown] | dict[str, ModelIntrinsicParam],
    param_names: list[str],
) -> tuple[dict[str, torch.LongTensor], torch.Tensor, torch.Tensor]:
    """Extract transform functions and parameters (scale and shift) into tensors for efficient gaussian parameters transformation."""

    transforms = {
        "exp": torch.LongTensor(
            torch.tensor(
                [
                    param_names.index(p_name)
                    for p_name, p_content in param_dict.items()
                    if p_content.constraint.transform == "log"
                ],
                device=device,
                dtype=torch.long,
            )
        ),
        "sigmoid": torch.LongTensor(
            torch.tensor(
                [
                    param_names.index(p_name)
                    for p_name, p_content in param_dict.items()
                    if p_content.constraint.transform == "logit"
                ],
                device=device,
                dtype=torch.long,
            )
        ),
    }
    scale = torch.Tensor(
        [[[param_dict[param].constraint.scale for param in param_names]]]
    )

    shift = torch.Tensor(
        [[[param_dict[param].constraint.shift for param in param_names]]]
    )
    return transforms, shift, scale


def init_transform_function(
    param_dict: dict[str, PatientDescriptorUnknown] | dict[str, ModelIntrinsicParam],
    param_names: list[str],
) -> Callable:

    transforms, shift, scale = init_transform_tensors(
        param_dict=param_dict, param_names=param_names
    )

    def transform(params: torch.Tensor) -> torch.Tensor:

        new_params_raw = torch.zeros_like(params, device=device)
        new_params_raw[:, :, transforms["exp"]] = torch.exp(
            params[:, :, transforms["exp"]]
        )
        new_params_raw[:, :, transforms["sigmoid"]] = torch.sigmoid(
            params[:, :, transforms["sigmoid"]]
        )
        new_params_shifted = shift + scale * new_params_raw

        return new_params_shifted

    return transform


def inverse_transform_param(phi: np.ndarray, const: Constraint) -> np.ndarray:
    if const.transform == "log":
        return np.exp(phi) + const.shift
    elif const.transform == "logit":
        return expit(phi) * const.scale + const.shift
    else:
        raise NotImplementedError(
            f"The following transforms are currently supported: {get_args(TransformFunction)}"
        )


def theoretical_pdf(
    x: np.ndarray, mu: float, omega: float, const: Constraint
) -> np.ndarray:

    pdf = np.zeros_like(x)

    if const.transform == "log":
        mask = x > const.shift
        x_valid = x[mask]

        phi = np.log(x_valid - const.shift)
        derivative = 1.0 / (x_valid - const.shift)

        pdf[mask] = stats.norm.pdf(phi, loc=mu, scale=omega) * derivative

    elif const.transform == "logit":

        mask = (x > const.shift) & (x < const.shift + const.scale)
        x_valid = x[mask]

        shifted_x = (x_valid - const.shift) / const.scale
        phi = np.log(shifted_x / (1.0 - shifted_x))
        derivative = 1.0 / (const.scale * shifted_x * (1.0 - shifted_x))

        pdf[mask] = stats.norm.pdf(phi, loc=mu, scale=omega) * derivative

    else:
        raise NotImplementedError(
            f"The following transforms are currently supported: {get_args(TransformFunction)}"
        )

    return pdf
