from vpop_calibration.pynlme.params import (
    PatientDescriptorUnknown,
    ModelIntrinsicParam,
    Constraint,
    TransformFunction,
)
from vpop_calibration.config import device, default_dtype

import torch
from typing import Callable
from scipy.special import expit
from typing import get_args
import numpy as np


def init_transform_tensors(
    param_dict: dict[str, PatientDescriptorUnknown] | dict[str, ModelIntrinsicParam],
    param_names: list[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Extract transform functions and parameters (scale and shift) into tensors for efficient gaussian parameters transformation."""

    transforms = {
        "exp": torch.tensor(
            [
                param_names.index(p_name)
                for p_name, p_content in param_dict.items()
                if p_content.constraint.transform == "log"
            ],
            device=device,
            dtype=torch.long,
        ),
        "sigmoid": torch.tensor(
            [
                param_names.index(p_name)
                for p_name, p_content in param_dict.items()
                if p_content.constraint.transform == "logit"
            ],
            device=device,
            dtype=torch.long,
        ),
    }
    scale = torch.as_tensor(
        [[[param_dict[param].constraint.scale for param in param_names]]],
        device=device,
        dtype=default_dtype,
    )

    shift = torch.as_tensor(
        [[[param_dict[param].constraint.shift for param in param_names]]],
        device=device,
        dtype=default_dtype,
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
