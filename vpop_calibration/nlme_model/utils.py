from vpop_calibration.nlme_model.params import (
    PatientDescriptorUnknown,
    ModelIntrinsicParam,
)
from vpop_calibration.config import device

import torch
from typing import Callable


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
