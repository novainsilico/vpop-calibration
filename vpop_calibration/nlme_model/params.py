from pydantic import BaseModel, computed_field, Field, model_validator
import numpy as np
from typing import Literal, Optional, get_args, Any

TransformFunction = Literal["log"]


def transform_param(x: float, fun: TransformFunction) -> float:
    if fun == "log":
        return np.log(x)
    else:
        raise NotImplementedError(
            f"The following transforms are currently supported: {get_args(TransformFunction)}"
        )


class ModelIntrinsicParam(BaseModel):
    prior: float = Field(ge=0)
    transform: TransformFunction = "log"

    @computed_field
    @property
    def tansformed_prior(self) -> float:
        return transform_param(self.prior, self.transform)


class Covariate(BaseModel):
    coef_name: str
    prior: float


class PatientDescriptorUnknown(BaseModel):
    prior_mean: float = Field(ge=0)
    prior_omega: float = Field(ge=0)
    covariates: Optional[dict[str, Covariate]]
    transform: TransformFunction = "log"

    @computed_field
    @property
    def transformed_prior(self) -> float:
        return transform_param(self.prior_mean, self.transform)


class ErrorModel(BaseModel):
    type: Literal["additive", "proportional"]
    sigma: float = Field(ge=0)


class MixedEffectParameters(BaseModel):
    model_intrinsic: dict[str, ModelIntrinsicParam]
    pdu: dict[str, PatientDescriptorUnknown]
    error_model: dict[str, ErrorModel]

    # Properties to be assigned after initialization
    mi_names: Optional[list[str]] = None
    pdu_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
    beta_init: Optional[list[float]] = None
    beta_names: Optional[list[str]] = None
    covariate_names: Optional[list[str]] = None

    def model_post_init(self, context: Any) -> None:
        self.mi_names = list(self.model_intrinsic.keys())
        self.pdu_names = list(self.pdu.keys())
        self.output_names = list(self.error_model.keys())

        covariate_set = set()
        self.beta_init = []
        self.beta_names = []
        for pdu_name, pdu_val in self.pdu.items():
            self.beta_names.append(pdu_name)
            self.beta_init.append(pdu_val.transformed_prior)
            if pdu_val.covariates is not None:
                for cov_name, cov_val in pdu_val.covariates.items():
                    covariate_set.add(cov_name)
                    self.beta_names.append(cov_val.coef_name)
                    self.beta_init.append(cov_val.prior)
        self.covariate_names = list(covariate_set)
