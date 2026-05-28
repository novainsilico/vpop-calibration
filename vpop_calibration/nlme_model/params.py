from pydantic import BaseModel, computed_field, Field, model_validator, ConfigDict
import numpy as np
from typing import Literal, Optional, get_args, Any
from typing_extensions import Self

from vpop_calibration.nlme_model.data import ObsData

TransformFunction = Literal["log", "logit"]


class Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low: float | None = None
    high: float | None = None
    transform: TransformFunction = "log"

    # Not to be user-specified:
    shift: float = 0.0
    scale: float = 1.0

    def model_post_init(self, context: Any) -> None:
        if self.low is not None:
            self.shift = self.low
        else:
            self.shift = 0.0
        if self.high is not None:
            self.transform = "logit"
            self.scale = self.high - self.shift
        else:
            self.transform = "log"
            self.scale = 1.0


def transform_param(x: float, const: Constraint) -> float:
    if const.transform == "log":
        return np.log(x - const.shift)
    elif const.transform == "logit":
        shifted_x = (x - const.shift) / const.scale
        return np.log(shifted_x / (1 - shifted_x))
    else:
        raise NotImplementedError(
            f"The following transforms are currently supported: {get_args(TransformFunction)}"
        )


class PopulationParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prior: float = Field(ge=0)
    constraint: Constraint = Constraint()

    @model_validator(mode="after")
    def check_bounds(self) -> Self:
        if self.constraint.low is not None and self.prior < self.constraint.low:
            raise ValueError("Prior value cannot be lower than lower bound.")
        if self.constraint.high is not None and self.prior > self.constraint.high:
            raise ValueError("Prior value cannot be larger than higher bound.")
        return self

    @computed_field
    @property
    def tansformed_prior(self) -> float:
        return transform_param(self.prior, self.constraint)


class ModelIntrinsicParam(PopulationParameter):
    # Model intrinsic parameters are just simple population parameters
    pass


class Covariate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coef_name: str
    prior: float


class PatientDescriptorUnknown(PopulationParameter):
    # A PDU is a PopulationParameter with an omega prior and some covariates
    prior_omega: float = Field(ge=0)
    covariates: Optional[dict[str, Covariate]]

    @computed_field
    @property
    def transformed_prior(self) -> float:
        return transform_param(self.prior, self.constraint)


ErrorType = Literal["additive", "proportional"]


class ErrorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error_type: ErrorType
    sigma: float = Field(ge=0)


class MixedEffectParameters(BaseModel):
    """Main configuration class for mixed effects parameters (population parameters)"""

    model_config = ConfigDict(extra="forbid")
    model_intrinsic: dict[str, ModelIntrinsicParam]
    pdu: dict[str, PatientDescriptorUnknown]
    pdk: list[str]
    error_model: dict[str, ErrorModel]

    # Properties to be assigned after initialization
    beta_init: list[float] = []
    beta_names: list[str] = []
    covariate_names: list[str] = []

    @computed_field
    @property
    def mi_names(self) -> list[str]:
        return list(self.model_intrinsic.keys())

    @computed_field
    @property
    def pdu_names(self) -> list[str]:
        return list(self.pdu.keys())

    @computed_field
    @property
    def output_names(self) -> list[str]:
        return list(self.error_model.keys())

    @computed_field
    @property
    def descriptors(self) -> list[str]:
        return self.mi_names + self.pdu_names + self.pdk

    def model_post_init(self, context: Any) -> None:
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

    def validate_data(self, data: ObsData) -> None:
        """Validate an observed data set against the NLME parameters.

        This effectively checks that the supplied columns contain the necessary covariates, and the output names are consistent.
        """
        descriptors_known_params = set(self.pdk + self.covariate_names)
        assert set(data.descriptors_known) == set(
            descriptors_known_params
        ), f"Discrepancy between descriptor set and data set columns. The data set informs \n{data.descriptors_known}\n The input parameters inform\n{descriptors_known_params}"

        assert set(self.output_names) == set(
            data.observed_output_names
        ), f"Discrepancy in output names. The data set contains \n{data.observed_output_names}\n The input parameters contain \n{self.output_names}"
