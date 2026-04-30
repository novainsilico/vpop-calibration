from pydantic import BaseModel, Field
from typing import Literal


class CovariateModel(BaseModel):
    covariate: str
    coef: str
    init: float


class PduConstraint(BaseModel):
    low: float | None
    high: float | None


class PduDesc(BaseModel):
    name: str
    init_val: float = Field(ge=0)
    init_omega: float = Field(ge=0)
    covariates: list[CovariateModel] | None
    constraints: PduConstraint | None


class ModelIntrinsicDesc(BaseModel):
    name: str
    init_val: float


class ErrorModel(BaseModel):
    output: str
    init_sd: float = Field(ge=0)
    error_type: Literal["additive", "proportional"]


class NlmeModelParameters(BaseModel):
    pdu: list[PduDesc] | None
    mi: list[ModelIntrinsicDesc] | None
    error: list[ErrorModel]
