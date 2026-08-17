from typing import NamedTuple

from vpop_calibration.pynlme.config import NlmeConfigDict
from vpop_calibration.saem.config import SaemConfigDict


class Config(NamedTuple):
    seed: int = 0
    saem: SaemConfigDict = SaemConfigDict(mode="cli")
    nlme: NlmeConfigDict = NlmeConfigDict()
