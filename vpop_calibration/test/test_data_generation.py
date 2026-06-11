from vpop_calibration.data_generation import (
    generate_training_data,
    generate_synthetic_data,
)
from vpop_calibration.structural_model import StructuralAnalytical, StructuralModel

import pytest
import torch
import pandas as pd


@pytest.fixture
def structural_model() -> StructuralModel:

    def analytical_model(d, v, ka, cl, t):
        """Analytical expression of a 1 compartment PK model

        Args:
            t: time in h
            d: Dose in mg
            v: Distribution volume in mL
            ka: Absorption rate constant in mL/h
            cl: Clearance rate constant in mL/h

        Returns:
            y: Predicted concentration
        """
        ke = cl / v
        y = d * ka / (v * (ka - ke)) * (torch.exp(-ke * t) - torch.exp(-ka * t))
        return y

    protocol_design = pd.DataFrame(
        {"protocol_arm": ["dose-1", "dose-2", "dose-3"], "d": [0.1, 1, 10]}
    )

    struct_model = StructuralAnalytical(
        equations=analytical_model,
        variable_names=["concentration"],
        protocol_design=protocol_design,
    )
    return struct_model


@pytest.fixture
def param_ranges() -> dict:
    ranges = {
        "ranges": {
            "v": {"low": 0.0, "high": 100, "log": False},
            "ka": {"low": -2, "high": 1, "log": True},
            "cl": {"low": 0.0, "high": 3, "log": True},
        }
    }
    return ranges


def test_generate_training_data(structural_model, param_ranges):
    log_nb_patients = 2
    time: list[float] = [0, 1, 2]
    df = generate_training_data(
        struct_model=structural_model,
        ranges=param_ranges,
        log_nb_ind=log_nb_patients,
        time=time,
    )


@pytest.fixture
def param_distribs() -> dict:
    params = {
        "pdu": {
            "v": {"prior": 1, "prior_omega": 0.1},
            "cl": {"prior": 0.01, "prior_omega": 0.1},
        },
        "model_intrinsic": {"ka": {"prior": 0.5}},
        "pdk": [],
        "error_model": {"concentration": {"error_type": "additive", "sigma": 0.5}},
    }
    return params


def test_generate_synthetic_data(structural_model, param_distribs, np_rng):
    nb_patients = 3
    time: list[float] = [0, 1, 2]
    df = generate_synthetic_data(
        struct_model=structural_model,
        param_distrib=param_distribs,
        nb_patients=nb_patients,
        time=time,
        np_rng=np_rng,
    )
