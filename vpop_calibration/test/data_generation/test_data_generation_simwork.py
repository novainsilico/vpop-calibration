from vpop_calibration.data_generation import (
    generate_training_data,
    generate_synthetic_data,
)
from vpop_calibration.structural_model import (
    StructuralSimwork,
    StructuralModel,
    SimworkModelBinding,
)

import pytest


@pytest.fixture
def structural_model() -> StructuralModel:
    model = SimworkModelBinding(
        path_to_model="vpop_calibration/test/simwork_model/assets/model.json",
        path_to_solving_options="vpop_calibration/test/simwork_model/assets/options.json",
        inputs=["k_12", "k_21"],
        outputs=["A0", "A1", "A2"],
    )
    struct_model = StructuralSimwork(model=model, protocol_design=None)
    return struct_model


@pytest.fixture
def param_ranges() -> dict:
    ranges = {
        "k_12": {"low": 0.0, "high": 100, "log": False},
        "k_21": {"low": -2, "high": 1, "log": True},
    }

    return ranges


def test_generate_training_data_from_simwork(structural_model, param_ranges):
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
            "k_12": {"prior": 1, "prior_omega": 0.1},
            "k_21": {"prior": 0.01, "prior_omega": 0.1},
        },
        "error_model": {
            "A0": {"error_type": "additive", "sigma": 0.5},
            "A1": {"error_type": "additive", "sigma": 0.5},
            "A2": {"error_type": "additive", "sigma": 0.5},
        },
    }
    return params


def test_generate_synthetic_data_from_simwork(structural_model, param_distribs, np_rng):
    nb_patients = 3
    time: list[float] = [0, 1, 2]
    df = generate_synthetic_data(
        struct_model=structural_model,
        param_distrib=param_distribs,
        nb_patients=nb_patients,
        time=time,
        np_rng=np_rng,
    )
