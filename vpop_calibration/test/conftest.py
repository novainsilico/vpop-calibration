import pytest
import numpy as np
import matplotlib.pyplot as plt
import pytest_golden.yaml
import torch

pytest_golden.yaml.add_representer(
    np.float64, lambda dumper, data: dumper.represent_float(float(data))
)


@pytest.fixture(scope="function")
def np_rng():
    # Initialize the seeds for all random operators used in the tests
    rng = np.random.default_rng(0)
    return rng


@pytest.fixture(autouse=True)
def clean_matplotlib_figures():
    """
    Automatically closes all matplotlib figures after each test.
    """
    yield  # Run the test

    # Teardown: Close all open figures
    plt.close("all")


@pytest.fixture(autouse=True)
def deterministic_threads():
    torch.set_num_threads(1)
    yield
