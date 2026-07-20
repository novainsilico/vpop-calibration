import pytest
import numpy as np
import matplotlib.pyplot as plt
import pytest_golden.yaml

pytest_golden.yaml.add_representer(
    np.float64, lambda dumper, data: dumper.represent_float(float(data))
)


@pytest.fixture(scope="session")
def np_rng():
    # Initialize the seeds for all random operators used in the tests
    rng = np.random.default_rng(42)
    return rng


@pytest.fixture(autouse=True)
def clean_matplotlib_figures():
    """
    Automatically closes all matplotlib figures after each test.
    """
    yield  # Run the test

    # Teardown: Close all open figures
    plt.close("all")
