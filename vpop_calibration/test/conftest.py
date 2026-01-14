import pytest
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


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
