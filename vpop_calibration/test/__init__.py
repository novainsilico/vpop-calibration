import numpy as np
import torch
import pandas as pd
from vpop_calibration.utils import smoke_test

# Initialize the seeds for all random operators used in the tests
torch.manual_seed(0)

print(f"Initializing test pipeline. {smoke_test=}")
