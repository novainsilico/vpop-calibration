import numpy as np
import torch

# Initialize the seeds for all random operators used in the tests
np_rng = np.random.default_rng(42)
np.random.seed(42)
torch.manual_seed(0)

__all__ = ["np_rng"]
