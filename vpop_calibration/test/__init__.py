import numpy as np
import torch

# Initialize the seeds for all random operators used in the tests
np_rng = np.random.default_rng(42)
np.random.seed(42)
torch.manual_seed(0)
saem_mi_maxfun = 1
multithreaded = False

__all__ = ["np_rng", "saem_mi_maxfun", "multithreaded"]
