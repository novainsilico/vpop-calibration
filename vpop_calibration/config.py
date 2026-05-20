import os
import torch

if "IS_PYTEST_RUNNING" in os.environ:
    smoke_test = True
else:
    smoke_test = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
