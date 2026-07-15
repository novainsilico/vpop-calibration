import os
import torch

if "IS_PYTEST_RUNNING" in os.environ:
    smoke_test = True
else:
    smoke_test = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
