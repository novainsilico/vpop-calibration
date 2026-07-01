from vpop_calibration.saem.utils import cov_to_corr

import torch


def test_cov_to_corr():
    cov = torch.tensor([[2, 0], [1, 2]])
    cor = torch.tensor([[1, 0], [1 / 2, 1]])
    torch.testing.assert_close(cor, cov_to_corr(cov))
